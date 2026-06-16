#!/usr/bin/env python3
# debug-asr-cossim.py: single run conformance for qwenasr.cpp.
#
# Drives qwenasr-transcribe --dump on the example wav for the C++ stage tensors,
# and the Qwen3-ASR transformers model on the same wav with forward hooks for
# the reference stage tensors, then compares the audio frontend and decoder
# prefill stage by stage and the final text. One C++ run, one reference run, no
# hand rolled reference forward. Dumps are raw f32, compared by cosine on the
# flat buffers, so axis order matches between both pipelines by construction.
#
# Reference runs CPU fp32 in this interpreter, needs transformers 4.57.6.
# Run from tests/ with the GGUF under test, eg:
#   GGML_BACKEND=CUDA0 ./debug-asr-cossim.py --model ../models/qwenasr-1.7B-BF16.gguf

import difflib
import os
import re
import subprocess
import sys

import numpy as np

TRANSCRIBE = os.environ.get("QA_TRANSCRIBE", "../build/qwenasr-transcribe")


def _arg(name, default):
    if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(name) + 1]
    return default


MODEL = _arg("--model", "../models/qwenasr-1.7B-BF16.gguf")
SIZE = "1.7B" if "1.7B" in MODEL else "0.6B"
REF_DIR = f"../checkpoints/Qwen3-ASR-{SIZE}"
EXAMPLE_WAV = "../examples/freeman.wav"
DUMP_CPP = "cpp"
DUMP_PT = "python"
FA_ARGS = ["--no-fa"] if "--no-fa" in sys.argv else []

# Audio frontend and decoder prefill stages. Each name is the dump file written
# by qwenasr-transcribe --dump on the C++ side and by the hooks on the reference
# side. The order follows the forward: mel -> stem -> windowed tower -> prefill.
STAGES = [
    ("Mel", "mel.bin"),
    ("Stem", "stem.bin"),
    ("Windowed", "windowed.bin"),
    ("Hidden", "hidden.bin"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_dump(path, data):
    import torch

    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.float32).cpu().numpy()
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)


def load_flat(path):
    return np.fromfile(path, dtype=np.float32)


def metric(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0
    d = np.abs(a - b)
    return cos, float(d.max()) if n else 0.0, float(d.mean()) if n else 0.0, n


def install_hooks(hf, dump_pt):
    # Capture the reference stages during the single transcribe forward. The
    # audio tower runs once, its input is the mel and the input of layer 0 is
    # the post positional stem. The decoder norm runs every step, so hidden is
    # taken on the prefill call where the sequence is longer than one token.
    # The top model wraps a thinker that owns the audio tower and the decoder.
    thinker = getattr(hf, "thinker", hf)
    tower = thinker.audio_tower
    state = {}

    def pre_tower(mod, args, kwargs):
        feats = args[0] if args else kwargs.get("input_features")
        save_dump(os.path.join(dump_pt, "mel.bin"), feats)

    def pre_layer0(mod, args, kwargs):
        save_dump(os.path.join(dump_pt, "stem.bin"), args[0])

    def post_tower(mod, args, kwargs, output):
        save_dump(os.path.join(dump_pt, "windowed.bin"), output.last_hidden_state)

    def post_norm(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if "hidden" not in state and h.shape[-2] > 1:
            state["hidden"] = True
            save_dump(os.path.join(dump_pt, "hidden.bin"), h.squeeze(0))

    tower.register_forward_pre_hook(pre_tower, with_kwargs=True)
    tower.layers[0].register_forward_pre_hook(pre_layer0, with_kwargs=True)
    tower.register_forward_hook(post_tower, with_kwargs=True)
    thinker.model.norm.register_forward_hook(post_norm)


def words(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def main():
    ensure_dir(DUMP_CPP)
    ensure_dir(DUMP_PT)

    out_cpp = os.path.join(DUMP_CPP, "asr-cpp.txt")
    cmd = [TRANSCRIBE, "--model", MODEL, "--file", EXAMPLE_WAV,
           "--lang", "English", "--dump", DUMP_CPP, "-o", out_cpp] + FA_ARGS
    print(f"[GGML] Cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    cpp_text = open(out_cpp, encoding="utf-8").read().strip()
    print(f"[GGML] Text: {cpp_text!r}")

    import torch
    from qwen_asr import Qwen3ASRModel

    # The reference runs in the checkpoint native bf16 by default so the stage
    # cosines compare the same arithmetic the BF16 GGUF runs. QA_REF_DTYPE=float32
    # upcasts the reference to expose the BF16 quantization drift instead.
    ref_dtype = getattr(torch, os.environ.get("QA_REF_DTYPE", "bfloat16"))

    # Both pipelines consume the exact 16k samples the C++ resampler produced,
    # dumped as pcm16.bin, so no second resampler diverges the mel inputs.
    pcm16 = load_flat(os.path.join(DUMP_CPP, "pcm16.bin"))
    asr = Qwen3ASRModel.from_pretrained(REF_DIR, dtype=ref_dtype,
                                        device_map="cpu", max_new_tokens=512)
    install_hooks(asr.model, DUMP_PT)
    res = asr.transcribe(audio=(pcm16, 16000), language="English",
                         return_time_stamps=False)
    ref_text = res[0].text.strip()
    print(f"[Python] Text: {ref_text!r}")

    ok = True
    for label, name in STAGES:
        cpp_path = os.path.join(DUMP_CPP, name)
        pt_path = os.path.join(DUMP_PT, name)
        if not (os.path.exists(cpp_path) and os.path.exists(pt_path)):
            print(f"[Cossim] {label} skipped (missing dump)")
            ok = False
            continue
        cpp_v = load_flat(cpp_path)
        pt_v = load_flat(pt_path)
        cos, mx, mn, n = metric(cpp_v, pt_v)
        tag = "" if cpp_v.size == pt_v.size else f" (cpp {cpp_v.size} pt {pt_v.size})"
        print(f"[Cossim] {label} cos {cos:.8f} max_abs {mx:.3e} n {n}{tag}")
        ok = ok and cos > 0.99

    cw, rw = words(cpp_text), words(ref_text)
    exact = 100.0 if cpp_text == ref_text else 0.0
    nexact = 100.0 if cw == rw else 0.0
    ratio = difflib.SequenceMatcher(None, cw, rw, autojunk=False).ratio()
    print(f"[Cossim] Text exact {exact:.2f}% norm_exact {nexact:.2f}% "
          f"word_ratio {ratio:.6f}")
    return ok and ratio > 0.8


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
