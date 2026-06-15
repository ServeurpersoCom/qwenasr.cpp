#!/usr/bin/env python3
"""Cossim debug: C++ qwenasr-feat vs the reference, stage by stage.

Stage 1 (mel): generates a deterministic 16 kHz signal, runs the mel frontend
on both sides and compares the [num_mel_bins, n_frames] matrices via cosine
similarity over the f32 payload. The reference is the transformers
WhisperFeatureExtractor with the exact Qwen3-ASR preprocessor config. Later
stages (conv stem, encoder) import the full qwen_asr reference from the sibling
clone at ../../Qwen3-ASR.

Run from tests/ with a python that has numpy + transformers.
"""

import os
import subprocess
import sys
import wave

import numpy as np

SR   = 16000
FEAT  = os.environ.get("QA_FEAT", "../build/qwenasr-feat")
THINKER = os.environ.get("QA_THINKER", "../build/qwenasr-thinker")
MODEL = "../models/qwenasr-0.6B-BF16.gguf"
CKPT  = "../checkpoints/Qwen3-ASR-0.6B/model.safetensors"
DUMP = "cpp"
WAV  = os.path.join(DUMP, "qwenasr-cossim-16k.wav")
WAV_LONG = os.path.join(DUMP, "qwenasr-cossim-long-16k.wav")

# Flash attention is on by default, matching transcribe and the server. Pass
# --no-fa to validate the manual F32 decoder path instead.
FA_ARGS = ["--no-fa"] if "--no-fa" in sys.argv else []


def make_signal(seconds=5.0, path=WAV):
    rng = np.random.default_rng(42)
    t   = np.arange(int(seconds * SR)) / SR
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.3 * np.sin(2 * np.pi * (300 + 200 * t) * t)
           + 0.1 * rng.standard_normal(t.shape[0]))
    sig = (sig / np.max(np.abs(sig)) * 0.9).astype(np.float32)
    pcm = (sig * 32767.0).astype(np.int16)
    os.makedirs(DUMP, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def cossim(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stage_mel():
    from transformers import WhisperFeatureExtractor

    subprocess.run([FEAT, "--file", WAV, "--dump", DUMP, "--pad-30s"], check=True)

    x   = read_wav(WAV)
    fe  = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                  hop_length=160, chunk_length=30, n_fft=400)
    ref = fe(x, sampling_rate=SR, return_tensors="np").input_features[0]
    cpp = np.fromfile(os.path.join(DUMP, "mel.bin"), dtype=np.float32).reshape(ref.shape)

    cos    = cossim(cpp, ref)
    maxabs = float(np.max(np.abs(cpp - ref)))
    print(f"[Mel] shape {cpp.shape} cosine {cos:.8f} max_abs {maxabs:.3e}")
    return cos > 0.9999



def stage_conv_stem():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    subprocess.run([FEAT, "--file", WAV, "--dump", DUMP, "--model", MODEL], check=True)

    with safe_open(CKPT, framework="pt") as f:
        g = lambda n: f.get_tensor("thinker.audio_tower." + n).float()
        w1, b1 = g("conv2d1.weight"), g("conv2d1.bias")
        w2, b2 = g("conv2d2.weight"), g("conv2d2.bias")
        w3, b3 = g("conv2d3.weight"), g("conv2d3.bias")
        wo = g("conv_out.weight")

    mel = np.fromfile(os.path.join(DUMP, "mel.bin"), dtype=np.float32).reshape(128, -1)
    x = torch.from_numpy(mel)[None, None]
    x = F.gelu(F.conv2d(x, w1, b1, stride=2, padding=1))
    x = F.gelu(F.conv2d(x, w2, b2, stride=2, padding=1))
    x = F.gelu(F.conv2d(x, w3, b3, stride=2, padding=1))
    b, c, fd, t = x.shape
    x = x.permute(0, 3, 1, 2).reshape(b, t, c * fd)
    ref = (x @ wo.t())[0].detach().numpy()
    cpp = np.fromfile(os.path.join(DUMP, "stem.bin"), dtype=np.float32).reshape(ref.shape)

    cos = cossim(cpp, ref)
    maxabs = float(np.max(np.abs(cpp - ref)))
    print(f"[Stem] shape {cpp.shape} cosine {cos:.8f} max_abs {maxabs:.3e}")
    return cos > 0.9999


def stage_encoder():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    subprocess.run([FEAT, "--file", WAV, "--dump", DUMP, "--model", MODEL,
                    "--encoder"], check=True)

    d_model, n_head = 896, 14
    head_dim = d_model // n_head
    eps = 1e-5
    p = "thinker.audio_tower."
    keys = ["self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
            "final_layer_norm.weight", "final_layer_norm.bias"]
    with safe_open(CKPT, framework="pt") as f:
        g = lambda n: f.get_tensor(p + n).float()
        layers = [{k: g(f"layers.{l}." + k) for k in keys} for l in range(18)]
        ln_post_w, ln_post_b = g("ln_post.weight"), g("ln_post.bias")
        proj1_w, proj1_b = g("proj1.weight"), g("proj1.bias")
        proj2_w, proj2_b = g("proj2.weight"), g("proj2.bias")

    stem = np.fromfile(os.path.join(DUMP, "stem.bin"), dtype=np.float32)
    stem = stem.reshape(-1, d_model)
    T = stem.shape[0]
    x = torch.from_numpy(stem)

    half = d_model // 2
    log_inc = np.log(10000.0) / (half - 1)
    inv = np.exp(-log_inc * np.arange(half))
    scaled = np.arange(T)[:, None] * inv[None, :]
    pe = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)
    x = x + torch.from_numpy(pe)

    def ln(t, w, b):
        return F.layer_norm(t, (d_model,), w, b, eps)

    for w in layers:
        r = x
        h = ln(x, w["self_attn_layer_norm.weight"], w["self_attn_layer_norm.bias"])
        q = h @ w["self_attn.q_proj.weight"].t() + w["self_attn.q_proj.bias"]
        k = h @ w["self_attn.k_proj.weight"].t() + w["self_attn.k_proj.bias"]
        v = h @ w["self_attn.v_proj.weight"].t() + w["self_attn.v_proj.bias"]
        q = q.reshape(T, n_head, head_dim).transpose(0, 1)
        k = k.reshape(T, n_head, head_dim).transpose(0, 1)
        v = v.reshape(T, n_head, head_dim).transpose(0, 1)
        scores = (q @ k.transpose(1, 2)) * (head_dim ** -0.5)
        o = scores.softmax(-1) @ v
        o = o.transpose(0, 1).reshape(T, d_model)
        o = o @ w["self_attn.out_proj.weight"].t() + w["self_attn.out_proj.bias"]
        x = r + o
        r = x
        h = ln(x, w["final_layer_norm.weight"], w["final_layer_norm.bias"])
        h = h @ w["fc1.weight"].t() + w["fc1.bias"]
        h = F.gelu(h)
        h = h @ w["fc2.weight"].t() + w["fc2.bias"]
        x = r + h

    x = ln(x, ln_post_w, ln_post_b)
    x = x @ proj1_w.t() + proj1_b
    x = F.gelu(x)
    x = x @ proj2_w.t() + proj2_b
    ref = x.detach().numpy()

    cpp = np.fromfile(os.path.join(DUMP, "encoder.bin"), dtype=np.float32).reshape(ref.shape)
    cos = cossim(cpp, ref)
    maxabs = float(np.max(np.abs(cpp - ref)))
    print(f"[Enc] shape {cpp.shape} cosine {cos:.8f} max_abs {maxabs:.3e}")
    return cos > 0.9999


def stage_windowed():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    make_signal(seconds=9.0, path=WAV_LONG)
    subprocess.run([FEAT, "--file", WAV_LONG, "--dump", DUMP, "--model", MODEL,
                    "--windowed"], check=True)

    d_model, n_head, n_mels = 896, 14, 128
    head_dim = d_model // n_head
    eps = 1e-5
    chunk_mel, window_chunks = 100, 8
    p = "thinker.audio_tower."
    keys = ["self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
            "final_layer_norm.weight", "final_layer_norm.bias"]
    with safe_open(CKPT, framework="pt") as f:
        g = lambda n: f.get_tensor(p + n).float()
        c1w, c1b = g("conv2d1.weight"), g("conv2d1.bias")
        c2w, c2b = g("conv2d2.weight"), g("conv2d2.bias")
        c3w, c3b = g("conv2d3.weight"), g("conv2d3.bias")
        cow = g("conv_out.weight")
        layers = [{k: g(f"layers.{l}." + k) for k in keys} for l in range(18)]
        ln_post_w, ln_post_b = g("ln_post.weight"), g("ln_post.bias")
        proj1_w, proj1_b = g("proj1.weight"), g("proj1.bias")
        proj2_w, proj2_b = g("proj2.weight"), g("proj2.bias")

    mel = np.fromfile(os.path.join(DUMP, "mel.bin"), dtype=np.float32).reshape(n_mels, -1)
    n_frames = mel.shape[1]
    mel_t = torch.from_numpy(mel.copy())  # [n_mels, n_frames]

    def conv_out_len(L):
        o = L
        for _ in range(3):
            o = (o - 1) // 2 + 1
        return o

    chunk_lengths = []
    off = 0
    while off < n_frames:
        L = min(chunk_mel, n_frames - off)
        chunk_lengths.append(L)
        off += L

    half = d_model // 2
    log_inc = np.log(10000.0) / (half - 1)
    inv = np.exp(-log_inc * np.arange(half))
    pe_rows = max(conv_out_len(L) for L in chunk_lengths)
    sc = np.arange(pe_rows)[:, None] * inv[None, :]
    pe = torch.from_numpy(
        np.concatenate([np.sin(sc), np.cos(sc)], axis=1).astype(np.float32))

    segs = []
    off = 0
    for L in chunk_lengths:
        x = mel_t[:, off:off + L][None, None]  # [1, 1, n_mels, L]
        x = F.gelu(F.conv2d(x, c1w, c1b, stride=2, padding=1))
        x = F.gelu(F.conv2d(x, c2w, c2b, stride=2, padding=1))
        x = F.gelu(F.conv2d(x, c3w, c3b, stride=2, padding=1))
        b, c, fd, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * fd)
        x = (x @ cow.t())[0]  # [t, d_model]
        x = x + pe[:x.shape[0]]
        segs.append(x)
        off += L
    x = torch.cat(segs, 0)  # [S, d_model]
    S = x.shape[0]

    window_aftercnn = conv_out_len(chunk_mel) * window_chunks
    mask = torch.full((S, S), float("-inf"))
    for b0 in range(0, S, window_aftercnn):
        b1 = min(b0 + window_aftercnn, S)
        mask[b0:b1, b0:b1] = 0.0

    def ln(t, w, b):
        return F.layer_norm(t, (d_model,), w, b, eps)

    for w in layers:
        r = x
        h = ln(x, w["self_attn_layer_norm.weight"], w["self_attn_layer_norm.bias"])
        q = h @ w["self_attn.q_proj.weight"].t() + w["self_attn.q_proj.bias"]
        k = h @ w["self_attn.k_proj.weight"].t() + w["self_attn.k_proj.bias"]
        v = h @ w["self_attn.v_proj.weight"].t() + w["self_attn.v_proj.bias"]
        q = q.reshape(S, n_head, head_dim).transpose(0, 1)
        k = k.reshape(S, n_head, head_dim).transpose(0, 1)
        v = v.reshape(S, n_head, head_dim).transpose(0, 1)
        scores = (q @ k.transpose(1, 2)) * (head_dim ** -0.5) + mask[None]
        o = scores.softmax(-1) @ v
        o = o.transpose(0, 1).reshape(S, d_model)
        o = o @ w["self_attn.out_proj.weight"].t() + w["self_attn.out_proj.bias"]
        x = r + o
        r = x
        h = ln(x, w["final_layer_norm.weight"], w["final_layer_norm.bias"])
        h = h @ w["fc1.weight"].t() + w["fc1.bias"]
        h = F.gelu(h)
        h = h @ w["fc2.weight"].t() + w["fc2.bias"]
        x = r + h

    x = ln(x, ln_post_w, ln_post_b)
    x = x @ proj1_w.t() + proj1_b
    x = F.gelu(x)
    x = x @ proj2_w.t() + proj2_b
    ref = x.detach().numpy()

    cpp = np.fromfile(os.path.join(DUMP, "windowed.bin"), dtype=np.float32).reshape(ref.shape)
    cos = cossim(cpp, ref)
    maxabs = float(np.max(np.abs(cpp - ref)))
    print(f"[Win] shape {cpp.shape} chunks {len(chunk_lengths)} cosine {cos:.8f} "
          f"max_abs {maxabs:.3e}")
    return cos > 0.9999


_THINKER_KEYS = ["input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


def _load_thinker():
    import torch
    from safetensors import safe_open
    p = "thinker."
    with safe_open(CKPT, framework="pt") as f:
        g = lambda n: f.get_tensor(p + n).float()
        embed = g("model.embed_tokens.weight")
        norm_w = g("model.norm.weight")
        layers = [{k: g(f"model.layers.{l}." + k) for k in _THINKER_KEYS} for l in range(28)]
    return embed, norm_w, layers


def _qwen3_decode_ref(x, layers, norm_w):
    import torch
    import torch.nn.functional as F
    hidden, n_q, n_kv, head_dim = 1024, 16, 8, 128
    eps, theta = 1e-6, 1e6
    rep = n_q // n_kv
    T = x.shape[0]

    pos = torch.arange(T).float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = pos[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], -1)
    sin = torch.cat([freqs.sin(), freqs.sin()], -1)

    def rms(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w

    def rope(t):
        z1 = t[..., :head_dim // 2]
        z2 = t[..., head_dim // 2:]
        rot = torch.cat([-z2, z1], -1)
        return t * cos[:, None, :] + rot * sin[:, None, :]

    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)

    h = x
    for w in layers:
        r = h
        hn = rms(h, w["input_layernorm.weight"])
        q = (hn @ w["self_attn.q_proj.weight"].t()).view(T, n_q, head_dim)
        k = (hn @ w["self_attn.k_proj.weight"].t()).view(T, n_kv, head_dim)
        v = (hn @ w["self_attn.v_proj.weight"].t()).view(T, n_kv, head_dim)
        q = rms(q, w["self_attn.q_norm.weight"])
        k = rms(k, w["self_attn.k_norm.weight"])
        q = rope(q)
        k = rope(k)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = (q @ k.transpose(1, 2)) / (head_dim ** 0.5) + causal[None]
        o = scores.softmax(-1) @ v
        o = o.transpose(0, 1).reshape(T, n_q * head_dim)
        o = o @ w["self_attn.o_proj.weight"].t()
        h = r + o
        r = h
        hn = rms(h, w["post_attention_layernorm.weight"])
        gate = hn @ w["mlp.gate_proj.weight"].t()
        up = hn @ w["mlp.up_proj.weight"].t()
        h = r + (F.silu(gate) * up) @ w["mlp.down_proj.weight"].t()

    h = rms(h, norm_w)
    return h.detach().numpy()


def stage_thinker():
    import torch
    embed, norm_w, layers = _load_thinker()
    rng = np.random.default_rng(1234)
    T = 12
    ids = rng.integers(0, embed.shape[0], size=T)
    x = embed[torch.from_numpy(ids)].contiguous()
    x.numpy().astype(np.float32).tofile(os.path.join(DUMP, "embeds.bin"))

    subprocess.run([THINKER, "--model", MODEL, "--embeds",
                    os.path.join(DUMP, "embeds.bin"), "--tokens", str(T),
                    "--dump", DUMP] + FA_ARGS, check=True)

    ref = _qwen3_decode_ref(x, layers, norm_w)
    cpp = np.fromfile(os.path.join(DUMP, "hidden.bin"), dtype=np.float32).reshape(ref.shape)
    cos = cossim(cpp, ref)
    maxabs = float(np.max(np.abs(cpp - ref)))
    print(f"[Dec] shape {cpp.shape} cosine {cos:.8f} max_abs {maxabs:.3e}")
    return cos > 0.9999


def stage_prefix():
    import torch
    hidden = 1024
    AUDIO_START, AUDIO_END, AUDIO_PH = 151669, 151670, 151676
    audio = np.fromfile(os.path.join(DUMP, "windowed.bin"), dtype=np.float32).reshape(-1, hidden)
    S = audio.shape[0]

    pre = [5, 1023, 99]
    suf = [42, 7]
    ids = pre + [AUDIO_START] + [AUDIO_PH] * S + [AUDIO_END] + suf
    T = len(ids)
    audio_pos = len(pre) + 1
    np.array(ids, dtype=np.int32).tofile(os.path.join(DUMP, "ids.bin"))

    subprocess.run([THINKER, "--model", MODEL, "--ids", os.path.join(DUMP, "ids.bin"),
                    "--audio", os.path.join(DUMP, "windowed.bin"), "--audio-pos",
                    str(audio_pos), "--tokens", str(T), "--dump", DUMP] + FA_ARGS,
                   check=True)

    embed, norm_w, layers = _load_thinker()
    x = embed[torch.tensor(ids, dtype=torch.long)].float().contiguous()
    x[audio_pos:audio_pos + S] = torch.from_numpy(audio)

    ref = _qwen3_decode_ref(x, layers, norm_w)
    cpp = np.fromfile(os.path.join(DUMP, "hidden.bin"), dtype=np.float32).reshape(ref.shape)
    cos = cossim(cpp, ref)
    maxabs = float(np.max(np.abs(cpp - ref)))
    print(f"[Pre] shape {cpp.shape} S {S} cosine {cos:.8f} max_abs {maxabs:.3e}")
    return cos > 0.9999


def main():
    make_signal()
    ok = stage_mel()
    ok = stage_conv_stem() and ok
    ok = stage_encoder() and ok
    ok = stage_windowed() and ok
    ok = stage_thinker() and ok
    ok = stage_prefix() and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
