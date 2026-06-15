#!/usr/bin/env python3
# convert.py: safetensors to GGUF for Qwen3-ASR (audio tower + Qwen3 Thinker
# decoder + tokenizer). Reads checkpoints/Qwen3-ASR-{size}/, writes one byte
# perfect BF16 GGUF to models/. The source checkpoint is BF16, so this
# converter never downcasts: every tensor keeps its native BF16 payload and
# quantize.sh derives Q8_0 / Q4_K_M from this BF16 GGUF.
#
# Tensor names are renamed to the llama.cpp convention (thinker.blk.N.* for the
# text decoder, thinker.audio.* for the audio tower), matching the sibling
# projects. The C++ loaders look them up under these names.
#
# Usage: ./convert.py --size 0.6B
#        ./convert.py --size 1.7B

import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def log(tag, msg):
    print("[%s] %s" % (tag, msg), file=sys.stderr, flush=True)

# safetensors header reader
def read_sf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return meta, 8 + n

# resolve the safetensors shards of a checkpoint. A single model.safetensors, or
# the model-0000N-of-0000M.safetensors files referenced by the index weight_map.
def resolve_shards(ckpt_dir):
    single = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return []
    with open(index, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    names = []
    for fn in weight_map.values():
        if fn not in names:
            names.append(fn)
    return [os.path.join(ckpt_dir, fn) for fn in names]

# map a HF tensor name to the llama.cpp convention. Raises on any unexpected
# name so a checkpoint layout change is caught loudly.
_LM_LEAF = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_AUDIO_TOP = {
    "conv2d1.weight": "conv1.weight", "conv2d1.bias": "conv1.bias",
    "conv2d2.weight": "conv2.weight", "conv2d2.bias": "conv2.bias",
    "conv2d3.weight": "conv3.weight", "conv2d3.bias": "conv3.bias",
    "conv_out.weight": "conv_out.weight",
    "ln_post.weight": "post_norm.weight", "ln_post.bias": "post_norm.bias",
    "proj1.weight": "proj1.weight", "proj1.bias": "proj1.bias",
    "proj2.weight": "proj2.weight", "proj2.bias": "proj2.bias",
}
_AUDIO_LEAF = {
    "self_attn_layer_norm.weight": "attn_norm.weight",
    "self_attn_layer_norm.bias": "attn_norm.bias",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.out_proj.weight": "attn_output.weight", "self_attn.out_proj.bias": "attn_output.bias",
    "final_layer_norm.weight": "ffn_norm.weight", "final_layer_norm.bias": "ffn_norm.bias",
    "fc1.weight": "ffn_up.weight", "fc1.bias": "ffn_up.bias",
    "fc2.weight": "ffn_down.weight", "fc2.bias": "ffn_down.bias",
}

def rename_tensor(name):
    if name == "thinker.model.embed_tokens.weight":
        return "thinker.token_embd.weight"
    if name == "thinker.model.norm.weight":
        return "thinker.output_norm.weight"
    if name == "thinker.lm_head.weight":
        return "thinker.output.weight"
    m = re.match(r"thinker\.model\.layers\.(\d+)\.(.+)$", name)
    if m:
        idx, leaf = m.group(1), m.group(2)
        if leaf in _LM_LEAF:
            return "thinker.blk.%s.%s" % (idx, _LM_LEAF[leaf])
        raise RuntimeError("unmapped thinker layer tensor: %s" % name)
    m = re.match(r"thinker\.audio_tower\.(.+)$", name)
    if m:
        rest = m.group(1)
        if rest in _AUDIO_TOP:
            return "thinker.audio.%s" % _AUDIO_TOP[rest]
        ml = re.match(r"layers\.(\d+)\.(.+)$", rest)
        if ml:
            idx, leaf = ml.group(1), ml.group(2)
            if leaf in _AUDIO_LEAF:
                return "thinker.audio.blk.%s.%s" % (idx, _AUDIO_LEAF[leaf])
        raise RuntimeError("unmapped audio tower tensor: %s" % name)
    raise RuntimeError("unmapped tensor: %s" % name)

# write a tensor to GGUF preserving its source dtype. The Qwen3-ASR checkpoint
# is BF16 everywhere; the F32 and F16 branches stay as defensive code in case a
# future release switches dtype upstream.
def add_tensor_passthrough(w, name, sf_path, hdr_size, info):
    dtype_str = info["dtype"]
    shape = info["shape"]
    off0, off1 = info["data_offsets"]
    nbytes = off1 - off0
    with open(sf_path, "rb") as f:
        f.seek(hdr_size + off0)
        raw = f.read(nbytes)
    if dtype_str == "BF16":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
        return nbytes
    if dtype_str == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        w.add_tensor(name, arr)
        return nbytes
    if dtype_str == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        w.add_tensor(name, arr)
        return nbytes
    raise RuntimeError("unsupported dtype %s for %s" % (dtype_str, name))

# GPT2 byte level tokenizer: token list from vocab.json, merges from merges.txt,
# special tokens folded in from tokenizer_config.json. Only the id to token map
# is needed to detokenize ASR output; merges ship for symmetry with the sibling
# projects.
def add_tokenizer(w, ckpt_dir, tag):
    with open(os.path.join(ckpt_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(ckpt_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        tcfg = json.load(f)

    added = tcfg.get("added_tokens_decoder", {})
    max_id = max(
        max(vocab.values(), default=-1),
        max((int(i) for i in added.keys()), default=-1),
    )
    tokens = [""] * (max_id + 1)
    for tok, tid in vocab.items():
        if 0 <= tid <= max_id:
            tokens[tid] = tok
    for tid, info in added.items():
        tokens[int(tid)] = info["content"]

    merges = []
    mpath = os.path.join(ckpt_dir, "merges.txt")
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            merges.append(line)

    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    log(tag, "tokenizer: %d vocab, %d merges, %d added" % (len(tokens), len(merges), len(added)))

def convert(size):
    tag = size
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints", "Qwen3-ASR-%s" % size)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    out_path = os.path.join(SCRIPT_DIR, "models", "qwenasr-%s-BF16.gguf" % size)

    shards = resolve_shards(ckpt_dir)
    if not shards:
        log(tag, "missing safetensors in %s, run checkpoints.sh first" % ckpt_dir)
        sys.exit(1)
    if os.path.exists(out_path):
        log(tag, "skip: %s exists" % os.path.basename(out_path))
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    thinker = cfg["thinker_config"]
    tc = thinker["text_config"]
    ac = thinker["audio_config"]

    log(tag, "writing %s" % os.path.basename(out_path))
    w = gguf.GGUFWriter(out_path, "qwenasr", use_temp_file=True)
    w.add_name("Qwen3-ASR-%s" % size)

    # text decoder metadata (Qwen3 Thinker)
    w.add_block_count(tc["num_hidden_layers"])
    w.add_embedding_length(tc["hidden_size"])
    w.add_feed_forward_length(tc["intermediate_size"])
    w.add_head_count(tc["num_attention_heads"])
    w.add_head_count_kv(tc["num_key_value_heads"])
    w.add_key_length(tc["head_dim"])
    w.add_value_length(tc["head_dim"])
    w.add_vocab_size(tc["vocab_size"])
    w.add_context_length(tc["max_position_embeddings"])
    w.add_layer_norm_rms_eps(tc["rms_norm_eps"])
    w.add_rope_freq_base(float(tc["rope_theta"]))
    w.add_bool("qwenasr.tie_word_embeddings", bool(tc.get("tie_word_embeddings", False)))

    # audio tower metadata
    w.add_uint32("qwenasr.audio.d_model", ac["d_model"])
    w.add_uint32("qwenasr.audio.downsample_hidden_size", ac["downsample_hidden_size"])
    w.add_uint32("qwenasr.audio.encoder_layers", ac["encoder_layers"])
    w.add_uint32("qwenasr.audio.encoder_attention_heads", ac["encoder_attention_heads"])
    w.add_uint32("qwenasr.audio.encoder_ffn_dim", ac["encoder_ffn_dim"])
    w.add_uint32("qwenasr.audio.num_mel_bins", ac["num_mel_bins"])
    w.add_uint32("qwenasr.audio.output_dim", ac["output_dim"])
    w.add_uint32("qwenasr.audio.n_window", ac["n_window"])
    w.add_uint32("qwenasr.audio.n_window_infer", ac["n_window_infer"])
    w.add_uint32("qwenasr.audio.max_source_positions", ac["max_source_positions"])

    # audio placeholder token ids (text stream side)
    w.add_uint32("qwenasr.audio_start_token_id", thinker["audio_start_token_id"])
    w.add_uint32("qwenasr.audio_end_token_id", thinker["audio_end_token_id"])
    w.add_uint32("qwenasr.audio_token_id", thinker["audio_token_id"])

    add_tokenizer(w, ckpt_dir, tag)

    n_tensors, n_bytes = 0, 0
    for sf_path in shards:
        meta, hdr_size = read_sf_header(sf_path)
        for name in sorted(meta.keys()):
            out_name = rename_tensor(name)
            n_bytes += add_tensor_passthrough(w, out_name, sf_path, hdr_size, meta[name])
            n_tensors += 1
    log(tag, "tensors: %d, %.1f MB" % (n_tensors, n_bytes / (1 << 20)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    log(tag, "wrote %.0f MB -> %s" % (os.path.getsize(out_path) / (1 << 20), out_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["0.6B", "1.7B"], required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(SCRIPT_DIR, "models"), exist_ok=True)
    convert(args.size)

if __name__ == "__main__":
    main()
