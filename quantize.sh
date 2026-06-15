#!/bin/bash
# Derive lighter GGUFs from the BF16 source-of-truth produced by convert.py.
# Each BF16 model under models/ is quantized to Q8_0 and Q4_K_M. BF16 is the
# source here (the Qwen3-ASR checkpoint is BF16), so it is not re-derived.
#
# The per tensor target type policy lives in tools/quantize.cpp.

set -eu

Q="./build/quantize"

quantize() {
    local src="$1" type="$2"
    local out="${src/-BF16.gguf/-${type}.gguf}"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$src" "$out" "$type"
    fi
}

for src in models/qwenasr-*-BF16.gguf; do
    [ -f "$src" ] || continue
    quantize "$src" Q8_0
    quantize "$src" Q4_K_M
done
