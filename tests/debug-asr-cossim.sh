#!/bin/bash

for size in 1.7B 0.6B; do
    for backend in CUDA0 Vulkan0 CPU; do
        for quant in BF16 Q8_0 Q4_K_M; do
            GGML_BACKEND=$backend ./debug-asr-cossim.py \
                --model ../models/qwenasr-${size}-${quant}.gguf \
                2>&1 | tee asr-${size}-${backend}-${quant}.log
        done
    done
done
