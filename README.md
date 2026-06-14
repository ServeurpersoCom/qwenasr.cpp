# qwenasr.cpp

Minimalist C++ / GGML port of Qwen3-ASR. Native wav -> text

## Pipeline

    pcm -> log-mel (real FFT) -> conv2d stem -> audio encoder
        -> audio prefix -> Qwen3 Thinker decoder (KV cache) -> BPE detokenize

Both model sizes are first class: 0.6B for the embedded target, 1.7B for top
quality. Dimensions come from the GGUF metadata, one code path for both.

## Layout

    src/        modules (mel, conv-stem, audio-enc, qwen3-decoder, pipeline-asr)
    src/qwenasr.h   public C ABI (qa_*), parsed directly by bindings
    tools/      qwenasr-transcribe, qwenasr-feat, asr-server, quantize
    tests/      abi-c.c, cossim harness vs ../../Qwen3-ASR
    ggml/       submodule, ServeurpersoCom/ggml fork

## Build

    git submodule update --init --recursive
    ./buildcuda.sh        # or buildcpu.sh / buildvulkan.sh

## Models

    ./checkpoints.sh      # Qwen/Qwen3-ASR-0.6B and 1.7B
    ./convert.py --size 0.6B
    ./convert.py --size 1.7B
    ./quantize.sh         # derive Q8_0 / Q4_K_M from the BF16 base
