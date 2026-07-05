# qwenasr.cpp

Local AI speech-to-text, powered by GGML. C++17 port of Qwen3-ASR
(Qwen team, Alibaba). 30 languages and 22 Chinese dialects, any input
resampled to 16 kHz mono, runs on CPU, CUDA, Vulkan, SYCL.

## Features

- Native audio -> text, any sample rate and channel count resampled
  and downmixed to 16 kHz mono
- Two model sizes, 0.6B for the embedded target and 1.7B for top
  quality, one code path with every dimension read from the GGUF
  metadata
- Audio tower: log-mel real FFT -> conv2d stem -> windowed Whisper
  style encoder, spliced into the decoder as an audio prefix
- Qwen3 Thinker decoder: KV cached, NEOX RoPE, GQA with qk-norm,
  SwiGLU, flash attention, greedy decode to im_end
- Language auto detection or a forced hint, plus an optional system
  context for biasing
- Q8_0 and Q4_K_M quantisation of both sizes, derived from the BF16
  base
- Two CLI tools: `qwenasr-transcribe` (audio -> text) and `asr-server`
  (OpenAI `/v1/audio/transcriptions`)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwenasr.cpp.git
cd qwenasr.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildsycl.sh                   # Intel GPU (SYCL)
./buildcpu.sh                    # CPU only
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

## Model conversion

```
./checkpoints.sh      # hf download Qwen/Qwen3-ASR-0.6B and 1.7B -> checkpoints/
./convert.py          # BF16 GGUF for both sizes -> models/
./quantize.sh         # Q8_0 and Q4_K_M derived from the BF16 base
```

The BF16 GGUF is the source of truth (the Qwen3-ASR checkpoint is BF16),
so quantize derives only Q8_0 and Q4_K_M. Every step skips outputs that
already exist.

## Quick start

Each block is the command run by the matching script in `examples/`.

Transcribe a file (`transcribe.sh`):

```
./build/qwenasr-transcribe \
    --model models/qwenasr-1.7B-BF16.gguf \
    --file freeman.wav \
    --lang English -o output.txt
```

OpenAI-compatible server (`server.sh`):

```
./build/asr-server \
    --model models/qwenasr-1.7B-BF16.gguf \
    --host 127.0.0.1 --port 8090
```

Client call (`client.sh`):

```
curl -s -X POST http://127.0.0.1:8090/v1/audio/transcriptions \
    -F file=@freeman.wav \
    -F response_format=json
```

Language is auto detected when `--lang` is omitted. Pass `--context`
(CLI) or the `prompt` field (server) to bias the transcript. The form
also accepts optional sampling overrides (`temperature`, `top_k`,
`top_p`, `repetition_penalty`, `max_new_tokens`, `seed`); unset fields
keep the engine defaults and a fixed seed makes a sampled request
reproducible.

## Embedding the library

The CLI tools are thin wrappers over a public ABI. Single-header,
single-name-prefix, plain C linkage so that C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way.

```c
#include "qwenasr.h"

struct qa_init_params iparams = qa_init_default_params();
iparams.model_path = "models/qwenasr-1.7B-BF16.gguf";

struct qa_context * ctx = qa_init(&iparams);

struct qa_transcribe_params params = qa_transcribe_default_params();
params.language = "English"; /* NULL for auto detect */

/* samples is the caller's mono f32 audio, resampled to 16 kHz internally */
char * text = NULL;
qa_transcribe(ctx, samples, n_samples, sample_rate, &params, &text);
/* text is a NUL terminated UTF-8 transcript */
qa_free_text(text);
qa_free(ctx);
```

`tests/abi-c.c` is built with `-std=c99 -Wall -Werror -pedantic` on
every build (the `test-abi-c` target), so any regression that breaks
plain C consumability fails the build, not just an opt-in target.

For a binding-friendly shared library (libqwenasr.so / .dll / .dylib),
configure with `cmake -DQWENASR_SHARED=ON ...`. The shared target
exports only the `qa_*` symbols; every internal `pipeline_*` and
`backend_*` stays hidden inside the .so.

## License

MIT. See [LICENSE](LICENSE).

Upstream model: Qwen3-ASR by Alibaba / Qwen team, Apache 2.0.
