# Architecture

Technical reference for qwenasr.cpp, the GGML port of Qwen3-ASR (Qwen team,
Alibaba). This document covers the model, the conversion to GGUF, the
inference pipeline, the GGML graph conventions, and the CLI tools.

## Upstream model

Qwen3-ASR (Qwen team / Alibaba, Apache 2.0) is a multilingual speech-to-text
system covering 30 languages with 22 Chinese dialects. Any input is resampled
to 16 kHz mono before the frontend. Two checkpoint sizes ship, the same code
path drives both, every dimension read from the GGUF metadata :

  0.6B  embedded target, smaller decoder and audio encoder
  1.7B  top quality, wider decoder and a deeper audio encoder

The system is not autoregressive on the audio side. A convolutional plus
transformer audio tower turns the waveform into a fixed sequence of states,
those states splice into the text input embeddings as an audio prefix, and a
single Qwen3 Thinker decoder reads that prefix and generates the transcript
autoregressively to the chat end marker.

Public checkpoints, two sizes :

  Thinker      Qwen3 decoder, 28 layers, output over vocab 151936
  Audio tower  log-mel frontend, conv2d stem, windowed Whisper style encoder
  Tokenizer    GPT2 byte level BPE, shared with the chat template
  Audio rate   16 kHz mono, log-mel n_fft 400 hop 160, 128 mel bins
  Frame rate   100 Hz mel, 8x time downsample in the conv stem -> 12.5 Hz

Per size, read from the metadata :

  0.6B  Thinker hidden 1024, FFN 3072 ; audio d_model 896, output_dim 1024
  1.7B  Thinker hidden 2048, FFN 6144 ; audio d_model 1024, output_dim 2048

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

The GGML submodule is stock `https://github.com/ggml-org/ggml`. qwenasr needs
no custom op : the frontend, the audio tower and the decoder all run on the
upstream ops (`ggml_conv_2d`, `ggml_im2col`, `ggml_mul_mat`, `ggml_rope_ext`,
`ggml_flash_attn_ext`, `ggml_soft_max_ext`). The build raises
`GGML_MAX_NAME` to 128 because the Qwen3-ASR tensor names exceed the default 64.

## Model conversion

```
./checkpoints.sh      # hf download Qwen/Qwen3-ASR-0.6B and 1.7B -> checkpoints/
./convert.py          # BF16 GGUF for both sizes -> models/
./quantize.sh         # Q8_0 and Q4_K_M derived from the BF16 base
```

`convert.py` is a byte preserving pass. The Qwen3-ASR checkpoint is BF16, so
the converter never downcasts : every tensor keeps its native BF16 payload,
and `quantize.sh` derives only Q8_0 and Q4_K_M from that base. F32 and F16
branches stay in the writer as defensive code in case a future release changes
the upstream dtype. Every step skips outputs that already exist.

Tensor names are renamed to the llama.cpp convention. The text decoder lands
under `thinker.blk.N.*`, the audio tower under `thinker.audio.*`, and any name
the renamer does not recognise raises loudly so a checkpoint layout change is
caught at convert time. The tokenizer is written as a GPT2 model : the id to
token list, the merge list and the special tokens folded in from
`tokenizer_config.json`. Only the id to token map is needed to detokenize ASR
output, the merges ship for symmetry with the sibling projects.

Quantisation policy, centralised in `tools/quantize.cpp` :

  Norms and biases are 1D, never quantised, kept at their loaded type or F16.
  Every 2D weight quantises to the variant base type.

  The token embedding takes the variant embed type (Q6_K for the K-quants,
  Q8_0 for Q8_0, BF16 for BF16).

  The attention value and the FFN down projections are bumped a notch above
  the base on the medium variants, output projection joins them on the
  large variants. Same per tensor rule llama.cpp uses for K-quant mixes.

## GGUF layout

`qwenasr-{size}-{variant}.gguf` (arch `qwenasr`, 708 tensors on the 1.7B
build) :

```
metadata
  general.architecture                          qwenasr

  qwenasr.block_count                           28
  qwenasr.embedding_length                      1024 (0.6B) | 2048 (1.7B)
  qwenasr.feed_forward_length                   3072 (0.6B) | 6144 (1.7B)
  qwenasr.attention.head_count                  16
  qwenasr.attention.head_count_kv               8     (GQA 2:1)
  qwenasr.attention.key_length                  128   (head_dim)
  qwenasr.attention.value_length                128
  qwenasr.vocab_size                            151936
  qwenasr.context_length                        65536
  qwenasr.attention.layer_norm_rms_epsilon      rms_norm_eps
  qwenasr.rope.freq_base                        1e6
  qwenasr.tie_word_embeddings                   true

  qwenasr.audio.d_model                         896 (0.6B) | 1024 (1.7B)
  qwenasr.audio.downsample_hidden_size          480 (both sizes), conv channel width
  qwenasr.audio.encoder_layers                  18 (0.6B) | 24 (1.7B)
  qwenasr.audio.encoder_attention_heads         14 (0.6B) | 16 (1.7B)
  qwenasr.audio.encoder_ffn_dim                 3584 (0.6B) | 4096 (1.7B)
  qwenasr.audio.num_mel_bins                    128
  qwenasr.audio.output_dim                      1024 (0.6B) | 2048 (1.7B)
  qwenasr.audio.n_window                        50    (chunk_mel = n_window * 2 = 100)
  qwenasr.audio.n_window_infer                  800   (window_chunks = n_window_infer / chunk_mel = 8)
  qwenasr.audio.max_source_positions            1500  (positional table length)

  qwenasr.audio_start_token_id                  151669  <|audio_start|>
  qwenasr.audio_end_token_id                    151670  <|audio_end|>
  qwenasr.audio_token_id                        151676  audio placeholder
  tokenizer (gpt2 BPE, 151705 vocab, 151291 merges, eos 151643)

tensors
  thinker.token_embd.weight                     text token embedding
  thinker.output_norm.weight                    final RMSNorm
  thinker.output.weight                         hidden -> vocab logits
  thinker.blk.0..27.attn_norm / ffn_norm        RMSNorm, no bias
  thinker.blk.0..27.attn_q / attn_k / attn_v / attn_output   GQA, no bias
  thinker.blk.0..27.attn_q_norm / attn_k_norm   per-head RMSNorm (head_dim,)
  thinker.blk.0..27.ffn_gate / ffn_up / ffn_down             SwiGLU, no bias

  thinker.audio.conv1 / conv2 / conv3 .weight, .bias         conv2d stem
  thinker.audio.conv_out.weight                 stem -> d_model, Linear no bias
  thinker.audio.blk.0..L.attn_norm .weight, .bias            LayerNorm w/ bias
  thinker.audio.blk.0..L.attn_q / attn_k / attn_v / attn_output .weight, .bias
  thinker.audio.blk.0..L.ffn_norm .weight, .bias             LayerNorm w/ bias
  thinker.audio.blk.0..L.ffn_up / ffn_down .weight, .bias    gelu MLP
  thinker.audio.post_norm .weight, .bias        encoder final LayerNorm
  thinker.audio.proj1 / proj2 .weight, .bias    output head, d_model -> output_dim
```

## Component architecture

### Mel frontend

The log-mel frontend is the WhisperFeatureExtractor equivalent. GGML has no
native FFT op, so the DFT folds into two real matmuls over im2col frames, all
on the backend. Config : sample rate 16000, n_fft 400, hop 160, 128 mel bins,
fmin 0, fmax 8000.

```
audio                                         f32 mono, resampled to 16 kHz
  -> reflect pad by n_fft / 2 on the host     Whisper center=True
  -> im2col into [n_fft, n_frames] frames      hop stride, drop nothing yet
  -> multiply by the Hann window
  -> spec_re = dft_real @ frames               [n_freq, n_frames], prec F32
     spec_im = dft_imag @ frames
  -> power = spec_re^2 + spec_im^2             magnitude squared, no sqrt
  -> mel = power @ slaney_filterbank           [n_frames, n_mels], prec F32
  -> clamp(1e-10, 1e30) -> log -> scale 1/ln10 log10 mel
```

The Hann window, the cos / sin DFT matrices and the Slaney filterbank are
baked once on the host with the trig evaluated in f64 to keep roundoff under
the f32 ULP, then uploaded as graph inputs. The graph emits the log10 mel laid
out `[n_frames_full, n_mels]` in ne (memory n_mels outer, frame inner). The
data dependent tail runs on the host after readback, since GGML has no global
max reduction : drop the trailing frame Whisper style, then
`log_spec = (max(log_spec, log_spec.max() - 8) + 4) / 4`. The result is the
normalized mel `[n_mels, n_frames]` the conv stem reads directly.

### Conv2d subsampling stem

The stem treats the mel matrix as a single channel image
`[1, num_mel_bins, n_frames]`. Three conv2d (kernel 3, stride 2, padding 1)
with gelu collapse the frequency axis 128 -> 16 and downsample time 8x.

```
mel [n_frames, n_mels, 1, 1]
  -> conv2d k3 s2 p1 (1 -> downsample_hidden) + bias + gelu
  -> conv2d k3 s2 p1 + bias + gelu
  -> conv2d k3 s2 p1 + bias + gelu             [t_out, 16, downsample_hidden, 1]
  -> permute + cont                            frequency inner, channel outer
  -> reshape [downsample_hidden * 16, t_out]   7680 on the 0.6B
  -> conv_out (Linear, no bias)                [d_model, t_out]
```

The permute matches the reference `permute(0, 3, 1, 2).view(b, t, c * f)`, so
the contiguous order puts frequency inner and channel outer before the Linear.
The stem is built on `ggml_conv_2d` (im2col plus matmul), never col2im, since
this is the analysis side. The after-cnn length of a chunk of `mel_len` frames
is `(((mel_len - 1) / 2 + 1 - 1) / 2 + 1 - 1) / 2 + 1`, the three strided
convolutions applied in turn, matching the reference output length helper.

### Windowed audio tower

The tower runs the stem and the encoder in fixed windows so the attention cost
stays bounded for long audio. The mel splits into `chunk_mel = n_window * 2`
frame chunks (100 by default), the conv stem runs per chunk with the
positional table reset per chunk, the after-cnn frames concatenate, and the
encoder attends under a block-diagonal mask over windows of
`window_aftercnn = conv_out(chunk_mel) * window_chunks` frames (13 * 8 = 104
by default).

```
mel [n_frames, n_mels]
  for each chunk (chunk_mel frames, tail carries the remainder) :
    stem chunk           -> [d_model, t_c]
    add sinusoidal PE     PE sliced to t_c, reset per chunk
    concat on the time axis
  -> encoder over the concatenated sequence, block-diagonal mask
  -> [output_dim, S]      S = sum of per-chunk after-cnn lengths
```

The positional table is a classic sinusoidal grid,
`pe[t, i] = sin(t * inv[i])`, `pe[t, half + i] = cos(t * inv[i])` with
`inv[i] = exp(-ln(10000) / (half - 1) * i)`, trig in f64.

The encoder is a pre-norm Whisper style transformer : 18 layers on the 0.6B
(24 on the 1.7B), d_model 896 / 1024, 14 / 16 heads of head_dim 64, FFN
3584 / 4096 with gelu, LayerNorm with bias, bias on q / k / v / output. Attention is plain
bidirectional MHA, no RoPE and no QK-norm, scaled by `1 / sqrt(head_dim)` and
masked through `ggml_soft_max_ext` with the additive block-diagonal mask. A
single window collapses to one dense block (NULL mask). After the stack a
`post_norm` LayerNorm, then the output head `proj1 -> gelu -> proj2` projects
d_model to `output_dim`, which equals the Thinker hidden width so the states
drop straight into the decoder embeddings.

### Qwen3 Thinker decoder

A decoder-only Qwen3 LM, KV cached, the text side of Qwen3-ASR. 28 layers both
sizes, GQA with 16 query heads and 8 KV heads (2:1), head_dim 128, RoPE theta
1e6, per-head QK-norm before RoPE, SwiGLU MLP, two residuals per layer, RMSNorm
throughout, no bias anywhere. The two sizes differ only in width : hidden 1024
and FFN 3072 on the 0.6B, hidden 2048 and FFN 6144 on the 1.7B. The token
embedding is tied to the output projection.

```
input embeddings [T, hidden]      text tokens, audio prefix spliced in
  per layer :
    RMSNorm -> q/k/v proj
    per-head RMSNorm on q and k (RMS over head_dim, then a [head_dim] gain)
    1D NEOX RoPE on q and k
    write fresh K/V into the cache at [n_past, n_past + T)
    attend over [0, n_past + T), GQA scaled dot product
    output proj, residual add
    RMSNorm -> SwiGLU (silu(gate) * up) -> down proj, residual add
  -> final RMSNorm -> output proj  [vocab, T]
```

The reference multimodal RoPE carries three sections, but the ASR timeline
assigns the same position id to all three axes, so the interleaved mrope
collapses exactly to a plain 1D NEOX rope. Attention has two paths : the fused
`ggml_flash_attn_ext` with F16 accumulation guarded by `set_prec F32`, or a
manual F32 chain (`mul_mat -> soft_max_ext -> mul_mat`) when flash attention is
off. `clamp_fp16` inserts `ggml_clamp(-65504, 65504)` on V before attention
and on the residual stream, guarding FP16 matmul accumulation on sub-Ampere
CUDA targets.

The decoder exposes two entry points over the same stack. Prefill rewinds the
KV cache and writes the whole prompt span in one shot. Decode feeds a single
embedding, appends one position at `cur_len`, and attends over the
`[0, cur_len + 1)` window. The cache is sized at init for `T + max_new`
positions and never reallocates.

## Inference pipeline

### Prompt assembly and audio splice

The chat prompt is built as a token id sequence with a single audio
placeholder span that the pipeline overwrites with the encoder states.

```
<|im_start|>system\n{context}<|im_end|>\n
<|im_start|>user\n<|audio_start|>{audio_pad x S}<|audio_end|><|im_end|>\n
<|im_start|>assistant\n[language {Lang}<asr_text>]
```

Literal text runs through the BPE encoder, special tokens emit single ids. The
audio placeholder repeats `audio_pad` once per tower frame, `audio_offset`
records the index of the first placeholder and `audio_count` equals `S`. The
pipeline embeds the full id sequence with `ggml_get_rows` on the token
embedding, then memcpys the `S * hidden` encoder states over the placeholder
span. The `language ...<asr_text>` tail is present only for a forced language ;
auto detect leaves the assistant turn open so the model emits its own
preamble.

### Language resolution and auto detect

A forced language is normalised to canonical casing (first byte upper, rest
lower, ASCII fold only) and accepted only if it is one of the 30 supported
names. The prompt then receives the canonical name and the `<asr_text>` marker,
priming the decoder with the forced tag. An empty string, `auto`, `none` or any
unsupported name resolves to empty, which leaves the model to detect the
language and emit a `language {Lang}<asr_text>` preamble itself.

### Decode loop and transcript extraction

```
resample to 16 kHz -> mel -> windowed tower -> prompt build + audio splice
prefill the Thinker over the prompt span
loop until im_end / eos or max_new_tokens (default 512) :
    sample next token   (greedy argmax when temperature <= 0)
    get_rows embed the token, decode one step
transcript = detokenize(tokens after the last <asr_text> marker)
```

The transcript is everything after the last `<asr_text>` id, so a forced run
keeps the full generation while an auto run drops the detected-language
preamble. With `on_token` set the loop emits the new suffix as a UTF-8 delta
each step and a false return unwinds with `QA_STATUS_CANCELLED`. Sampling
mirrors the HuggingFace `generate()` chain in F32
(`repetition_penalty -> temperature -> top_k -> top_p -> softmax ->
multinomial`), the uniform draw coming from `philox_uniform_fill` so a fixed
seed replays byte for byte. A seed of -1 draws a hardware seed.

Every stage is timed to a device readback and routed through `qa_log` at info
level : resample, mel, tower, build, prefill, per token embed, decode, total,
plus the realtime factor over the input duration.

## Public API

### Top-level public ABI : src/qwenasr.h

Single-header, plain C99, `extern "C"`. The opaque `qa_context` handle
aggregates the mel frontend, the conv stem, the audio encoder, the Qwen3
Thinker decoder, the BPE tokenizer and the GGML backend pair. One init, one
free, one transcribe call covers the full wav to text path, consumable
identically from C, C++, Python ctypes, Rust bindgen or Go cgo.

```c
#include "qwenasr.h"

struct qa_init_params iparams = qa_init_default_params();
iparams.model_path = "models/qwenasr-1.7B-BF16.gguf";

struct qa_context * ctx = qa_init(&iparams);

struct qa_transcribe_params params = qa_transcribe_default_params();
params.language = "English"; /* NULL for auto detect */

char * text = NULL;
qa_transcribe(ctx, samples, n_samples, sample_rate, &params, &text);
/* text is a NUL terminated UTF-8 transcript, samples mono f32, any rate */
qa_free_text(text);
qa_free(ctx);
```

Status codes :

```
QA_STATUS_OK                0
QA_STATUS_INVALID_PARAMS   -1
QA_STATUS_LOAD_FAILED      -2
QA_STATUS_GENERATE_FAILED  -3
QA_STATUS_OOM              -4
QA_STATUS_CANCELLED        -5
```

`qa_transcribe_params` carries the language hint, an optional `context` system
text for biasing, `max_new_tokens`, the full sampling set (temperature, top_k,
top_p, repetition_penalty, seed) and the `on_token` streaming callback.
`qa_init_params` selects the GPU backend, the thread count, `flash_attn` and
`clamp_fp16`. `qa_log_set` installs a callback receiving every internal
diagnostic, NULL routes to stderr. `qa_last_error` returns a thread-local
message valid right after a failure.

`QA_ABI_VERSION` guards struct growth : callers set `abi_version` (or let the
default-params helpers do it) and the entries reject a struct laid out for a
newer header. `qa_version()` returns the git short hash and commit date.

### ABI guarantee

`tests/abi-c.c` is built on every build as the `test-abi-c` target with
`-std=c99 -Wall -Werror -pedantic`. It includes the public header, exercises
the early-return paths, and never loads a model. Any regression that breaks
plain C consumability fails the main build, not just an opt-in target.

The static `libqwenasr-core.a` is the default artefact the CLI tools link.
For binding consumers, configure with `-DQWENASR_SHARED=ON` to add
`libqwenasr.so` (or `.dll` / `.dylib`) exporting only the `qa_*` symbols ;
every internal `pipeline_*` and `backend_*` stays hidden behind
`-fvisibility=hidden`.

### Low-level API : src/pipeline-asr.h

Direct access to `pipeline_asr_load`, `pipeline_asr_run` and
`pipeline_asr_free`. The public ABI wraps these, and the debug tools call them
directly for staged dumps. C++ types in the signatures, not part of the public
ABI.

## CLI tools

### qwenasr-transcribe

```
Usage: ./build/qwenasr-transcribe --model <gguf> --file <wav> [options]

Required:
  --model <gguf>          Qwen3-ASR GGUF (F32 / BF16 / Q8_0 / Q4_K_M)
  --file <wav>            Input audio, any rate (resampled to 16 kHz mono)

Optional:
  --lang <code>           Language hint, ISO code or name (default: auto detect)
  --context <txt>         System context for biasing the transcript
  --max-tokens <n>        Decode budget in tokens (default: 512)
  -o <path>               Output transcript file (stdout otherwise)

Sampling:
  --temp <t>              Temperature, <= 0 stays greedy (default: 0)
  --top-k <k>             Keep the top k logits, 0 disables (default: 0)
  --top-p <p>             Nucleus probability, 1.0 disables (default: 1.0)
  --rep-pen <r>           Repetition penalty, 1.0 is a no-op (default: 1.0)
  --seed <n>              PRNG seed, -1 draws a random one (default: -1)

Debug:
  --clamp-fp16            Clamp hidden states to FP16 range
  --no-fa                 Disable flash attention
  --dump <dir>            Dump intermediate tensors (f32) to <dir>
```

### asr-server

```
Usage: ./build/asr-server --model <gguf> [options]

Required:
  --model <gguf>     Qwen3-ASR GGUF

Optional:
  --host <addr>      Bind address (default: 127.0.0.1)
  --port <n>         Bind port (default: 8090)
  --lang <code>      Default language hint (default: auto detect)
  --clamp-fp16       Clamp hidden states to FP16 range
  --no-fa            Disable flash attention
```

An OpenAI compatible HTTP server over the transcription pipeline.
`POST /v1/audio/transcriptions` takes multipart form-data with `file` (audio,
required) plus optional `model`, `language`, `prompt` and `response_format`.
`GET /v1/models` returns the single loaded model. The server runs over
cpp-httplib with no SSL, meant to sit behind a reverse proxy, and parses and
emits JSON through yyjson.

### Stage dumps

`qwenasr-transcribe --dump <dir>` writes the mel, conv stem, windowed tower,
prefill hidden and logits as raw f32 in one end to end run, plus the resampled
16k pcm. The cossim harness feeds that pcm to the transformers reference and
compares each stage against the real model through forward hooks.

## Module map

```
src/
  backend.h                 GGML backend init, scheduler factory, shared refcounted
  weight-ctx.h              Generic weight context for GGUF loaders
  gguf-weights.h            mmap GGUF, gf_load_tensor_f32, gf_load_conv_f16, gf_get_*
  kv-cache.h                Persistent per-layer KV cache, fixed max len, rewind reset
  audio-io.h / wav.h        WAV read, mono write
  audio-resample.h          Kaiser polyphase resampler to 16 kHz mono
  audio-mel.h               Log-mel frontend, im2col DFT, Slaney filterbank
  philox.h                  Philox4x32-10 counter-based PRNG
  sampling.h                Thinker sampling chain (rep_pen / temp / top_k / top_p)
  utf8.h                    UTF-8 helpers for streaming detokenize
  bpe.h                     GPT2 byte-level BPE, GGUF loader
  lang-map.h                Forced language normalisation, 30 supported names

  conv-stem.h               Conv2d subsampling stem, freq 128 -> 16, time 8x
  audio-enc.h               Whisper style transformer encoder, sinusoidal PE
  audio-tower.h             Windowed forward, chunk plan, block-diagonal mask
  prompt-asr.h              Chat prompt build, audio placeholder span
  thinker-weights.h         Qwen3 Thinker decoder weights
  thinker-forward.h         Prefill + decode forwards, KV cached, GQA, NEOX RoPE
  pipeline-asr.{h,cpp}      wav -> text orchestration
  qwenasr.{h,cpp}           Public ABI : opaque qa_context, plain C99 header
  qa-error.h                Thread-local error slot, qa_throw, redirectable log
  timer.h                   Wall clock spans for the perf log

tools/
  qwenasr-transcribe.cpp    CLI : audio -> text, --dump for stage tensors
  asr-server.cpp            OpenAI /v1/audio/transcriptions server
  quantize.cpp              GGUF requantizer with the per tensor policy
  version.cmake             Embeds the git short hash into the binary

tests/
  debug-asr-cossim.py       Per-stage cossim vs the transformers reference
  abi-c.c                   Plain C99 smoke test for the public ABI
  asr-{size}-{backend}-{quant}.log   Reference runs across backends and quants
```

## GGML conventions

### Tensor shape and layout

PyTorch `(out, in)` for a Linear stores as ggml `ne[0] = in, ne[1] = out`.
`ggml_mul_mat(A, B)` with `A.ne[0] = K`, `A.ne[1] = M`, `B.ne[1] = N` gives
output `(N, M)`, equal to `A @ B^T` in PyTorch terms. Weights load as F32 at
init through `gf_load_tensor_f32`, eliminating runtime cast nodes and giving
the cossim harness a clean high precision path.

### Mel DFT without an FFT op

GGML has no FFT op, so the short-time transform runs as `ggml_im2col` to frame
the padded audio, an elementwise Hann multiply, then two `ggml_mul_mat` against
the precomputed cos and sin DFT matrices. The two spectral matmuls and the
filterbank matmul carry `ggml_mul_mat_set_prec(GGML_PREC_F32)` so the power
spectrum is computed in F32 regardless of the backend default.

### Conv2d stem

The stem uses `ggml_conv_2d` (im2col plus matmul) over a single channel mel
image. The bias adds as a reshaped `[1, 1, channels, 1]` broadcast and the
activation is `ggml_gelu_erf`, the exact erf gelu, not the tanh approximation.
`gf_load_conv_f16` exists in the loader for backends whose im2col asserts an
F16 kernel (ARM aarch64), while the stem path here loads F32 for the high
precision dumps.

### Attention and RoPE

Decoder attention is `ggml_flash_attn_ext` with `set_prec F32`, falling back to
an explicit F32 `mul_mat / soft_max_ext / mul_mat` chain when flash attention
is disabled. RoPE is `ggml_rope_ext` in `GGML_ROPE_TYPE_NEOX` mode (rotate
half). The encoder attention is bidirectional and uses `ggml_soft_max_ext`
with the additive window mask, no RoPE.

### Backend lifecycle

`backend_init("Pipeline")` loads every available backend, picks the best GPU
and keeps the CPU as fallback, then `backend_sched_new` builds the scheduler.
Backend handles are shared across modules in a binary, refcounted. The CPU
thread count defaults to one per physical core. `flash_attn` collapses to the
manual chain on CPU-only backends.

## Validation

The harness is `tests/debug-asr-cossim.py`. It runs the same input through the
PyTorch reference (the transformers WhisperFeatureExtractor for the mel, the
`qwen_asr` reference for the later stages) and through the C++ binaries, dumps
each stage with `--dump`, and reports cosine similarity per stage. Latest run,
1.7B BF16 on CUDA0, greedy :

```
Forward fidelity (single pass, the correctness signal)
  Mel    [128, 3000]   cos 0.99999988
  Stem   [63, 1024]    cos 0.99999976
  Enc    [63, 2048]    cos 0.99999756
  Win    [117, 2048]   cos 0.99999881   (9 chunks)
  Dec    [12, 2048]    cos 0.99999934
  Pre    [124, 2048]   cos 0.99999726   (full prompt prefill, S 117)

End to end transcript
  norm_exact 100.00%   word_ratio 1.000000
```

Every forward stage sits at cosine 0.99999 and above : the mel, the conv stem,
the encoder, the windowed tower, the decoder prefill and the spliced prompt all
match the reference. The large per-channel max-abs values at the decoder taps
are the usual pre-final-norm outlier channels, cosine stays high because the
direction is preserved.

The transcript check is the meaningful end to end metric. Greedy decode is
deterministic, so the text matches the reference word for word ; only
punctuation and quoting differ at the margins, which is why the harness reports
the normalised word ratio rather than raw string equality. A representative run
transcribes 17.26 s of audio in 393.8 ms on the RTX PRO 6000 (mel 4.2 ms, tower
36.0 ms, prefill 13.6 ms, decode 333.4 ms over 51 tokens at 153 tok/s), a
realtime factor of 0.023.

The reference logs ship for every size, backend and quant under
`tests/asr-{size}-{backend}-{quant}.log` (0.6B and 1.7B across CPU, CUDA0 and
Vulkan0, in BF16 / Q8_0 / Q4_K_M), so a backend regression shows up as a stage
cosine drop against the committed baseline.

## Glossary

  Thinker      The Qwen3 decoder that generates the transcript. Reads the
                spliced audio prefix and the chat prompt, emits text tokens
                autoregressively to the chat end marker.

  Audio tower  The conv stem plus the transformer encoder, the path turning
                16 kHz audio into the state sequence spliced into the decoder.

  Conv stem    Three strided conv2d collapsing the mel frequency axis and
                downsampling time 8x before the encoder.

  after-cnn    A frame count measured after the conv stem stride 8 reduction,
                the unit the tower windows and masks operate in.

  Window mask  The block-diagonal additive attention mask that keeps each
                window of after-cnn frames attending only within itself.

  QK-norm      Per-head RMSNorm applied to the query and key vectors before
                RoPE, a Qwen3 attention feature.

  GQA          Grouped Query Attention. Fewer KV heads than query heads,
                16 query and 8 KV heads here (2:1).

  NEOX RoPE    Rotary position embedding in the rotate-half layout. The ASR
                timeline gives all three mrope axes the same id, collapsing
                the interleaved mrope to plain 1D NEOX.

  Slaney mel   The mel scale of librosa.filters.mel(htk=False), the Whisper
                filterbank convention used by the frontend.

  im2col DFT   The FFT-free short-time transform : im2col frames then two
                matmuls against precomputed cos and sin matrices.

  Audio splice The memcpy that overwrites the audio placeholder embeddings
                with the tower states inside the prompt embedding buffer.

  RTF          Realtime factor, transcription wall time divided by the input
                audio duration. Below 1.0 is faster than realtime.

  Philox       Counter-based PRNG used by PyTorch CUDA. Skip-ahead friendly,
                aligns the multinomial draw across runs for a fixed seed.
