#pragma once
// pipeline-asr.h: wav -> text orchestration. Owns the backend pair, the loaded
// weights, the BPE tokenizer, and drives mel -> conv stem -> audio encoder ->
// audio prefix -> Qwen3 decoder loop -> detokenize. The public ABI in
// qwenasr.h wraps this; the CLI tools call it directly for staged debug dumps.

#include "qa-error.h"
#include "qwenasr.h"

#include <string>
#include <vector>

struct pipeline_asr;

struct pipeline_asr_params {
  std::string model_path;
  int n_threads = 0;
  bool use_gpu = true;
  bool clamp_fp16 = false;
  bool flash_attn = true;
  std::string dump_dir; // when set, write the stage tensors there as f32 dumps
};

pipeline_asr *pipeline_asr_load(const pipeline_asr_params &params);
void pipeline_asr_free(pipeline_asr *p);

// Full transcription. params carries the language hint, the optional context,
// max_new_tokens and the streaming callback. text receives the decoded UTF-8
// string.
qa_status pipeline_asr_run(pipeline_asr *p, const float *pcm, size_t n_samples,
                           int sample_rate, const qa_transcribe_params *params,
                           std::string &text);
