#pragma once
// prompt-asr.h: builds the Qwen3-ASR chat prompt as a token id sequence. The
// single audio placeholder expands to one id per audio tower frame, marking the
// span where the pipeline splices the encoder states into the input embeddings.

#include "bpe.h"
#include "gguf-weights.h"

#include <string>
#include <vector>

// Control and audio token ids resolved from the model. Audio ids come from the
// qwenasr.* metadata, the chat control ids from the tokenizer vocab.
struct AsrSpecials {
    int im_start;
    int im_end;
    int audio_start;
    int audio_end;
    int audio_pad;
    int asr_text;
    int eos;
};

// Resolves a vocab token string to its id, -1 when absent.
static int asr_vocab_id(const BPETokenizer & tok, const std::string & s) {
    auto it = tok.vocab.find(s);
    return it != tok.vocab.end() ? it->second : -1;
}

static AsrSpecials asr_resolve_specials(const BPETokenizer & tok, const GGUFModel & gf) {
    AsrSpecials sp;
    sp.audio_start = (int) gf_get_u32(gf, "qwenasr.audio_start_token_id");
    sp.audio_end   = (int) gf_get_u32(gf, "qwenasr.audio_end_token_id");
    sp.audio_pad   = (int) gf_get_u32(gf, "qwenasr.audio_token_id");
    sp.im_start    = asr_vocab_id(tok, "<|im_start|>");
    sp.im_end      = asr_vocab_id(tok, "<|im_end|>");
    sp.asr_text    = asr_vocab_id(tok, "<asr_text>");
    sp.eos         = tok.eos_id;
    return sp;
}

// Prompt id sequence plus the audio splice span. audio_offset is the index of
// the first audio placeholder, audio_count equals the tower frame count.
struct AsrPrompt {
    std::vector<int> ids;
    int              audio_offset;
    int              audio_count;
};

// Chat template:
//   <|im_start|>system\n{context}<|im_end|>\n
//   <|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n[language {language}<asr_text>]
// Literal text runs through the BPE encoder, special tokens emit single ids.
// language is the canonical name resolved by the caller, empty for auto detect.
// When set the decoder is primed with the forced language tag, otherwise it
// detects the language itself.
static AsrPrompt asr_build_prompt(const BPETokenizer & tok,
                                  const AsrSpecials &  sp,
                                  const std::string &  context,
                                  const std::string &  language,
                                  int                  n_audio_frames) {
    AsrPrompt p;
    auto      text = [&](const std::string & s) {
        std::vector<int> e = bpe_encode(&tok, s, false);
        p.ids.insert(p.ids.end(), e.begin(), e.end());
    };

    p.ids.push_back(sp.im_start);
    text("system\n" + context);
    p.ids.push_back(sp.im_end);
    text("\n");

    p.ids.push_back(sp.im_start);
    text("user\n");
    p.ids.push_back(sp.audio_start);
    p.audio_offset = (int) p.ids.size();
    p.audio_count  = n_audio_frames;
    for (int i = 0; i < n_audio_frames; i++) {
        p.ids.push_back(sp.audio_pad);
    }
    p.ids.push_back(sp.audio_end);
    p.ids.push_back(sp.im_end);
    text("\n");

    p.ids.push_back(sp.im_start);
    text("assistant\n");
    if (!language.empty()) {
        text("language " + language);
        p.ids.push_back(sp.asr_text);
    }
    return p;
}
