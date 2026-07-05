#pragma once
// detok-stream.h: incremental streaming detokenizer for the ASR decode loop.
// Feeds one decoded token per step, returns the transcript bytes that newly
// complete a UTF-8 boundary, and holds back a multibyte codepoint split across
// tokens until its tail lands. The reset marker (asr_text) restarts the segment
// so the auto detect preamble never reaches the stream. Each token is decoded
// once, so the whole stream is linear where a full re-decode per step is
// quadratic.

#include "bpe.h"

#include <string>

// Byte length of the longest prefix of s ending on a UTF-8 boundary. A trailing
// lead byte whose continuation bytes have not all arrived yet stays held back.
static size_t detok_utf8_complete_len(const std::string & s) {
    size_t n = s.size();
    if (n == 0) {
        return 0;
    }
    size_t i = n - 1;
    while (i > 0 && ((unsigned char) s[i] & 0xC0) == 0x80) {
        i--;
    }
    unsigned char lead = (unsigned char) s[i];
    size_t        need;
    if (lead < 0x80) {
        need = 1;
    } else if ((lead & 0xE0) == 0xC0) {
        need = 2;
    } else if ((lead & 0xF0) == 0xE0) {
        need = 3;
    } else if ((lead & 0xF8) == 0xF0) {
        need = 4;
    } else {
        need = 1;
    }
    return (n - i >= need) ? n : i;
}

struct StreamDetok {
    const BPETokenizer * tok;       // tokenizer backing the per token byte decode
    int                  reset_id;  // token id that restarts the segment (asr_text)
    std::string          bytes;     // raw transcript bytes for the current segment
    size_t               emitted;   // bytes already returned to the caller
};

// Bind the detok to a tokenizer and the segment reset marker, empty segment.
static void detok_init(StreamDetok * d, const BPETokenizer * tok, int reset_id) {
    d->tok      = tok;
    d->reset_id = reset_id;
    d->bytes.clear();
    d->emitted = 0;
}

// Feed one decoded token id. Returns the transcript bytes newly completed since
// the last call, empty when the id is the reset marker, decodes to nothing, or
// only extends a pending multibyte codepoint. The reset marker clears the
// segment so a later language preamble drops out of the stream.
static std::string detok_feed(StreamDetok * d, int id) {
    if (id == d->reset_id) {
        d->bytes.clear();
        d->emitted = 0;
        return std::string();
    }
    bpe_decode_append(d->tok, id, d->bytes);
    size_t safe = detok_utf8_complete_len(d->bytes);
    if (safe <= d->emitted) {
        return std::string();
    }
    std::string delta = d->bytes.substr(d->emitted, safe - d->emitted);
    d->emitted        = safe;
    return delta;
}

// Full transcript bytes accumulated for the current segment, including any tail
// not yet streamed through the callback. The decode loop reads this for the
// final result, no re-decode.
static const std::string & detok_text(const StreamDetok * d) {
    return d->bytes;
}
