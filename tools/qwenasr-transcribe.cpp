// qwenasr-transcribe: wav -> text. Reads a file as mono PCM, resampled to
// 16 kHz, runs one transcription, writes the transcript to stdout or to -o.
// Normal runs go through the public ABI in qwenasr.h. With --dump it drives the
// pipeline directly to write the stage tensors. Backend selection follows the
// GGML_BACKEND env var, CPU otherwise.

#include "audio-io.h"
#include "pipeline-asr.h"
#include "qwenasr.h"
#include "utf8.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr, "qwenasr.cpp %s\n\n", QWENASR_VERSION);
    fprintf(stderr,
            "Transcribe an audio file to text with Qwen3-ASR.\n\n"
            "Usage: %s --model <gguf> --file <wav> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Qwen3-ASR GGUF (F32 / BF16 / Q8_0 / Q4_K_M)\n"
            "  --file <wav>            Input audio, any rate (resampled to 16 kHz "
            "mono)\n\n"
            "Optional:\n"
            "  --lang <code>           Language hint, ISO code or name (default: "
            "auto "
            "detect)\n"
            "  --context <txt>         System context for biasing the transcript\n"
            "  --max-tokens <n>        Decode budget in tokens (default: 512)\n"
            "  -o <path>               Write transcript to a file instead of "
            "stdout\n\n"
            "Sampling (greedy by default, matching the reference):\n"
            "  --temp <t>              Temperature, <= 0 stays greedy (default: 0)\n"
            "  --top-k <k>             Keep the top k logits, 0 disables (default: "
            "0)\n"
            "  --top-p <p>             Nucleus probability, 1.0 disables (default: "
            "1.0)\n"
            "  --rep-pen <r>           Repetition penalty, 1.0 is a no-op (default: "
            "1.0)\n"
            "  --seed <n>              PRNG seed, -1 draws a random one (default: "
            "-1)\n\n"
            "Debug:\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --no-fa                 Disable flash attention\n"
            "  --dump <dir>            Dump intermediate tensors (f32) to <dir>\n",
            prog);
}

static int main_impl(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    const char * model_path  = nullptr;
    const char * file_path   = nullptr;
    const char * lang        = nullptr;
    const char * context     = nullptr;
    const char * out_path    = nullptr;
    const char * dump_dir    = nullptr;
    int          max_tokens  = 0;
    float        temperature = 0.0f;
    int          top_k       = 0;
    float        top_p       = 1.0f;
    float        rep_pen     = 1.0f;
    long long    seed        = -1;
    bool         clamp_fp16  = false;
    bool         flash_attn  = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            file_path = argv[++i];
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            lang = argv[++i];
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            context = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--rep-pen") == 0 && i + 1 < argc) {
            rep_pen = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
            clamp_fp16 = true;
        } else if (strcmp(argv[i], "--no-fa") == 0) {
            flash_attn = false;
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || !file_path) {
        print_usage(argv[0]);
        return 2;
    }

    int     n_samples = 0;
    float * pcm       = audio_read_mono(file_path, 16000, &n_samples);
    if (!pcm) {
        fprintf(stderr, "[CLI] ERROR: failed to read audio: %s\n", file_path);
        return 1;
    }

    struct qa_transcribe_params tp = qa_transcribe_default_params();
    tp.language                    = lang;
    tp.context                     = context;
    tp.temperature                 = temperature;
    tp.top_k                       = top_k;
    tp.top_p                       = top_p;
    tp.repetition_penalty          = rep_pen;
    tp.seed                        = seed;
    if (max_tokens > 0) {
        tp.max_new_tokens = max_tokens;
    }

    // Normal runs go through the public ABI. --dump drives the pipeline directly
    // so the stage tensors land on disk for the cossim harness.
    std::string text;
    qa_status   rc;
    if (dump_dir) {
        pipeline_asr_params pp;
        pp.model_path       = model_path;
        pp.clamp_fp16       = clamp_fp16;
        pp.flash_attn       = flash_attn;
        pp.dump_dir         = dump_dir;
        pipeline_asr * pipe = pipeline_asr_load(pp);
        rc                  = pipeline_asr_run(pipe, pcm, (size_t) n_samples, 16000, &tp, text);
        pipeline_asr_free(pipe);
    } else {
        struct qa_init_params ip = qa_init_default_params();
        ip.model_path            = model_path;
        ip.clamp_fp16            = clamp_fp16;
        ip.flash_attn            = flash_attn;
        qa_context * ctx         = qa_init(&ip);
        if (!ctx) {
            fprintf(stderr, "[CLI] ERROR: qa_init: %s\n", qa_last_error());
            free(pcm);
            return 1;
        }
        char * out_text = nullptr;
        rc              = qa_transcribe(ctx, pcm, (size_t) n_samples, 16000, &tp, &out_text);
        if (rc == QA_STATUS_OK) {
            text = out_text;
            qa_free_text(out_text);
        }
        qa_free(ctx);
    }
    free(pcm);

    if (rc != QA_STATUS_OK) {
        fprintf(stderr, "[CLI] ERROR: transcribe: %s\n", qa_last_error());
        return 1;
    }

    int ret = 0;
    if (out_path) {
        FILE * f = fopen(out_path, "wb");
        if (f) {
            fputs(text.c_str(), f);
            fputc('\n', f);
            fclose(f);
        } else {
            fprintf(stderr, "[CLI] ERROR: cannot open %s\n", out_path);
            ret = 1;
        }
    } else {
        printf("%s\n", text.c_str());
    }
    return ret;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[qwenasr-transcribe] FATAL: %s\n", e.what());
        return 1;
    }
}
