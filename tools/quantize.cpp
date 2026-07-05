// quantize.cpp: GGUF requantizer for Qwen3-ASR. Reads the BF16 GGUF, writes a
// quantized GGUF with a mixed-precision K-quant policy mirroring
// llama-quantize: important tensors (attn_v, ffn_down) get bumped in S/M/L
// variants, the token embedding takes the embed type, norms and biases stay at
// their source dtype. Streaming write: one tensor at a time, low memory
// footprint.
//
// Usage: quantize <input.gguf> <output.gguf> <type>
// Types: Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#    define NOMINMAX
#    include <windows.h>
#    define strcasecmp _stricmp
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#include "ggml.h"
#include "gguf.h"
#include "utf8.h"
#include "version.h"

// Quant variant: base type + optional bump rules for important tensors
struct QuantVariant {
    const char *   name;
    enum ggml_type base;
    enum ggml_type bump;   // type for "important" tensors (or COUNT = no bump)
    enum ggml_type embed;  // type for embed_tokens (or COUNT = same as base)
    // bump_mode: 0=none, 1=first N layers, 2=first+last+every 3rd, 3=all
    // important
    int            bump_mode;
    int            bump_n;  // for mode 1: number of layers to bump
};

static const QuantVariant VARIANTS[] = {
    // name       base            bump            embed           mode  n
    { "BF16",   GGML_TYPE_BF16, GGML_TYPE_COUNT, GGML_TYPE_BF16, 0, 0 },
    { "Q2_K",   GGML_TYPE_Q2_K, GGML_TYPE_Q4_K,  GGML_TYPE_Q6_K, 1, 4 },
    { "Q3_K_S", GGML_TYPE_Q3_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0 },
    { "Q3_K_M", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 2, 0 },
    { "Q3_K_L", GGML_TYPE_Q3_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 3, 0 },
    { "Q4_K_S", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, 1, 4 },
    { "Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0 },
    { "Q5_K_S", GGML_TYPE_Q5_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0 },
    { "Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,  GGML_TYPE_Q6_K, 2, 0 },
    { "Q6_K",   GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_Q6_K, 0, 0 },
    { "Q8_0",   GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_Q8_0, 0, 0 },
};

static const QuantVariant * find_variant(const char * s) {
    for (const auto & v : VARIANTS) {
        if (strcasecmp(s, v.name) == 0) {
            return &v;
        }
    }
    return nullptr;
}

// Extract layer index from a llama.cpp tensor name: blk.N.xxx -> N, else -1.
static int extract_layer(const char * name) {
    const char * p = strstr(name, "blk.");
    if (!p) {
        return -1;
    }
    return atoi(p + 4);
}

// Important tensors for S/M: attn_v + ffn_down.
static bool is_important_sm(const char * name) {
    return (strstr(name, "attn_v.weight") != nullptr) || (strstr(name, "ffn_down.weight") != nullptr);
}

// Important tensors for L: attn_v + ffn_down + attn_output.
static bool is_important_l(const char * name) {
    return is_important_sm(name) || (strstr(name, "attn_output.weight") != nullptr);
}

// Token embedding, accessed via ggml_get_rows. It takes the embed type so the
// get_rows kernel reads a layout it supports.
static bool is_embed(const char * name) {
    return strstr(name, "token_embd.weight") != nullptr;
}

// Should this tensor be quantized at all? Norms and biases are 1D and stay at
// their source dtype. Conv kernels (ne0 = 3) do not divide a K-quant block and
// fall back to F16 in the main loop. Every other 2D tensor quantizes.
static bool should_quantize(int n_dims) {
    return n_dims >= 2;
}

// Decide target type for a single tensor given the variant + layer info
static enum ggml_type pick_type(const char * name, int n_dims, const QuantVariant & v, int n_layers) {
    if (!should_quantize(n_dims)) {
        return GGML_TYPE_COUNT;
    }

    // token embedding takes the embed type (get_rows safe layout)
    if (is_embed(name)) {
        return (v.embed != GGML_TYPE_COUNT) ? v.embed : v.base;
    }

    // Important tensor bump logic
    bool important = (v.bump_mode == 3) ? is_important_l(name) : is_important_sm(name);

    if (important && v.bump != GGML_TYPE_COUNT) {
        int  layer  = extract_layer(name);
        bool bumped = false;
        switch (v.bump_mode) {
            case 1:  // first N layers only
                bumped = (layer >= 0 && layer < v.bump_n);
                break;
            case 2:
                {  // M variant: first few + last few + every 3rd
                    int ql = n_layers;
                    bumped = (layer >= 0) && (layer < ql / 9 || layer >= ql - ql / 7 || layer % 3 == 0);
                    break;
                }
            case 3:  // L variant: all important tensors (v+down+o_proj)
                bumped = true;
                break;
        }
        if (bumped) {
            return v.bump;
        }
    }

    return v.base;
}

// Convert source data to F32
static bool to_f32(const void * src, float * dst, int64_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
            return true;
        case GGML_TYPE_F32:
            memcpy(dst, src, (size_t) n * sizeof(float));
            return true;
        default:
            return false;
    }
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    if (argc != 4) {
        fprintf(stderr, "qwenasr.cpp %s\n\n", QWENASR_VERSION);
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        fprintf(stderr, "Types:");
        for (const auto & v : VARIANTS) {
            fprintf(stderr, " %s", v.name);
        }
        fprintf(stderr, "\n");
        return 1;
    }

    const char *         inp_path = argv[1];
    const char *         out_path = argv[2];
    const QuantVariant * variant  = find_variant(argv[3]);

    if (!variant) {
        fprintf(stderr, "[Quantize] Unknown type: %s\n", argv[3]);
        return 1;
    }

    fprintf(stderr, "[Quantize] %s -> %s (%s)\n", inp_path, out_path, variant->name);

    // Mmap input file
#ifdef _WIN32
    std::wstring winp_path = utf8_to_wide(inp_path);
    HANDLE       fh =
        CreateFileW(winp_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[Quantize] Failed to open %s\n", inp_path);
        return 1;
    }
    HANDLE mh = CreateFileMappingW(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "[Quantize] CreateFileMapping failed %s\n", inp_path);
        CloseHandle(fh);
        return 1;
    }
    void * mapping = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!mapping) {
        fprintf(stderr, "[Quantize] MapViewOfFile failed %s\n", inp_path);
        CloseHandle(mh);
        CloseHandle(fh);
        return 1;
    }
#else
    int fd = open(inp_path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t) st.st_size;
    void * mapping   = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
#endif

    // Parse input GGUF
    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct ggml_context *   meta   = nullptr;
    params.ctx                     = &meta;

    struct gguf_context * inp = gguf_init_from_file(inp_path, params);
    if (!inp) {
        fprintf(stderr, "[Quantize] Failed to read %s\n", inp_path);
#ifdef _WIN32
        UnmapViewOfFile(mapping);
        CloseHandle(mh);
        CloseHandle(fh);
#else
        munmap(mapping, file_size);
        close(fd);
#endif
        return 1;
    }

    const size_t data_off  = gguf_get_data_offset(inp);
    const int    n_tensors = (int) gguf_get_n_tensors(inp);

    // Read architecture
    char arch[64] = "unknown";
    {
        int64_t idx = gguf_find_key(inp, "general.architecture");
        if (idx >= 0) {
            const char * s = gguf_get_val_str(inp, (int) idx);
            snprintf(arch, sizeof(arch), "%s", s);
        }
    }

    // Read block count for bump policy
    int n_layers = 0;
    {
        char key[128];
        snprintf(key, sizeof(key), "%s.block_count", arch);
        int64_t idx = gguf_find_key(inp, key);
        if (idx >= 0) {
            n_layers = (int) gguf_get_val_u32(inp, (int) idx);
        }
    }

    fprintf(stderr, "[Quantize] Arch=%s Layers=%d\n", arch, n_layers);

    // Create output GGUF: copy KV metadata
    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, inp);
    gguf_set_val_u32(out, "general.quantization_version", 2);
    gguf_set_val_str(out, "general.file_type", variant->name);

    // Plan: for each tensor, decide target type
    struct TensorPlan {
        enum ggml_type target;
        bool           quantize;
    };

    std::vector<TensorPlan> plans((size_t) n_tensors);

    for (int i = 0; i < n_tensors; i++) {
        const char *         name   = gguf_get_tensor_name(inp, i);
        struct ggml_tensor * t      = ggml_get_tensor(meta, name);
        const int            n_dims = ggml_n_dims(t);

        gguf_add_tensor(out, t);
        plans[(size_t) i] = { GGML_TYPE_COUNT, false };

        enum ggml_type target = pick_type(name, n_dims, *variant, n_layers);

        if (target == GGML_TYPE_COUNT) {
            continue;
        }

        bool can_convert = (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32);
        bool aligned     = (t->ne[0] % ggml_blck_size(target) == 0);

        // Conv kernels (K=7,3,1,...) cannot fit a block-quant row: fall back
        // to F16. F16 has no block size, 10-bit mantissa beats BF16 (7) and
        // Q* effective on these weights, and gf_load_conv_f16 memcpys F16
        // source straight to the F16 backend tensor at load time.
        if (can_convert && !aligned) {
            target  = GGML_TYPE_F16;
            aligned = true;
        }

        if (can_convert && aligned) {
            gguf_set_tensor_type(out, name, target);
            plans[(size_t) i] = { target, true };
        }
    }

    // Write metadata only (header + tensor info, no data)
    bool ok = gguf_write_to_file(out, out_path, true);
    if (!ok) {
        fprintf(stderr, "[Quantize] Failed to write metadata %s\n", out_path);
        return 1;
    }

    // Stream tensor data one at a time (low memory)
    FILE * fout = utf8_fopen(out_path, "ab");
    if (!fout) {
        fprintf(stderr, "[Quantize] Failed to open %s for append\n", out_path);
        return 1;
    }

    const size_t alignment   = gguf_get_alignment(out);
    int          n_quantized = 0;
    int64_t      bytes_in = 0, bytes_out = 0;
    size_t       data_pos = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char *         name     = gguf_get_tensor_name(inp, i);
        struct ggml_tensor * t        = ggml_get_tensor(meta, name);
        const int64_t        nel      = ggml_nelements(t);
        const size_t         src_size = ggml_nbytes(t);
        const size_t         t_off    = gguf_get_tensor_offset(inp, i);
        const void *         src      = (const uint8_t *) mapping + data_off + t_off;

        bytes_in += (int64_t) src_size;

        // Pad to alignment boundary
        size_t pad = (alignment - (data_pos % alignment)) % alignment;
        if (pad > 0) {
            uint8_t zeros[64] = {};
            fwrite(zeros, 1, pad, fout);
            data_pos += pad;
        }

        const TensorPlan & plan = plans[(size_t) i];

        if (plan.quantize) {
            // Quantize: src -> f32 -> target
            std::vector<float> f32((size_t) nel);
            to_f32(src, f32.data(), nel, t->type);

            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = nel / n_per_row;
            const size_t  qsize     = ggml_row_size(plan.target, n_per_row) * (size_t) nrows;

            std::vector<uint8_t> qbuf(qsize);
            ggml_quantize_chunk(plan.target, f32.data(), qbuf.data(), 0, nrows, n_per_row, nullptr);

            fwrite(qbuf.data(), 1, qsize, fout);
            data_pos += qsize;
            bytes_out += (int64_t) qsize;
            n_quantized++;
        } else {
            // Keep as-is
            fwrite(src, 1, src_size, fout);
            data_pos += src_size;
            bytes_out += (int64_t) src_size;
        }
    }

    fclose(fout);

    fprintf(stderr, "[Quantize] Quantized %d/%d tensors\n", n_quantized, n_tensors);
    fprintf(stderr, "[Quantize] %.1f GB -> %.1f GB (%.1fx)\n", (double) bytes_in / 1e9, (double) bytes_out / 1e9,
            bytes_out > 0 ? (double) bytes_in / (double) bytes_out : 0.0);
    fprintf(stderr, "[Quantize] Wrote %s\n", out_path);

    gguf_free(out);
    gguf_free(inp);
    ggml_free(meta);
#ifdef _WIN32
    UnmapViewOfFile(mapping);
    CloseHandle(mh);
    CloseHandle(fh);
#else
    munmap(mapping, file_size);
    close(fd);
#endif

    return 0;
}
