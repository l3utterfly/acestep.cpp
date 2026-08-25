#pragma once
// weight-ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf-weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;  // byte offset into dst tensor (0 for regular loads)
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    struct UploadGroup {
        struct ggml_tensor *                        tensor;
        std::vector<const WeightCtx::PendingCopy *> copies;
        size_t                                      nbytes;
    };

    // Some model loaders fuse multiple source tensors (Q+K+V or gate+up) into
    // one destination tensor. Keep those source slices lightweight while the
    // model is being described, then coalesce them immediately before upload.
    // This is required by backends such as OpenCL's quantized SoA path, which
    // converts and replaces tensor->extra on the first set_tensor call and
    // therefore cannot accept a second partial upload for the same tensor.
    std::vector<UploadGroup> groups;
    groups.reserve(wctx->pending.size());
    std::unordered_map<struct ggml_tensor *, size_t> group_by_tensor;
    group_by_tensor.reserve(wctx->pending.size());

    size_t total = 0;
    for (const auto & pc : wctx->pending) {
        if (!pc.tensor || (!pc.src && pc.nbytes != 0)) {
            fprintf(stderr, "[WeightCtx] FATAL: invalid pending weight copy\n");
            return false;
        }

        auto [it, inserted] = group_by_tensor.emplace(pc.tensor, groups.size());
        if (inserted) {
            groups.push_back({ pc.tensor, {}, ggml_nbytes(pc.tensor) });
        }
        groups[it->second].copies.push_back(&pc);
        total += pc.nbytes;
    }

    // Validate before allocating backend memory or uploading any data. Requiring
    // exact, non-overlapping coverage prevents an incomplete fused tensor from
    // being silently zero-filled by the temporary coalescing buffer.
    for (auto & group : groups) {
        std::sort(group.copies.begin(), group.copies.end(),
                  [](const auto * lhs, const auto * rhs) { return lhs->offset < rhs->offset; });

        size_t covered = 0;
        for (const auto * pc : group.copies) {
            if (pc->offset != covered || pc->offset > group.nbytes || pc->nbytes > group.nbytes - pc->offset) {
                fprintf(stderr,
                        "[WeightCtx] FATAL: pending copies for tensor '%s' do not exactly cover %zu bytes "
                        "(expected offset %zu, got offset %zu size %zu)\n",
                        group.tensor->name, group.nbytes, covered, pc->offset, pc->nbytes);
                return false;
            }
            covered += pc->nbytes;
        }
        if (covered != group.nbytes) {
            fprintf(stderr, "[WeightCtx] FATAL: pending copies for tensor '%s' cover %zu of %zu bytes\n",
                    group.tensor->name, covered, group.nbytes);
            return false;
        }
    }

    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        fprintf(stderr, "[WeightCtx] FATAL: failed to allocate backend buffer\n");
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto & group : groups) {
        if (group.copies.size() == 1) {
            const auto * pc = group.copies.front();
            ggml_backend_tensor_set(group.tensor, pc->src, 0, group.nbytes);
            continue;
        }

        // Allocate only one fused tensor's staging memory at a time. Holding
        // staging for every fused model weight until wctx_alloc would add a
        // near-model-sized RAM spike on mobile devices.
        std::vector<uint8_t> coalesced(group.nbytes);
        for (const auto * pc : group.copies) {
            memcpy(coalesced.data() + pc->offset, pc->src, pc->nbytes);
        }
        ggml_backend_tensor_set(group.tensor, coalesced.data(), 0, group.nbytes);
    }
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend\n", groups.size(),
            (float) total / (1024 * 1024));
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
