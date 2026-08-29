#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// On Android, the engine's diagnostics (ggml log + the [Load] backend line) go to
// stderr, which the platform discards -- so nothing about backend selection ever
// reaches logcat. Mirror them to logcat so it is visible whether OpenCL actually
// came up and which backend each stage ran on. Filter with: adb logcat -s AceStep-ggml:V AceStep:V
#if defined(__ANDROID__)
#include <android/log.h>
#endif

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// CPU-only stages and the DiT stage must not share a backend cache: otherwise
// whichever stage loads first selects the backend (and weight location) for all
// later stages. The accelerated pool is used only by DiT; every other stage
// uses the CPU pool.
struct BackendCache {
    BackendPair pair;
    int         refs;
};

static BackendCache g_backend_caches[2] = {};

// Physical core count heuristic (logical / 2 for HT/SMT).
// Used for GGML CPU thread count: GEMM shares SIMD units across hyperthreads,
// so one thread per physical core is optimal.
static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
    }
    return cpu;
}

// Collapse exact consecutive duplicate ggml log lines and report the total
// count when the run ends (tames the CUDA graph capture "reused" flood).
static void acestep_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    static char last[256] = { 0 };
    static int  count     = 0;

    if (count > 0 && strcmp(text, last) == 0) {
        count++;
        return;
    }

    if (count > 1) {
        fprintf(stderr, "[Dedup] Previous line repeated %d times total\n", count);
    }

    fputs(text, stderr);
#if defined(__ANDROID__)
    // Surface ggml's own device-detection / OpenCL-init messages to logcat.
    __android_log_write(ANDROID_LOG_INFO, "AceStep-ggml", text);
#endif
    strncpy(last, text, sizeof(last) - 1);
    last[sizeof(last) - 1] = 0;
    count                  = 1;
    fflush(stderr);
}

// Initialize a real CPU backend, or (when use_gpu is true) select the best
// available accelerator and keep CPU as its scheduler fallback.
// label:   log prefix, e.g. "DiT", "VAE", "LM"
// use_gpu: run this module on the accelerator pool. Threaded from the caller's
//          user flag; only DiT and the VAE decoder ever pass true (every other
//          stage is CPU-only by policy).
// GGML_BACKEND is now a developer override only: =CPU still forces even a
// use_gpu=true module onto CPU, and =CUDA0/=Vulkan0/etc. picks a specific device
// instead of ggml_backend_init_best(). It no longer gates GPU on/off.
static BackendPair backend_init(const char * label, bool use_gpu) {
    static bool log_installed = false;
    if (!log_installed) {
        ggml_log_set(acestep_ggml_log, nullptr);
        log_installed = true;
    }

    const char * force_backend = std::getenv("GGML_BACKEND");
    const bool   force_cpu     = force_backend && strcmp(force_backend, "CPU") == 0;
    const bool   use_gpu_pool  = use_gpu && !force_cpu;
    BackendCache & cache       = g_backend_caches[use_gpu_pool ? 1 : 0];

    if (cache.refs > 0) {
        cache.refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared, %s)\n", label,
                ggml_backend_name(cache.pair.backend), use_gpu_pool ? "accelerator pool" : "CPU pool");
#if defined(__ANDROID__)
        __android_log_print(ANDROID_LOG_INFO, "AceStep", "Engine backend [%s]: %s (shared, %s)",
                            label, ggml_backend_name(cache.pair.backend),
                            cache.pair.has_gpu ? "GPU" : "CPU-only");
#endif
        return cache.pair;
    }

    ggml_backend_load_all();
    BackendPair bp = {};

    if (!use_gpu_pool) {
        bp.backend     = cpu_backend_new(backend_cpu_n_threads());
        bp.cpu_backend = bp.backend;
    } else {
        // GGML_BACKEND may force a specific accelerator for DiT. Device names:
        // CUDA0, Vulkan0, etc. (see ggml_backend_dev_name).
        if (force_backend) {
            bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
            if (!bp.backend) {
                fprintf(stderr, "[Load] FATAL: GGML_BACKEND=%s not found. Available:", force_backend);
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
                }
                fprintf(stderr, "\n");
                exit(1);
            }
        } else {
            bp.backend = ggml_backend_init_best();
        }
    }
    if (!bp.backend) {
        fprintf(stderr, "[Load] FATAL: no backend available\n");
        exit(1);
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (!use_gpu_pool) {
        // The configured standalone CPU backend is already ready.
    } else if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        fprintf(stderr, "[Load] FATAL: failed to init CPU backend\n");
        exit(1);
    }
    bp.has_gpu = !best_is_cpu;
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n", label, ggml_backend_name(bp.backend), n_threads);
#if defined(__ANDROID__)
    // The decisive line: non-DiT stages always report CPU; DiT reports the
    // selected accelerator or CPU when disabled/unavailable.
    __android_log_print(ANDROID_LOG_INFO, "AceStep", "Engine backend [%s]: %s (%s, CPU threads: %d)",
                        label, ggml_backend_name(bp.backend), bp.has_gpu ? "GPU" : "CPU-only", n_threads);
#endif

    cache.pair = bp;
    cache.refs = 1;
    return bp;
}

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    for (BackendCache & cache : g_backend_caches) {
        if (cache.refs <= 0 || cache.pair.backend != backend || cache.pair.cpu_backend != cpu_backend) {
            continue;
        }
        cache.refs--;
        if (cache.refs == 0) {
            if (backend && backend != cpu_backend) {
                ggml_backend_free(backend);
            }
            if (cpu_backend) {
                ggml_backend_free(cpu_backend);
            }
            cache.pair = {};
        }
        return;
    }
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        fprintf(stderr, "[Load] FATAL: failed to create scheduler\n");
        exit(1);
    }
    return sched;
}
