// =============================================================================
// Layla ACE-Step C ABI  (self-contained fork binding)
//
// This is the ONLY Layla-specific addition inside the acestep.cpp fork. It lives
// under layla/ so upstream merges never touch it, and it exposes a small, stable
// extern "C" surface so the React-Native TurboModule glue (modules/ace-step/cpp)
// can drive the engine WITHOUT including any engine or ggml headers itself.
//
// Why this exists: the engine (acestep-core + its vendored ggml, incl. the
// OpenCL backend) is linked into a self-contained shared library,
// libacestep_engine.so, that is separate from the app's libappmodules.so. That
// isolation is what lets the OpenCL backend import libOpenCL.so (DT_NEEDED ->
// device vendor driver) without colliding with MNN's cl* symbols and without a
// nested ICD loader. For that isolation to hold, ggml/cl* symbols must stay
// OUT of libappmodules.so — so the glue may only speak this C ABI, and every
// engine/ggml include stays behind it in acestep_capi.cpp.
//
// String results are heap-allocated by the engine and must be released with
// acestep_string_free().
// =============================================================================
#ifndef ACESTEP_LAYLA_CAPI_H
#define ACESTEP_LAYLA_CAPI_H

#include <stdbool.h>

// The engine .so hides everything by default (version script + -fvisibility);
// force DEFAULT visibility on the exported API so it survives regardless. A
// hidden-visibility symbol is dropped from the dynamic table even if a version
// script lists it as global, so this attribute is what actually exports the ABI.
#ifndef ACESTEP_CAPI_EXPORT
#  if defined(_WIN32)
#    define ACESTEP_CAPI_EXPORT
#  else
#    define ACESTEP_CAPI_EXPORT __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Phase-level progress. ctx is the opaque pointer passed to the run_* call.
typedef void (*acestep_progress_cb)(void *ctx, const char *status, int current, int total);

// Cancellation poll: return non-zero to request a clean stop. ctx is the opaque
// pointer passed to the run_* call. Polled between LM tokens / DiT steps.
typedef int (*acestep_stop_cb)(void *ctx);

typedef enum {
    ACESTEP_OK        = 0,  // success; *result set to a malloc'd JSON string
    ACESTEP_ERROR     = 1,  // failure;  *error  set to a malloc'd message
    ACESTEP_CANCELLED = 2,  // stopped via stop_cb; no result/error
} acestep_status;

// Runs the LM stage. Returns a JSON array body (as the server's /lm produces).
// `progress`/`stop` may be NULL. `result`/`error` out-pointers may be NULL to
// ignore; when non-NULL, the set one owns a malloc'd string (acestep_string_free).
ACESTEP_CAPI_EXPORT acestep_status acestep_run_lm(
    const char *model_path,
    const char *request_json,
    acestep_progress_cb progress, void *progress_ctx,
    acestep_stop_cb stop, void *stop_ctx,
    char **result, char **error);

// Runs Text-Encoder + DiT + VAE for one request, writes audio to output_path,
// and returns a JSON metadata body. Callback/out-pointer contract as above.
// use_gpu: when true the VAE decoder joins the shared accelerator pool (GPU) and
//   the VAE tile size is capped at 256 (overlap tracks it at 1/16); when false the
//   decoder stays on a dedicated CPU backend. DiT GPU selection is still governed
//   by the GGML_BACKEND env var set by the caller.
ACESTEP_CAPI_EXPORT acestep_status acestep_run_synth(
    const char *text_encoder_path,
    const char *dit_path,
    const char *vae_path,
    const char *request_json,
    const char *output_path,
    bool use_flash_attn,
    int vae_tile_size,
    bool use_gpu,
    acestep_progress_cb progress, void *progress_ctx,
    acestep_stop_cb stop, void *stop_ctx,
    char **result, char **error);

// Logs the ggml compute devices the statically-linked backends registered
// (diagnostic; on Android -> logcat tag "AceStep"). Safe to call repeatedly;
// enumerates once per process.
ACESTEP_CAPI_EXPORT void acestep_log_devices(void);

// Releases a string returned through a run_* result/error out-pointer.
ACESTEP_CAPI_EXPORT void acestep_string_free(char *s);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ACESTEP_LAYLA_CAPI_H
