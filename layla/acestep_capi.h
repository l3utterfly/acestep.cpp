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
// use_gpu: threaded directly to the engine's DiT loader -- true runs DiT on the
//   accelerator pool (GPU, CPU fallback), false runs it on CPU. The VAE decoder
//   has a GPU path wired to this flag too, but is currently forced to CPU inside
//   the implementation because GPU VAE is slower. An externally-exported
//   GGML_BACKEND remains a developer-only override (=CPU forces CPU; =CUDA0/
//   =Vulkan0/etc. pins a device).
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

// Runs the reverse pipeline (the server's POST /understand): audio -> VAE encode
// -> FSQ tokenize -> LM, producing metadata + lyrics + caption + audio_codes for
// an existing track. Returns a JSON object body:
//   { "request": {...}, "latentPath": "...", "latentFrames": N,
//     "durationSeconds": D }
// `request` is a full AceRequest, directly feedable to acestep_run_synth().
//
// Exactly one source is required, mirroring the endpoint's multipart contract:
//   audio_path       WAV or MP3 file, any sample rate (resampled to 48 kHz).
//   src_latent_path  a raw latent file (f32 [T, 64] time-major, no header --
//                    the same bytes acestep_run_vae_encode() writes). When set,
//                    the VAE encoder pass is skipped. Wins when both are given.
// latent_out_path: optional. When non-empty AND the audio path was taken, the
//   latents that fed the tokenizer are written there in the wire format above,
//   so a later call can skip the encode. Ignored (nothing is written) when the
//   caller supplied src_latent_path -- the bytes would be what it just passed in.
// use_gpu: accepted for symmetry with the other entry points. The understand
//   pipeline's stages (VAE encoder, FSQ tokenizer, LM) all run on CPU today, so
//   the flag currently has no effect here.
ACESTEP_CAPI_EXPORT acestep_status acestep_run_understand(
    const char *lm_path,
    const char *dit_path,
    const char *vae_path,
    const char *request_json,
    const char *audio_path,
    const char *src_latent_path,
    const char *latent_out_path,
    bool use_flash_attn,
    int vae_tile_size,
    bool use_gpu,
    acestep_progress_cb progress, void *progress_ctx,
    acestep_stop_cb stop, void *stop_ctx,
    char **result, char **error);

// VAE encode -- the "audio in" half of the server's POST /vae. Decodes
// audio_path (WAV or MP3, any sample rate), resamples to 48 kHz stereo, runs the
// VAE encoder and writes the latents to latent_out_path as raw f32 [T, 64]
// time-major with no header: byte-identical to the endpoint's octet-stream body,
// and accepted back by acestep_run_vae_decode() / acestep_run_understand().
// Returns a JSON object body:
//   { "latentPath": "...", "latentFrames": N, "sampleRate": 48000,
//     "durationSeconds": D }
// request_json is parsed for validation only; the encode direction reads no
// field from it today (the VAE is selected by path, not by registry name).
// use_gpu is accepted for symmetry: the VAE encoder is CPU-only in the engine.
ACESTEP_CAPI_EXPORT acestep_status acestep_run_vae_encode(
    const char *vae_path,
    const char *request_json,
    const char *audio_path,
    const char *latent_out_path,
    int vae_tile_size,
    bool use_gpu,
    acestep_progress_cb progress, void *progress_ctx,
    acestep_stop_cb stop, void *stop_ctx,
    char **result, char **error);

// VAE decode -- the "latents in" half of the server's POST /vae. Reads a raw
// latent file (f32 [T, 64] time-major, no header), runs the VAE decoder and
// writes the rendered track to output_path, mp3 when the path ends in ".mp3"
// (same rule as acestep_run_synth). Honors output_format for the WAV subformat,
// plus mp3_bitrate and peak_clip from request_json. Returns a JSON object body:
//   { "outputPath": "...", "sampleRate": 48000, "numSamples": N,
//     "durationSeconds": D }
// use_gpu: as in acestep_run_synth, the VAE decoder is pinned to CPU inside the
// implementation because GPU VAE is currently slower; the GPU path stays wired.
ACESTEP_CAPI_EXPORT acestep_status acestep_run_vae_decode(
    const char *vae_path,
    const char *request_json,
    const char *latent_path,
    const char *output_path,
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
