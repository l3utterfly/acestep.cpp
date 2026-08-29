// =============================================================================
// Layla ACE-Step C ABI implementation  (self-contained fork binding)
//
// All engine/ggml includes are confined to this TU so they compile inside
// libacestep_engine.so and never leak into the app's libappmodules.so. See
// acestep_capi.h for the rationale.
// =============================================================================
#include "acestep_capi.h"

// Engine (acestep-core) + ggml. Available on the engine include paths
// (src/, tools/, ggml/include) set by layla/CMakeLists.txt.
#include "audio-io.h"            // audio_write, audio_parse_format, WavFormat
#include "ggml-backend.h"        // ggml_backend_dev_* enumeration (device logging)
#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-synth.h"
#include "request.h"
#include "synth-batch-runner.h"  // synth_batch_run (tools/, header-only)
#include "task-types.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#define ACE_LOG_TAG "AceStep"
#define ACE_LOGI(...) __android_log_print(ANDROID_LOG_INFO, ACE_LOG_TAG, __VA_ARGS__)
#define ACE_LOGW(...) __android_log_print(ANDROID_LOG_WARN, ACE_LOG_TAG, __VA_ARGS__)
#define ACE_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, ACE_LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define ACE_LOGI(...) do { std::fprintf(stderr, "[AceStep] ");    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define ACE_LOGW(...) do { std::fprintf(stderr, "[AceStep][W] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define ACE_LOGE(...) do { std::fprintf(stderr, "[AceStep][E] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

namespace {

using ace_clock = std::chrono::steady_clock;

long ace_elapsed_ms(ace_clock::time_point start) {
  return static_cast<long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(ace_clock::now() - start).count());
}

std::string base_name(const std::string &path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string log_snippet(const std::string &s, size_t max = 80) {
  std::string one_line = s;
  for (char &c : one_line) {
    if (c == '\n' || c == '\r' || c == '\t') c = ' ';
  }
  if (one_line.size() > max) {
    one_line.resize(max);
    one_line += "...";
  }
  return one_line;
}

std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

bool resolve_lm_mode(const std::string &name, int *out_mode) {
  if (name.empty() || name == LM_MODE_NAME_GENERATE) {
    *out_mode = LM_MODE_GENERATE;
  } else if (name == LM_MODE_NAME_INSPIRE) {
    *out_mode = LM_MODE_INSPIRE;
  } else if (name == LM_MODE_NAME_FORMAT) {
    *out_mode = LM_MODE_FORMAT;
  } else {
    return false;
  }
  return true;
}

// ── C-callback bridges ───────────────────────────────────────────────────────
// The engine speaks its own AceProgress (C fn ptr + user_data) and a
// `bool cancel(void*)` poll. Bridge both onto the acestep_capi callbacks.
struct CallbackCtx {
  acestep_progress_cb progress;
  void               *progress_ctx;
  acestep_stop_cb     stop;
  void               *stop_ctx;
};

void progress_bridge(const char *status, int current, int total, void *user_data) {
  auto *cb = static_cast<CallbackCtx *>(user_data);
  if (cb && cb->progress) cb->progress(cb->progress_ctx, status, current, total);
}

bool stop_bridge(void *user_data) {
  auto *cb = static_cast<CallbackCtx *>(user_data);
  return cb && cb->stop && cb->stop(cb->stop_ctx) != 0;
}

char *dup_cstr(const std::string &s) {
  char *out = static_cast<char *>(std::malloc(s.size() + 1));
  if (out) std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

// Set *result on success; return ACESTEP_OK.
acestep_status ok(const std::string &body, char **result) {
  if (result) *result = dup_cstr(body);
  return ACESTEP_OK;
}

// Set *error; return ACESTEP_ERROR.
acestep_status err(const std::string &message, char **error) {
  if (error) *error = dup_cstr(message);
  return ACESTEP_ERROR;
}

}  // namespace

// =============================================================================
// acestep_run_lm  (moved from AceStep.cpp run_lm, adapted to the C ABI)
// =============================================================================
extern "C" acestep_status acestep_run_lm(
    const char *model_path,
    const char *request_json,
    acestep_progress_cb progress, void *progress_ctx,
    acestep_stop_cb stop, void *stop_ctx,
    char **result, char **error) {
  if (result) *result = nullptr;
  if (error)  *error  = nullptr;

  CallbackCtx cb{progress, progress_ctx, stop, stop_ctx};
  void *cancel_data = stop ? &cb : nullptr;

  ModelStore *store = nullptr;
  AceLm *ctx = nullptr;
  try {
    const std::string modelPath = model_path ? model_path : "";

    AceRequest req;
    if (!request_parse_json(&req, request_json ? request_json : "")) {
      return err("Invalid request JSON", error);
    }
    if (req.caption.empty()) {
      return err("Caption is required", error);
    }

    int mode = 0;
    if (!resolve_lm_mode(req.lm_mode, &mode)) {
      return err("Invalid lm_mode (use: generate, inspire, format)", error);
    }

    int lm_batch_size = req.lm_batch_size;
    if (lm_batch_size < 1) lm_batch_size = 1;
    else if (lm_batch_size > 9) lm_batch_size = 9;

    ACE_LOGI("LM: start (model=%s, mode=%s, batch=%d, caption=\"%s\")", base_name(modelPath).c_str(),
             req.lm_mode.empty() ? "generate" : req.lm_mode.c_str(), lm_batch_size,
             log_snippet(req.caption).c_str());

    AceLmParams params;
    ace_lm_default_params(&params);
    params.model_path = modelPath.c_str();
    params.max_batch = lm_batch_size;

    store = store_create(EVICT_STRICT);
    if (!store) return err("Failed to create model store", error);

    ACE_LOGI("LM: loading model...");
    if (progress) progress(progress_ctx, "Loading model", 0, 1);
    const auto t_load = ace_clock::now();
    ctx = ace_lm_load(store, &params);
    if (!ctx) {
      store_free(store);
      return err("Failed to load LM model: " + modelPath, error);
    }
    ACE_LOGI("LM: model loaded in %ld ms", ace_elapsed_ms(t_load));

    request_resolve_lm_seed(&req);
    request_resolve_seed(&req);
    std::vector<AceRequest> reqs(lm_batch_size, req);
    for (int b = 0; b < lm_batch_size; b++) {
      reqs[b].lm_seed = req.lm_seed + b;
      reqs[b].seed = req.seed + b;
    }

    ACE_LOGI("LM: generating %d variant(s) (lm_seed=%lld, seed=%lld)...", lm_batch_size,
             (long long) req.lm_seed, (long long) req.seed);
    const auto t_gen = ace_clock::now();
    std::vector<AceRequest> out(lm_batch_size);
    AceProgress prog{progress_bridge, &cb};
    const int rc = ace_lm_generate(ctx, reqs.data(), lm_batch_size, out.data(),
                                   nullptr, nullptr,
                                   cancel_data ? stop_bridge : nullptr, cancel_data, mode,
                                   progress ? &prog : nullptr);
    ace_lm_free(ctx);
    ctx = nullptr;
    store_free(store);
    store = nullptr;

    if (rc != 0) {
      if (stop_bridge(cancel_data)) {
        ACE_LOGI("LM: generation stopped by request after %ld ms", ace_elapsed_ms(t_gen));
        return ACESTEP_CANCELLED;
      }
      ACE_LOGE("LM: generation failed (rc=%d) after %ld ms", rc, ace_elapsed_ms(t_gen));
      return err("LM generation failed", error);
    }
    ACE_LOGI("LM: done (%d variant(s)) in %ld ms", lm_batch_size, ace_elapsed_ms(t_gen));

    std::string body = "[";
    for (int i = 0; i < lm_batch_size; i++) {
      if (i > 0) body += ",";
      body += request_to_json(&out[i]);
    }
    body += "]";
    return ok(body, result);
  } catch (const std::exception &e) {
    if (ctx) ace_lm_free(ctx);
    if (store) store_free(store);
    return err(e.what(), error);
  } catch (...) {
    if (ctx) ace_lm_free(ctx);
    if (store) store_free(store);
    return err("Unknown ACE-Step LM error", error);
  }
}

// =============================================================================
// acestep_run_synth  (moved from AceStep.cpp run_synth, adapted to the C ABI)
// =============================================================================
extern "C" acestep_status acestep_run_synth(
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
    char **result, char **error) {
  if (result) *result = nullptr;
  if (error)  *error  = nullptr;

  CallbackCtx cb{progress, progress_ctx, stop, stop_ctx};
  void *cancel_data = stop ? &cb : nullptr;

  ModelStore *store = nullptr;
  AceSynth *ctx = nullptr;
  AceAudio audio = {};
  try {
    const std::string outputPath = output_path ? output_path : "";

    AceRequest req;
    if (!request_parse_json(&req, request_json ? request_json : "")) {
      return err("Invalid request JSON", error);
    }
    if (req.caption.empty()) {
      return err("Caption is required", error);
    }

    bool is_mp3 = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(req.output_format.empty() ? nullptr : req.output_format.c_str(), is_mp3, wav_fmt)) {
      return err("Invalid output_format (use: mp3, wav16, wav24, wav32)", error);
    }

    ACE_LOGI("Synth: start (format=%s, steps=%d, guidance=%.2f, shift=%.2f, solver=%s, "
             "duration=%.1fs, caption=\"%s\")",
             req.output_format.empty() ? "mp3" : req.output_format.c_str(), req.inference_steps,
             req.guidance_scale, req.shift, req.solver.empty() ? "euler" : req.solver.c_str(),
             req.duration, log_snippet(req.caption).c_str());

    AceSynthParams params;
    ace_synth_default_params(&params);
    params.text_encoder_path = text_encoder_path ? text_encoder_path : "";
    params.dit_path = dit_path ? dit_path : "";
    params.vae_path = vae_path ? vae_path : "";
    params.adapter_scale = req.adapter_scale;
    params.use_fa = use_flash_attn;
    if (vae_tile_size > 0) params.vae_chunk = vae_tile_size;
    params.vae_use_gpu = use_gpu;
    // On GPU the VAE decoder's activation buffers must stay within mobile GPU
    // per-allocation limits, so cap the tile at 256 latent frames.
    if (use_gpu && params.vae_chunk > 256) params.vae_chunk = 256;
    // Overlap is always 1/16 of the VAE tile size (kept in lockstep with the tile).
    params.vae_overlap = params.vae_chunk / 16;
    ACE_LOGI("Synth: flash_attn=%s, vae_gpu=%s, vae_tile=%d, vae_overlap=%d", params.use_fa ? "on" : "off",
             params.vae_use_gpu ? "on" : "off", params.vae_chunk, params.vae_overlap);

    store = store_create(EVICT_STRICT);
    if (!store) return err("Failed to create model store", error);

    ACE_LOGI("Synth: loading models (text-encoder/DiT/VAE)...");
    if (progress) progress(progress_ctx, "Loading models", 0, 1);
    const auto t_load = ace_clock::now();
    ctx = ace_synth_load(store, &params);
    if (!ctx) {
      store_free(store);
      return err("Failed to load synth models (text-encoder/DiT/VAE)", error);
    }
    ACE_LOGI("Synth: models loaded in %ld ms", ace_elapsed_ms(t_load));

    AceProgress prog{progress_bridge, &cb};
    ace_synth_set_progress(ctx, progress ? &prog : nullptr);

    request_resolve_seed(&req);
    std::vector<std::vector<AceRequest>> groups(1);
    groups[0].push_back(req);

    ACE_LOGI("Synth: running Text-Encoder + DiT + VAE (seed=%lld)...", (long long) groups[0][0].seed);
    const auto t_synth = ace_clock::now();
    const int rc = synth_batch_run(ctx, groups, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, &audio,
                                   nullptr, cancel_data ? stop_bridge : nullptr, cancel_data);
    ace_synth_free(ctx);
    ctx = nullptr;
    store_free(store);
    store = nullptr;

    if (rc != 0 || !audio.samples) {
      const bool cancelled = stop_bridge(cancel_data);
      ace_audio_free(&audio);
      if (cancelled) {
        ACE_LOGI("Synth: synthesis stopped by request after %ld ms", ace_elapsed_ms(t_synth));
        return ACESTEP_CANCELLED;
      }
      ACE_LOGE("Synth: synthesis failed (rc=%d) after %ld ms", rc, ace_elapsed_ms(t_synth));
      return err("Synthesis failed", error);
    }
    ACE_LOGI("Synth: render done in %ld ms (%d samples @ %d Hz)", ace_elapsed_ms(t_synth),
             audio.n_samples, audio.sample_rate);

    ACE_LOGI("Synth: writing %s -> %s", is_mp3 ? "mp3" : "wav", base_name(outputPath).c_str());
    const auto t_write = ace_clock::now();
    const bool wrote = audio_write(outputPath.c_str(), audio.samples, audio.n_samples, audio.sample_rate,
                                   req.mp3_bitrate, wav_fmt);
    const int n_samples = audio.n_samples;
    const int sample_rate = audio.sample_rate;
    ace_audio_free(&audio);
    audio = {};

    if (!wrote) {
      ACE_LOGE("Synth: failed to write output: %s", outputPath.c_str());
      return err("Failed to write output: " + outputPath, error);
    }
    ACE_LOGI("Synth: wrote output in %ld ms", ace_elapsed_ms(t_write));

    const double duration = sample_rate > 0 ? (double) n_samples / (double) sample_rate : 0.0;
    ACE_LOGI("Synth: complete (%.1fs of audio)", duration);

    std::string out = "{";
    out += "\"outputPath\":\"" + json_escape(outputPath) + "\",";
    out += "\"seed\":" + std::to_string(groups[0][0].seed) + ",";
    out += "\"sampleRate\":" + std::to_string(sample_rate) + ",";
    out += "\"numSamples\":" + std::to_string(n_samples) + ",";
    out += "\"durationSeconds\":" + std::to_string(duration) + ",";
    out += "\"request\":" + request_to_json(&groups[0][0]);
    out += "}";
    return ok(out, result);
  } catch (const std::exception &e) {
    if (ctx) ace_synth_free(ctx);
    if (store) store_free(store);
    ace_audio_free(&audio);
    return err(e.what(), error);
  } catch (...) {
    if (ctx) ace_synth_free(ctx);
    if (store) store_free(store);
    ace_audio_free(&audio);
    return err("Unknown ACE-Step synth error", error);
  }
}

// =============================================================================
// acestep_log_devices  (moved from AceStep.cpp log_backend_devices)
// =============================================================================
extern "C" void acestep_log_devices(void) {
  static bool logged = false;
  if (logged) return;
  logged = true;

  const size_t n = ggml_backend_dev_count();
  ACE_LOGI("Backends: %zu ggml compute device(s) registered", n);
  bool any_gpu = false;
  for (size_t i = 0; i < n; i++) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (!dev) continue;
    const char *name = ggml_backend_dev_name(dev);
    const char *desc = ggml_backend_dev_description(dev);
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    const char *type_str = type == GGML_BACKEND_DEVICE_TYPE_GPU    ? "GPU"
                           : type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "iGPU"
                           : type == GGML_BACKEND_DEVICE_TYPE_CPU  ? "CPU"
                                                                   : "ACCEL";
    if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) any_gpu = true;
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    ACE_LOGI("  device[%zu]: type=%s name=%s desc=\"%s\" mem=%zu/%zu MiB", i, type_str,
             name ? name : "?", desc ? desc : "?", free_mem >> 20, total_mem >> 20);
  }
  if (!any_gpu) {
    ACE_LOGW("Backends: no GPU/OpenCL device registered -- generation will run on CPU. "
             "On Adreno this usually means the OpenCL driver (libOpenCL.so) was not found "
             "or clGetPlatformIDs returned no platforms.");
  }
}

extern "C" void acestep_string_free(char *s) {
  std::free(s);
}
