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
#include "pipeline-understand.h"
#include "request.h"
#include "synth-batch-runner.h"  // synth_batch_run (tools/, header-only)
#include "task-types.h"
#include "vae-enc.h"             // vae_enc_encode_tiled
#include "vae.h"                 // vae_ggml_decode_tiled

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

// ── Latent wire format ───────────────────────────────────────────────────────
// Raw f32 [T, 64] time-major with no header -- the exact octet-stream body the
// server's /vae encode returns and /vae decode + /understand accept. Keeping the
// on-disk format identical means a latent produced here is interchangeable with
// one produced by upstream, and with ace_synth_job_get_latent's layout.
constexpr int LATENT_CHANNELS    = 64;
constexpr int LATENT_FRAME_BYTES = LATENT_CHANNELS * (int) sizeof(float);
// 15000 frames @ 25 Hz = 10 minutes, matching the server's cap. One latent frame
// covers 1920 audio samples at 48 kHz.
constexpr int MAX_T_LATENT       = 15000;
constexpr int SAMPLES_PER_FRAME  = 1920;
constexpr int SAMPLE_RATE        = 48000;

bool read_whole_file(const std::string &path, std::string &out) {
  FILE *fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  if (std::fseek(fp, 0, SEEK_END) != 0) { std::fclose(fp); return false; }
  const long size = std::ftell(fp);
  if (size < 0) { std::fclose(fp); return false; }
  std::rewind(fp);
  out.resize((size_t) size);
  const size_t read = size > 0 ? std::fread(&out[0], 1, (size_t) size, fp) : 0;
  std::fclose(fp);
  return read == (size_t) size;
}

bool write_whole_file(const std::string &path, const void *data, size_t size) {
  FILE *fp = std::fopen(path.c_str(), "wb");
  if (!fp) return false;
  const size_t wrote = size > 0 ? std::fwrite(data, 1, size, fp) : 0;
  const bool flushed = std::fclose(fp) == 0;
  return wrote == size && flushed;
}

// Read a latent file in the wire format above. On failure returns false and
// sets `msg` to a message that names what was wrong with this specific file.
bool read_latent_file(const std::string &path, std::vector<float> &out, int &T_out, std::string &msg) {
  std::string bytes;
  if (!read_whole_file(path, bytes)) {
    msg = "Cannot read latent file: " + path;
    return false;
  }
  if (bytes.empty() || bytes.size() % (size_t) LATENT_FRAME_BYTES != 0) {
    msg = "Latent file size is not a non-zero multiple of 64*4 bytes: " + path;
    return false;
  }
  const size_t frames = bytes.size() / (size_t) LATENT_FRAME_BYTES;
  if (frames > (size_t) MAX_T_LATENT) {
    msg = "Latent file exceeds the max of 15000 frames (10 min): " + path;
    return false;
  }
  T_out = (int) frames;
  out.resize(frames * (size_t) LATENT_CHANNELS);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return true;
}

bool write_latent_file(const std::string &path, const std::vector<float> &latent, int T) {
  return write_whole_file(path, latent.data(), (size_t) T * (size_t) LATENT_FRAME_BYTES);
}

// Decode an audio file to 48 kHz interleaved stereo, the layout every pipeline
// entry point takes. WAV vs MP3 is auto-detected from the magic bytes rather
// than the extension, so a mislabeled file still loads. Returns a malloc'd
// buffer the caller frees, or NULL with `msg` set.
float *read_audio_48k_interleaved(const std::string &path, int &T_out, std::string &msg) {
  std::string bytes;
  if (!read_whole_file(path, bytes)) {
    msg = "Cannot read audio file: " + path;
    return nullptr;
  }
  int T_audio = 0;
  float *planar = audio_read_48k_buf((const uint8_t *) bytes.data(), bytes.size(), &T_audio);
  if (!planar || T_audio <= 0) {
    std::free(planar);
    msg = "Failed to decode audio (expected WAV or MP3): " + path;
    return nullptr;
  }
  if ((int64_t) T_audio / SAMPLES_PER_FRAME >= (int64_t) MAX_T_LATENT) {
    std::free(planar);
    msg = "Audio exceeds the max duration of 10 min: " + path;
    return nullptr;
  }
  float *interleaved = audio_planar_to_interleaved(planar, T_audio);
  std::free(planar);
  if (!interleaved) {
    msg = "Out of memory converting audio to interleaved stereo";
    return nullptr;
  }
  T_out = T_audio;
  return interleaved;
}

// Same tiling rule acestep_run_synth applies: <= 0 falls back to the engine
// default, GPU caps the tile so the decoder's activation buffers stay inside
// mobile per-allocation limits, and the overlap is always 1/16 of the tile.
void resolve_vae_tiles(int vae_tile_size, bool vae_use_gpu, int *chunk, int *overlap) {
  int c = vae_tile_size > 0 ? vae_tile_size : 1024;
  if (vae_use_gpu && c > 256) c = 256;
  *chunk   = c;
  *overlap = c / 16;
}

// Frees the interleaved audio buffer read by read_audio_48k_interleaved on
// every exit path, including the throwing ones.
struct AudioBufGuard {
  float *p;

  ~AudioBufGuard() { std::free(p); }
};

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
    // DiT follows the user's GPU flag directly (threaded to backend_init).
    params.dit_use_gpu = use_gpu;
    // GPU VAE is currently slower than CPU, so force the decoder onto the CPU
    // regardless of the caller's use_gpu request. The full GPU path (backend
    // selection + 256 tile cap) is retained below and reachable by flipping this
    // back to `use_gpu` once GPU VAE is worth enabling.
    const bool vae_use_gpu = false;
    params.vae_use_gpu = vae_use_gpu;
    // On GPU the VAE decoder's activation buffers must stay within mobile GPU
    // per-allocation limits, so cap the tile at 256 latent frames.
    if (vae_use_gpu && params.vae_chunk > 256) params.vae_chunk = 256;
    // Overlap is always 1/16 of the VAE tile size (kept in lockstep with the tile).
    params.vae_overlap = params.vae_chunk / 16;
    ACE_LOGI("Synth: flash_attn=%s, dit_gpu=%s, vae_gpu=%s, vae_tile=%d, vae_overlap=%d", params.use_fa ? "on" : "off",
             params.dit_use_gpu ? "on" : "off", params.vae_use_gpu ? "on" : "off", params.vae_chunk,
             params.vae_overlap);

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
// acestep_run_understand  (the server's POST /understand, in process)
// =============================================================================
extern "C" acestep_status acestep_run_understand(
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
    char **result, char **error) {
  if (result) *result = nullptr;
  if (error)  *error  = nullptr;

  CallbackCtx cb{progress, progress_ctx, stop, stop_ctx};
  void *cancel_data = stop ? &cb : nullptr;

  // The understand pipeline runs entirely on CPU today (VAE encoder, FSQ
  // tokenizer and LM all take CPU-only paths in the engine), so the caller's GPU
  // flag has nothing to steer here. Kept on the signature for symmetry.
  (void) use_gpu;

  ModelStore *store = nullptr;
  AceUnderstand *ctx = nullptr;
  float *src_interleaved = nullptr;
  try {
    const std::string audioPath     = audio_path ? audio_path : "";
    const std::string srcLatentPath = src_latent_path ? src_latent_path : "";
    const std::string latentOutPath = latent_out_path ? latent_out_path : "";

    AceRequest req;
    if (!request_parse_json(&req, request_json && *request_json ? request_json : "{}")) {
      return err("Invalid request JSON", error);
    }

    // Understand samples much colder than generation. The endpoint applies these
    // before parsing the body, which means a supplied body silently resets them;
    // here they are applied after the parse and only where the caller left the
    // generation defaults untouched, so an explicit value in the JSON still wins.
    {
      AceRequest defaults;
      request_init(&defaults);
      if (req.lm_temperature == defaults.lm_temperature) req.lm_temperature = 0.3f;
      if (req.lm_top_p == defaults.lm_top_p) req.lm_top_p = 1.0f;
    }

    // Exactly one source, latents winning when both are set (the endpoint's rule).
    std::vector<float> src_latents;
    int src_T_latent = 0;
    int src_len = 0;
    if (!srcLatentPath.empty()) {
      std::string msg;
      if (!read_latent_file(srcLatentPath, src_latents, src_T_latent, msg)) {
        return err(msg, error);
      }
      ACE_LOGI("Understand: source = %d latent frames (%.2fs) from %s", src_T_latent,
               (float) src_T_latent * (float) SAMPLES_PER_FRAME / (float) SAMPLE_RATE,
               base_name(srcLatentPath).c_str());
    } else if (!audioPath.empty()) {
      std::string msg;
      src_interleaved = read_audio_48k_interleaved(audioPath, src_len, msg);
      if (!src_interleaved) {
        return err(msg, error);
      }
      ACE_LOGI("Understand: source = %.2fs @ 48kHz from %s", (float) src_len / (float) SAMPLE_RATE,
               base_name(audioPath).c_str());
    } else {
      return err("Provide one of audioPath (audio in) or srcLatentPath (latents in)", error);
    }
    AudioBufGuard audio_guard{src_interleaved};

    const auto t_total = ace_clock::now();

    AceUnderstandParams params;
    ace_understand_default_params(&params);
    params.model_path = lm_path ? lm_path : "";
    params.dit_path   = dit_path ? dit_path : "";
    params.vae_path   = vae_path ? vae_path : "";
    params.use_fa     = use_flash_attn;
    // The VAE encoder is CPU-only here, so tile for CPU (no 256-frame GPU cap).
    resolve_vae_tiles(vae_tile_size, /*vae_use_gpu=*/false, &params.vae_chunk, &params.vae_overlap);
    ACE_LOGI("Understand: flash_attn=%s, vae_tile=%d, vae_overlap=%d", params.use_fa ? "on" : "off",
             params.vae_chunk, params.vae_overlap);

    store = store_create(EVICT_STRICT);
    if (!store) return err("Failed to create model store", error);

    ACE_LOGI("Understand: loading models (LM/DiT tokenizer/VAE)...");
    if (progress) progress(progress_ctx, "Loading models", 0, 1);
    const auto t_load = ace_clock::now();
    ctx = ace_understand_load(store, &params);
    if (!ctx) {
      store_free(store);
      store = nullptr;
      return err("Failed to load understand models (LM/DiT/VAE)", error);
    }
    ACE_LOGI("Understand: models loaded in %ld ms", ace_elapsed_ms(t_load));

    request_resolve_lm_seed(&req);

    // ace_understand_generate has no progress hook of its own, so the whole run
    // is reported as one indeterminate phase.
    if (progress) progress(progress_ctx, "Analyzing audio", 0, 1);
    ACE_LOGI("Understand: running VAE encode + FSQ + LM (lm_seed=%lld)...", (long long) req.lm_seed);
    const auto t_run = ace_clock::now();

    AceRequest out;
    std::vector<float> captured_latent;
    int captured_T_latent = 0;
    const int rc = ace_understand_generate(
        ctx, src_interleaved, src_len,
        src_latents.empty() ? nullptr : src_latents.data(), src_T_latent,
        &req, &out, &captured_latent, &captured_T_latent,
        cancel_data ? stop_bridge : nullptr, cancel_data);

    ace_understand_free(ctx);
    ctx = nullptr;
    store_free(store);
    store = nullptr;

    if (rc != 0) {
      if (stop_bridge(cancel_data)) {
        ACE_LOGI("Understand: stopped by request after %ld ms", ace_elapsed_ms(t_run));
        return ACESTEP_CANCELLED;
      }
      ACE_LOGE("Understand: failed (rc=%d) after %ld ms", rc, ace_elapsed_ms(t_run));
      return err("Understand failed", error);
    }
    if (progress) progress(progress_ctx, "Analyzing audio", 1, 1);
    ACE_LOGI("Understand: done in %ld ms (caption=\"%s\")", ace_elapsed_ms(t_run),
             log_snippet(out.caption).c_str());

    // The capture is only populated on the audio-in path: on the latents-in path
    // the buffer would be a copy of what the caller just supplied.
    std::string written_latent_path;
    int written_T_latent = 0;
    if (!latentOutPath.empty() && captured_T_latent > 0) {
      if (!write_latent_file(latentOutPath, captured_latent, captured_T_latent)) {
        return err("Failed to write latents: " + latentOutPath, error);
      }
      written_latent_path = latentOutPath;
      written_T_latent    = captured_T_latent;
      ACE_LOGI("Understand: wrote %d latent frames -> %s", written_T_latent,
               base_name(latentOutPath).c_str());
    }

    const int T_latent = src_T_latent > 0 ? src_T_latent : captured_T_latent;
    const double duration = src_len > 0
                                ? (double) src_len / (double) SAMPLE_RATE
                                : (double) T_latent * (double) SAMPLES_PER_FRAME / (double) SAMPLE_RATE;
    ACE_LOGI("Understand: complete in %ld ms total", ace_elapsed_ms(t_total));

    std::string body = "{";
    body += "\"request\":" + request_to_json(&out) + ",";
    body += "\"latentPath\":\"" + json_escape(written_latent_path) + "\",";
    body += "\"latentFrames\":" + std::to_string(written_T_latent) + ",";
    body += "\"durationSeconds\":" + std::to_string(duration);
    body += "}";
    return ok(body, result);
  } catch (const std::exception &e) {
    if (ctx) ace_understand_free(ctx);
    if (store) store_free(store);
    return err(e.what(), error);
  } catch (...) {
    if (ctx) ace_understand_free(ctx);
    if (store) store_free(store);
    return err("Unknown ACE-Step understand error", error);
  }
}

// =============================================================================
// acestep_run_vae_encode  (the "audio in" half of the server's POST /vae)
// =============================================================================
extern "C" acestep_status acestep_run_vae_encode(
    const char *vae_path,
    const char *request_json,
    const char *audio_path,
    const char *latent_out_path,
    int vae_tile_size,
    bool use_gpu,
    acestep_progress_cb progress, void *progress_ctx,
    acestep_stop_cb stop, void *stop_ctx,
    char **result, char **error) {
  if (result) *result = nullptr;
  if (error)  *error  = nullptr;

  CallbackCtx cb{progress, progress_ctx, stop, stop_ctx};
  void *cancel_data = stop ? &cb : nullptr;

  // store_require_vae_enc has no GPU path in the engine: the encoder always runs
  // on CPU. The flag stays on the signature for symmetry with the decoder.
  (void) use_gpu;

  ModelStore *store = nullptr;
  float *src_interleaved = nullptr;
  try {
    const std::string audioPath     = audio_path ? audio_path : "";
    const std::string latentOutPath = latent_out_path ? latent_out_path : "";
    if (audioPath.empty())     return err("Audio input path is required", error);
    if (latentOutPath.empty()) return err("Latent output path is required", error);

    // Parsed for validation only: the encode direction reads no request field
    // today (here the VAE is selected by path, not by registry name).
    AceRequest req;
    if (!request_parse_json(&req, request_json && *request_json ? request_json : "{}")) {
      return err("Invalid request JSON", error);
    }

    std::string msg;
    int src_len = 0;
    src_interleaved = read_audio_48k_interleaved(audioPath, src_len, msg);
    if (!src_interleaved) return err(msg, error);
    AudioBufGuard audio_guard{src_interleaved};

    const auto t_total = ace_clock::now();
    int vae_chunk = 0, vae_overlap = 0;
    resolve_vae_tiles(vae_tile_size, /*vae_use_gpu=*/false, &vae_chunk, &vae_overlap);
    ACE_LOGI("VAE-Encode: start (%.2fs audio, vae_tile=%d, vae_overlap=%d)",
             (float) src_len / (float) SAMPLE_RATE, vae_chunk, vae_overlap);

    store = store_create(EVICT_STRICT);
    if (!store) return err("Failed to create model store", error);

    ModelKey vae_key;
    vae_key.kind          = MODEL_VAE_ENC;
    vae_key.path          = vae_path ? vae_path : "";
    vae_key.adapter_scale = 1.0f;

    if (progress) progress(progress_ctx, "Loading VAE", 0, 1);
    const auto t_load = ace_clock::now();
    VAEEncoder *vae = store_require_vae_enc(store, vae_key);
    if (!vae) {
      store_free(store);
      store = nullptr;
      return err("Failed to load VAE encoder: " + vae_key.path, error);
    }
    ACE_LOGI("VAE-Encode: encoder loaded in %ld ms", ace_elapsed_ms(t_load));

    // 1 latent frame per 1920 audio samples, plus a tile of slack for the tiled
    // encoder's boundary rounding. Capped so an oversized input cannot overrun.
    int T_latent_max = src_len / SAMPLES_PER_FRAME + 64;
    if (T_latent_max > MAX_T_LATENT) T_latent_max = MAX_T_LATENT;

    // The encoder takes no cancel hook, so a stop is honored at the boundaries
    // around it: before the encode starts, and again before the file is written.
    if (stop_bridge(cancel_data)) {
      store_release(store, vae);
      store_free(store);
      store = nullptr;
      ACE_LOGI("VAE-Encode: stopped by request before encoding");
      return ACESTEP_CANCELLED;
    }

    std::vector<float> latent((size_t) T_latent_max * (size_t) LATENT_CHANNELS);
    int T_latent = -1;
    {
      ModelHandle vae_guard(store, vae);
      if (progress) progress(progress_ctx, "Encoding audio", 0, 1);
      const auto t_enc = ace_clock::now();
      T_latent = vae_enc_encode_tiled(vae, src_interleaved, src_len, latent.data(), T_latent_max,
                                      vae_chunk, vae_overlap);
      if (T_latent >= 0) {
        ACE_LOGI("VAE-Encode: %d audio samples (%.2fs) -> %d latent frames in %ld ms", src_len,
                 (float) src_len / (float) SAMPLE_RATE, T_latent, ace_elapsed_ms(t_enc));
      }
    }
    store_free(store);
    store = nullptr;

    if (T_latent < 0) {
      ACE_LOGE("VAE-Encode: vae_enc_encode_tiled failed");
      return err("VAE encode failed", error);
    }
    if (stop_bridge(cancel_data)) {
      ACE_LOGI("VAE-Encode: stopped by request after %ld ms", ace_elapsed_ms(t_total));
      return ACESTEP_CANCELLED;
    }

    if (progress) progress(progress_ctx, "Encoding audio", 1, 1);
    if (!write_latent_file(latentOutPath, latent, T_latent)) {
      return err("Failed to write latents: " + latentOutPath, error);
    }
    ACE_LOGI("VAE-Encode: wrote %d latent frames -> %s (%ld ms total)", T_latent,
             base_name(latentOutPath).c_str(), ace_elapsed_ms(t_total));

    const double duration = (double) T_latent * (double) SAMPLES_PER_FRAME / (double) SAMPLE_RATE;

    std::string body = "{";
    body += "\"latentPath\":\"" + json_escape(latentOutPath) + "\",";
    body += "\"latentFrames\":" + std::to_string(T_latent) + ",";
    body += "\"sampleRate\":" + std::to_string(SAMPLE_RATE) + ",";
    body += "\"durationSeconds\":" + std::to_string(duration);
    body += "}";
    return ok(body, result);
  } catch (const std::exception &e) {
    if (store) store_free(store);
    return err(e.what(), error);
  } catch (...) {
    if (store) store_free(store);
    return err("Unknown ACE-Step VAE encode error", error);
  }
}

// =============================================================================
// acestep_run_vae_decode  (the "latents in" half of the server's POST /vae)
// =============================================================================
extern "C" acestep_status acestep_run_vae_decode(
    const char *vae_path,
    const char *request_json,
    const char *latent_path,
    const char *output_path,
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
  try {
    const std::string latentPath = latent_path ? latent_path : "";
    const std::string outputPath = output_path ? output_path : "";
    if (latentPath.empty()) return err("Latent input path is required", error);
    if (outputPath.empty()) return err("Audio output path is required", error);

    AceRequest req;
    if (!request_parse_json(&req, request_json && *request_json ? request_json : "{}")) {
      return err("Invalid request JSON", error);
    }

    bool is_mp3 = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(req.output_format.empty() ? nullptr : req.output_format.c_str(), is_mp3,
                            wav_fmt)) {
      return err("Invalid output_format (use: mp3, wav16, wav24, wav32)", error);
    }

    std::vector<float> src_latents;
    int T_latent = 0;
    std::string msg;
    if (!read_latent_file(latentPath, src_latents, T_latent, msg)) {
      return err(msg, error);
    }

    const auto t_total = ace_clock::now();
    // GPU VAE is currently slower than CPU, so the decoder is forced onto the CPU
    // regardless of the caller's request -- the same override acestep_run_synth
    // applies. The GPU path (backend selection + 256-frame tile cap) stays wired
    // and is reachable by flipping this back to `use_gpu`.
    const bool vae_use_gpu = false;
    (void) use_gpu;
    int vae_chunk = 0, vae_overlap = 0;
    resolve_vae_tiles(vae_tile_size, vae_use_gpu, &vae_chunk, &vae_overlap);
    ACE_LOGI("VAE-Decode: start (%d latent frames, format=%s, vae_gpu=%s, vae_tile=%d, vae_overlap=%d)",
             T_latent, req.output_format.empty() ? "mp3" : req.output_format.c_str(),
             vae_use_gpu ? "on" : "off", vae_chunk, vae_overlap);

    store = store_create(EVICT_STRICT);
    if (!store) return err("Failed to create model store", error);

    ModelKey vae_key;
    vae_key.kind          = MODEL_VAE_DEC;
    vae_key.path          = vae_path ? vae_path : "";
    vae_key.adapter_scale = 1.0f;

    if (progress) progress(progress_ctx, "Loading VAE", 0, 1);
    const auto t_load = ace_clock::now();
    VAEGGML *vae = store_require_vae_dec(store, vae_key, vae_use_gpu);
    if (!vae) {
      store_free(store);
      store = nullptr;
      return err("Failed to load VAE decoder: " + vae_key.path, error);
    }
    ACE_LOGI("VAE-Decode: decoder loaded in %ld ms", ace_elapsed_ms(t_load));

    // The same headroom the server allows the tiled decoder: one tile of slack on
    // top of the 1920 samples each latent frame expands to.
    const int T_audio_max = (T_latent + 64) * SAMPLES_PER_FRAME;
    std::vector<float> audio_buf((size_t) T_audio_max * 2);
    int T_audio = -1;
    {
      ModelHandle vae_guard(store, vae);
      AceProgress prog{progress_bridge, &cb};
      const auto t_dec = ace_clock::now();
      T_audio = vae_ggml_decode_tiled(vae, src_latents.data(), T_latent, audio_buf.data(), T_audio_max,
                                      vae_chunk, vae_overlap,
                                      cancel_data ? stop_bridge : nullptr, cancel_data,
                                      progress ? &prog : nullptr);
      if (T_audio >= 0) {
        ACE_LOGI("VAE-Decode: %d latent frames -> %d audio samples (%.2fs) in %ld ms", T_latent, T_audio,
                 (float) T_audio / (float) SAMPLE_RATE, ace_elapsed_ms(t_dec));
      }
    }
    store_free(store);
    store = nullptr;

    if (T_audio < 0) {
      if (stop_bridge(cancel_data)) {
        ACE_LOGI("VAE-Decode: stopped by request after %ld ms", ace_elapsed_ms(t_total));
        return ACESTEP_CANCELLED;
      }
      ACE_LOGE("VAE-Decode: vae_ggml_decode_tiled failed");
      return err("VAE decode failed", error);
    }
    if (stop_bridge(cancel_data)) {
      ACE_LOGI("VAE-Decode: stopped by request after %ld ms", ace_elapsed_ms(t_total));
      return ACESTEP_CANCELLED;
    }

    // vae_ggml_decode_tiled writes planar [L..][R..], the layout audio_write
    // normalizes and encodes. The container follows the output path's extension,
    // as in acestep_run_synth; output_format only selects the WAV subformat.
    ACE_LOGI("VAE-Decode: writing %s -> %s", is_mp3 ? "mp3" : "wav", base_name(outputPath).c_str());
    const auto t_write = ace_clock::now();
    const bool wrote = audio_write(outputPath.c_str(), audio_buf.data(), T_audio, SAMPLE_RATE,
                                   req.mp3_bitrate, wav_fmt, req.peak_clip);
    if (!wrote) {
      ACE_LOGE("VAE-Decode: failed to write output: %s", outputPath.c_str());
      return err("Failed to write output: " + outputPath, error);
    }
    ACE_LOGI("VAE-Decode: wrote output in %ld ms", ace_elapsed_ms(t_write));

    const double duration = (double) T_audio / (double) SAMPLE_RATE;
    ACE_LOGI("VAE-Decode: complete (%.1fs of audio, %ld ms total)", duration, ace_elapsed_ms(t_total));

    std::string body = "{";
    body += "\"outputPath\":\"" + json_escape(outputPath) + "\",";
    body += "\"sampleRate\":" + std::to_string(SAMPLE_RATE) + ",";
    body += "\"numSamples\":" + std::to_string(T_audio) + ",";
    body += "\"durationSeconds\":" + std::to_string(duration);
    body += "}";
    return ok(body, result);
  } catch (const std::exception &e) {
    if (store) store_free(store);
    return err(e.what(), error);
  } catch (...) {
    if (store) store_free(store);
    return err("Unknown ACE-Step VAE decode error", error);
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
