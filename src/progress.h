#pragma once
// progress.h: lightweight progress-reporting hook for the ACE-Step pipelines
//
// A single, dependency-free callback the LM and synth pipelines invoke to report
// coarse and fine-grained progress. Mirrors the cancel() convention already
// threaded through the pipelines: an optional function pointer plus an opaque
// user_data, polled/called at meaningful boundaries.
//
// The callback receives three values, matching the frontend contract:
//   status  human-readable phase label ("Generating music", "Decoding audio", ...)
//   current 0-based/1-based position inside the current phase
//   total   number of units in the current phase (current == total when done)
//
// Coarse one-shot phases (text encode, source encode) report (label, 1, 1). The
// long phases animate: DiT reports (step+1, num_steps) every denoise step, the
// VAE reports (tile+1, num_tiles) every decode tile. Consumers that only want a
// phase label can ignore current/total; consumers driving a bar use both.

struct AceProgress {
    // Called with (status, current, total, user_data). NULL disables reporting.
    void (*cb)(const char * status, int current, int total, void * user_data);
    void * user_data;
};

// Safe helper: no-op when the reporter or its callback is NULL. Kept inline and
// header-only so every translation unit gets its own copy with no link deps.
static inline void ace_progress_report(const AceProgress * p, const char * status, int current, int total) {
    if (p && p->cb) {
        p->cb(status, current, total, p->user_data);
    }
}
