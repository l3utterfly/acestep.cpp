# Layla self-contained C-ABI binding for ACE-Step.
#
# Built only when ACESTEP_BUILD_LAYLA_BINDINGS is set (by Layla's ace-step module
# — modules/ace-step/android/CMakeLists.txt). Produces libacestep-layla.a, which
# the Layla side whole-archives into the self-contained libacestep_engine.so.
# Everything here is isolated under layla/ so upstream acestep.cpp merges never
# touch it.

add_library(acestep-layla STATIC acestep_capi.cpp)

target_include_directories(acestep-layla PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tools)      # synth-batch-runner.h (header-only)

target_compile_features(acestep-layla PRIVATE cxx_std_17)

# acestep-core supplies the engine symbols; link_ggml_backends adds the engine's
# include dirs (src/, ggml/include) and links ggml (ggml-opencl is resolved later
# at the self-contained .so link). link_ggml_backends is defined by the parent
# CMakeLists before this subdirectory is added.
target_link_libraries(acestep-layla PRIVATE acestep-core)
link_ggml_backends(acestep-layla)

set_target_properties(acestep-layla PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Compile hidden so internal helpers never become export candidates. The
# acestep_* API is force-exported by __attribute__((visibility("default"))) in
# acestep_capi.h (ACESTEP_CAPI_EXPORT) -- REQUIRED: a hidden-visibility symbol is
# dropped from the dynamic table even if the final .so's version script lists it
# as global, so the attribute (not the version script) is what actually exports
# the ABI. The version script then localizes everything else (ggml/cl*).
target_compile_options(acestep-layla PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
