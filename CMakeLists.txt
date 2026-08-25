cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc. Modern distros link it implicitly but aarch64
# and older x86_64 toolchains need the explicit dependency.
find_package(Threads REQUIRED)

# Suppress MSVC fopen/sprintf deprecation warnings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
# SYCL excluded: _FORTIFY_SOURCE swaps memcpy for __memcpy_chk, unresolved in device code
if(NOT MSVC AND NOT GGML_SYCL)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries.
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
# CUDA graphs default on: standalone ggml ships them off. Overridable with
# -DGGML_CUDA_GRAPHS=OFF or at runtime with GGML_CUDA_DISABLE_GRAPHS=1.
if(NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# Android's NDK provides OpenCL declarations neither as headers nor as a link
# library. When the OpenCL backend is requested, build a private static Khronos
# ICD loader and make it look like the system OpenCL package expected by ggml.
# Keeping this provider here leaves the pinned ggml submodule unmodified.
if(ANDROID AND GGML_OPENCL)
    if(NOT ACESTEP_OPENCL_DEPS_ROOT)
        message(FATAL_ERROR
            "GGML_OPENCL on Android requires ACESTEP_OPENCL_DEPS_ROOT to contain "
            "OpenCL-Headers and OpenCL-ICD-Loader")
    endif()

    set(ACESTEP_OPENCL_HEADERS_DIR "${ACESTEP_OPENCL_DEPS_ROOT}/OpenCL-Headers")
    set(ACESTEP_OPENCL_LOADER_DIR  "${ACESTEP_OPENCL_DEPS_ROOT}/OpenCL-ICD-Loader")
    if(NOT EXISTS "${ACESTEP_OPENCL_HEADERS_DIR}/CL/cl.h")
        message(FATAL_ERROR "OpenCL headers not found under ${ACESTEP_OPENCL_HEADERS_DIR}")
    endif()
    if(NOT EXISTS "${ACESTEP_OPENCL_LOADER_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "OpenCL ICD loader not found under ${ACESTEP_OPENCL_LOADER_DIR}")
    endif()

    set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENCL_LAYERS OFF CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_HEADERS_DIR "${ACESTEP_OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
    add_subdirectory(
        "${ACESTEP_OPENCL_LOADER_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/OpenCL-ICD-Loader")

    # ggml calls find_package(OpenCL). Prefer our small provider module, which
    # exposes the in-tree OpenCL target and headers through its expected legacy
    # variables, before ggml appends its own CMake module directory.
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/android")

    # The embedded-kernel generator runs on the build host. Avoid resolving a
    # Python executable from the Android sysroot while cross-compiling.
    if(NOT Python3_EXECUTABLE)
        find_program(ACESTEP_HOST_PYTHON NAMES python3 python py NO_CMAKE_FIND_ROOT_PATH)
        if(NOT ACESTEP_HOST_PYTHON)
            message(FATAL_ERROR "A host Python interpreter is required to embed OpenCL kernels")
        endif()
        set(Python3_EXECUTABLE "${ACESTEP_HOST_PYTHON}" CACHE FILEPATH "Host Python interpreter" FORCE)
    endif()
endif()

add_subdirectory(ggml)

# The Android NDK has vulkan.h but not the Vulkan-Hpp C++ bindings used by
# ggml-vulkan. Apply the platform customization from this parent project after
# ggml creates the target, instead of patching ggml's CMakeLists.
if(ANDROID AND TARGET ggml-vulkan)
    if(NOT DEFINED ENV{VULKAN_SDK})
        message(FATAL_ERROR "GGML_VULKAN on Android requires VULKAN_SDK for vulkan.hpp")
    endif()
    target_include_directories(ggml-vulkan PRIVATE "$ENV{VULKAN_SDK}/Include")
    target_compile_definitions(ggml-vulkan PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
endif()

# cpp-httplib (HTTP server library, used by ace-server)
add_subdirectory(vendor/cpp-httplib)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan sycl)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()

    # SYCL links its runtime PRIVATE, consumers need -fsycl to resolve libsycl.so
    if(TARGET ggml-sycl AND GGML_SYCL)
        target_link_options(${target} PRIVATE -fsycl)
    endif()
    add_dependencies(${target} version)
endmacro()

# yyjson (MIT, fast JSON parser/writer)
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
    src/model-store.cpp
    src/pipeline-lm.cpp
    src/pipeline-synth.cpp
    src/pipeline-synth-ops.cpp
    src/pipeline-understand.cpp
)
target_link_libraries(acestep-core PUBLIC yyjson)
link_ggml_backends(acestep-core)

# ace-synth: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(ace-synth tools/ace-synth.cpp)
target_link_libraries(ace-synth PRIVATE acestep-core)
link_ggml_backends(ace-synth)

# ace-lm: LLM inference (CoT + audio codes)
add_executable(ace-lm tools/ace-lm.cpp)
target_link_libraries(ace-lm PRIVATE acestep-core)
link_ggml_backends(ace-lm)

# webui: convert tools/webui/public/index.html.gz to a C header for embedding.
# the .gz is committed to git so the C++ build works without npm.
# to update: cd tools/webui && npm install && npm run build, then rebuild ace-server.
set(WEBUI_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/index.html.gz")
set(WEBUI_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/index.html.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_INPUT}" "-DOUTPUT=${WEBUI_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_INPUT}"
    COMMENT "Embedding webui into index.html.gz.hpp"
)
set_source_files_properties(${WEBUI_OUTPUT} PROPERTIES GENERATED TRUE)

# ace-server: HTTP server (LM + synth endpoints + embedded webui)
add_executable(ace-server tools/ace-server.cpp ${WEBUI_OUTPUT})
target_include_directories(ace-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ace-server PRIVATE acestep-core httplib)
link_ggml_backends(ace-server)

add_executable(ace-understand tools/ace-understand.cpp)
target_link_libraries(ace-understand PRIVATE acestep-core)
link_ggml_backends(ace-understand)

# quantize: GGUF requantizer (BF16 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# neural-codec: Oobleck VAE neural audio codec (encode/decode WAV <-> latent)
add_executable(neural-codec tools/neural-codec.cpp)
link_ggml_backends(neural-codec)

# mp3-codec: MP3 encoder/decoder (standalone, no ggml needed)
# The mp3/ headers are header-only and usable by ace-synth too via #include "mp3/mp3enc.h"
add_executable(mp3-codec tools/mp3-codec.cpp)
target_include_directories(mp3-codec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mp3-codec version)
if(MSVC)
    target_compile_options(mp3-codec PRIVATE /W4 /wd4100 /wd4505)
else()
    target_compile_options(mp3-codec PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mp3-codec PRIVATE m)
endif()
target_link_libraries(mp3-codec PRIVATE Threads::Threads)

add_subdirectory(tests)
