# Android OpenCL provider for the unmodified ggml submodule.
#
# The parent ACE-Step project creates the static Khronos ICD loader as the
# OpenCL target and supplies its headers before ggml calls find_package(OpenCL).

if(NOT TARGET OpenCL)
    set(OpenCL_FOUND FALSE)
    if(OpenCL_FIND_REQUIRED)
        message(FATAL_ERROR "ACE-Step's Android OpenCL provider requires the OpenCL target")
    endif()
    return()
endif()

if(NOT EXISTS "${ACESTEP_OPENCL_HEADERS_DIR}/CL/cl.h")
    set(OpenCL_FOUND FALSE)
    if(OpenCL_FIND_REQUIRED)
        message(FATAL_ERROR "ACE-Step OpenCL headers are missing from ${ACESTEP_OPENCL_HEADERS_DIR}")
    endif()
    return()
endif()

set(OpenCL_FOUND TRUE)
set(OpenCL_LIBRARY OpenCL)
set(OpenCL_LIBRARIES OpenCL)
set(OpenCL_INCLUDE_DIR "${ACESTEP_OPENCL_HEADERS_DIR}")
set(OpenCL_INCLUDE_DIRS "${ACESTEP_OPENCL_HEADERS_DIR}")
