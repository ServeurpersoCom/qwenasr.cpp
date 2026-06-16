cmake_minimum_required(VERSION 3.14)
project(qwenasr-ggml LANGUAGES C CXX)

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
# is not merged into libc.
find_package(Threads REQUIRED)

# Force UTF-8 source and execution charsets so non-ASCII string literals
# (CJK language names in lang-map.h) survive the MSVC compile without a BOM.
# Restricted to C and C++ since nvcc treats a bare /utf-8 as an input filename.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# Executables and backend .so share the build root so ggml_backend_load_all()
# finds the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Qwen3-ASR GGUF tensor names can exceed the default GGML_MAX_NAME of 64.
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite with warn_unused_result. SYCL excluded since
# _FORTIFY_SOURCE swaps memcpy for __memcpy_chk, unresolved in device code.
if(NOT MSVC AND NOT GGML_SYCL)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: Turing to Blackwell for distributed binaries.
# Override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags.
add_subdirectory(ggml)

# cpp-httplib (HTTP server, no SSL, behind reverse proxy). Used by asr-server.
add_subdirectory(vendor/cpp-httplib)

# yyjson (fast JSON parser/writer). Used by asr-server.
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Shared compile options and ggml linkage.
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
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()
    if(TARGET ggml-sycl AND GGML_SYCL)
        target_link_options(${target} PRIVATE -fsycl)
    endif()
    add_dependencies(${target} version)
endmacro()

# Core library, always STATIC: the CLI tools include pipeline-asr.h / backend.h
# directly for staged debug paths and need every internal symbol resolved
# without going through the public ABI. QWENASR_STATIC propagates PUBLIC so the
# lib's own .cpp see QA_API as empty on Windows, and consumers inherit it.
add_library(qwenasr-core STATIC
    src/qwenasr.cpp
    src/pipeline-asr.cpp
)
target_compile_definitions(qwenasr-core PUBLIC QWENASR_STATIC)
link_ggml_backends(qwenasr-core)

# Public shared library for ABI consumers (Python ctypes, Rust bindgen, Go cgo).
# Opt-in: -DQWENASR_SHARED=ON. Exports only QA_API symbols.
option(QWENASR_SHARED "Build the shared qwenasr library for ABI consumers" OFF)
if(QWENASR_SHARED)
    add_library(qwenasr SHARED
        src/qwenasr.cpp
        src/pipeline-asr.cpp
    )
    target_compile_definitions(qwenasr PRIVATE QWENASR_BUILD)
    set_target_properties(qwenasr PROPERTIES
        C_VISIBILITY_PRESET       hidden
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    link_ggml_backends(qwenasr)
endif()

# quantize: GGUF requantizer (F32 -> BF16 / K-quants), shared policy with the
# sibling projects.
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# qwenasr-transcribe: full wav -> text pipeline, --dump writes the stage tensors
# for the cossim harness.
add_executable(qwenasr-transcribe tools/qwenasr-transcribe.cpp)
target_link_libraries(qwenasr-transcribe PRIVATE qwenasr-core)
link_ggml_backends(qwenasr-transcribe)

# asr-server: OpenAI-compatible HTTP server over the transcription pipeline.
add_executable(asr-server tools/asr-server.cpp)
target_link_libraries(asr-server PRIVATE qwenasr-core httplib yyjson)
link_ggml_backends(asr-server)

# test-abi-c: pure C99 smoke test locking in the public ABI. Built by default
# so a regression breaks the main build.
add_executable(test-abi-c tests/abi-c.c)
set_target_properties(test-abi-c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
if(MSVC)
    target_compile_options(test-abi-c PRIVATE /W4 /WX)
else()
    target_compile_options(test-abi-c PRIVATE -Wall -Werror -pedantic)
endif()
target_include_directories(test-abi-c PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_link_libraries(test-abi-c PRIVATE qwenasr-core)
link_ggml_backends(test-abi-c)
