cmake_minimum_required(VERSION 3.5)

# Set the project name
project (dnn_benchmark)


set(CMAKE_CXX_STANDARD 17)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)

FetchContent_Declare(
  dlpack_dep
  GIT_REPOSITORY https://github.com/dmlc/dlpack.git
  GIT_TAG        v0.7
)

FetchContent_Declare(
  dmlc_core_dep
  GIT_REPOSITORY https://github.com/dmlc/dmlc-core.git
  GIT_TAG 09511cf9fe5ff103900a5eafb50870dc84cc17c8
)

FetchContent_MakeAvailable(
  Catch2
  dlpack_dep
  dmlc_core_dep
)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
list(APPEND CMAKE_PREFIX_PATH "/home/jack/code/tvm/3p/libtorch")

find_package(Torch)
find_package(tvm)
find_package(fmt)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

#############################################
# Unit tests

# enable CTest testing
enable_testing()

# Add a testing executable
add_executable(benchmark benchmark.cc)


target_link_libraries(benchmark
   PRIVATE
   Catch2::Catch2WithMain
   dlpack 
   dmlc
   tvm_runtime
   torch
   fmt::fmt
   onnxruntime
)

add_test(test_all benchmark)