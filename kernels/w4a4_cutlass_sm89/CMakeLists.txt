cmake_minimum_required(VERSION 3.18)
project(w4a4_sm89 LANGUAGES CXX CUDA)
if(NOT DEFINED CUTLASS_DIR)
  message(FATAL_ERROR "pass -DCUTLASS_DIR=/path/to/cutlass")
endif()
set(CMAKE_CUDA_STANDARD 17)
add_library(w4a4_sm89 STATIC w4a4_sm89.cu w4a4_fused.cu)
target_include_directories(w4a4_sm89 PUBLIC
  ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include
  ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(w4a4_sm89 PROPERTIES CUDA_ARCHITECTURES 89)
add_executable(w4a4_benchmark benchmark.cu)
target_link_libraries(w4a4_benchmark w4a4_sm89)
set_target_properties(w4a4_benchmark PROPERTIES CUDA_ARCHITECTURES 89)

add_executable(w4a4_benchmark_fused benchmark_fused.cu)
target_link_libraries(w4a4_benchmark_fused w4a4_sm89)
set_target_properties(w4a4_benchmark_fused PROPERTIES CUDA_ARCHITECTURES 89)
