cmake_minimum_required(VERSION 3.18)
project(PALLAS_VESTA LANGUAGES CXX CUDA)

# Use C++17 and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures if needed
set(CMAKE_CUDA_ARCHITECTURES 86)

# add_library(memory_obj OBJECT include/memory.cu)
# set_target_properties(memory_obj PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )
# target_compile_options(memory_obj PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -rdc=true>
# )
# Add main source
add_executable(main_exec src/main.cu)#  $<TARGET_OBJECTS:memory_obj>)

# Enable CUDA separable compilation and relocatable device code
set_target_properties(main_exec PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 86
)

# Set compile features
target_compile_features(main_exec PUBLIC cxx_std_17)

# Add NVCC-specific compile options
target_compile_options(main_exec PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -rdc=true>
)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(cuda_tests tests/test_runner.cu)

target_link_libraries(cuda_tests PRIVATE gtest gtest_main)
target_compile_features(cuda_tests PUBLIC cxx_std_17)
set_target_properties(cuda_tests PROPERTIES 
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_ARCHITECTURES 86
                    )
target_compile_options(cuda_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -rdc=true>
)

                    
# # Add include directories if needed
# target_include_directories(cuda_tests PRIVATE src)
# target_include_directories(cuda_tests PRIVATE include)

# Test discovery
include(GoogleTest)
gtest_discover_tests(cuda_tests)
configure_file(tests/constant_sage.py constant_sage.py COPYONLY)
configure_file(tests/point_sage.py point_sage.py COPYONLY)