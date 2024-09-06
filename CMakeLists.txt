cmake_minimum_required(VERSION 3.18)
project(oneLLM LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Set compiler flags for debug mode
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

# Specify the source files
set(SOURCES
        user_entry.cpp
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Add the executable target
add_executable(oneLLM ${SOURCES})

# Link the CUDA libraries (including cuBLAS)
target_link_libraries(oneLLM ${CUDA_LIBRARIES} -lcublas)

# Include directories for CUDA headers
target_include_directories(oneLLM PRIVATE ${CUDA_INCLUDE_DIRS})