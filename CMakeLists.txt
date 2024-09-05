cmake_minimum_required(VERSION 3.18)
project(oneLLM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(SOURCES
        user_entry.cpp
)

add_subdirectory(src)
add_subdirectory(tests)


add_executable(oneLLM ${SOURCES})

target_link_libraries(oneLLM ${CUDA_LIBRARIES})
target_include_directories(oneLLM PRIVATE ${CUDA_INCLUDE_DIRS})
