cmake_minimum_required(VERSION 3.18)
project(oneLLM LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(SOURCES
        user_entry.cpp
)

add_subdirectory(src)
add_subdirectory(tests)


add_executable(oneLLM ${SOURCES})

target_link_libraries(oneLLM ${CUDA_LIBRARIES})
target_include_directories(oneLLM PRIVATE ${CUDA_INCLUDE_DIRS})
