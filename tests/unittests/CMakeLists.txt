find_package(CUDA REQUIRED)

add_executable(tensor_tests tensor_test.cpp)
target_include_directories(tensor_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(tensor_tests ${CUDA_LIBRARIES})
target_include_directories(tensor_tests PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(load_embeddings_test load_embeddings_test.cu)
target_include_directories(load_embeddings_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(load_embeddings_test ${CUDA_LIBRARIES} load_embeddings)
target_include_directories(load_embeddings_test PRIVATE ${CUDA_INCLUDE_DIRS})


