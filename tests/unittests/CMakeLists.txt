add_executable(tensor_tests tensor_test.cpp)
target_include_directories(tensor_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(tensor_tests ${CUDA_LIBRARIES})
target_include_directories(tensor_tests PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(load_embeddings_test load_embeddings_test.cu)
target_include_directories(load_embeddings_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(load_embeddings_test ${CUDA_LIBRARIES} load_embeddings)
target_include_directories(load_embeddings_test PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(cal_padding_offset_test cal_padding_offset_test.cu)
target_include_directories(cal_padding_offset_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cal_padding_offset_test ${CUDA_LIBRARIES} cal_padding_offset)
target_include_directories(cal_padding_offset_test PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(rmsnorm_test rmsnorm_test.cu)
target_include_directories(rmsnorm_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(rmsnorm_test ${CUDA_LIBRARIES} rmsnorm)
target_include_directories(rmsnorm_test PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(build_causal_mask_test build_causal_mask_test.cu)
target_include_directories(build_causal_mask_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(build_causal_mask_test ${CUDA_LIBRARIES} build_causal_mask)
target_include_directories(build_causal_mask_test PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(linear_test linear_test.cu)
target_include_directories(linear_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(linear_test ${CUDA_LIBRARIES} linear)
target_include_directories(linear_test PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(qkv_bias_rope_test qkv_bias_rope_test.cu)
target_include_directories(qkv_bias_rope_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(qkv_bias_rope_test ${CUDA_LIBRARIES} qkv_bias_rope)
target_include_directories(qkv_bias_rope_test PRIVATE ${CUDA_INCLUDE_DIRS})

