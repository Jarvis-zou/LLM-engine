add_library(load_embeddings STATIC input_embedding.cu)
set_property(TARGET load_embeddings PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET load_embeddings PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET load_embeddings PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(load_embeddings PRIVATE ${CMAKE_SOURCE_DIR})

add_library(cal_padding_offset STATIC cal_padding_offset.cu)
set_property(TARGET cal_padding_offset PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cal_padding_offset PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cal_padding_offset PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(cal_padding_offset PRIVATE ${CMAKE_SOURCE_DIR})

add_library(rmsnorm STATIC rmsnorm.cu)
set_property(TARGET rmsnorm PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rmsnorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rmsnorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(rmsnorm PRIVATE ${CMAKE_SOURCE_DIR})

add_library(build_causal_mask STATIC build_causal_mask.cu)
set_property(TARGET build_causal_mask PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET build_causal_mask PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET build_causal_mask PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(build_causal_mask PRIVATE ${CMAKE_SOURCE_DIR})

add_library(cublasWrapper STATIC cublas_utils.cu)
set_property(TARGET cublasWrapper PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(cublasWrapper PRIVATE ${CMAKE_SOURCE_DIR})

add_library(linear STATIC linear.cu)
set_property(TARGET linear PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET linear PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET linear PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(linear PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(linear PUBLIC cudart cublas cublaslt cublasWrapper)

add_library(qkv_bias_rope STATIC qkv_bias_rope.cu)
set_property(TARGET qkv_bias_rope PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET qkv_bias_rope PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET qkv_bias_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_include_directories(qkv_bias_rope PRIVATE ${CMAKE_SOURCE_DIR})


