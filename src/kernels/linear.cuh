#pragma once
#include <stdint.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/kernels/cublas_utils.cuh"
#include "src/utils/tensor.h"
#include "src/utils/macro.h"
#include "src/weights/base_weights.h"

template<typename T>
void launchLinearGemm(TensorWrapper<T>* input,
                      BaseWeight<T>*    weight,
                      TensorWrapper<T>* res,
                      cublasWrapper*    cublas_wrapper,
                      bool              transa = false,
                      bool              transb = false);

template<typename T>
void launchLinearGemmStridedBatched(TensorWrapper<T>* input1,
                                    TensorWrapper<T>* input2,
                                    TensorWrapper<T>* res,
                                    cublasWrapper*    cublas_wrapper,
                                    bool              transa = false,
                                    bool              transb = false);
