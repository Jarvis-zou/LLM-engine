#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/weights/llama/norm_weights.h"
#include "src/utils/vectorize.h"
template<typename T>
void launchRMSNorm(TensorWrapper<T>* decoder_out,              // current decoder layer output, shape = [num_tokens, hidden_size]
                   TensorWrapper<T>* decoder_residual,         // former decoder layer output, shape = [num_token, hidden_size]
                   LayerNormWeight<T>& layer_norm_weights,     // using gamma for scaling weight
                   float eps,                                  // RMS hyper-param
                   bool is_last);                      // flag for last layer of decoder