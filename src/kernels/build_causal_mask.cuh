#pragma once
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/utils/tensor.h"
#include "src/utils/macro.h"

template<typename T>
void launchBuildCausalMask(TensorWrapper<T>* mask,              // shape = [batch_size, batch_max_q_len, batch_max_k_len]
                           TensorWrapper<int>* batch_q_len,     // shape = [batch_size], user query length for each user query in batch
                           TensorWrapper<int>* batch_k_len);    // shape = [batch_size], context token length for each user context in batch