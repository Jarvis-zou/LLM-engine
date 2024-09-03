#pragma once
#include "src/utils/tensor.h"
#include "src/utils/macro.h"

void launchCalPaddingOffset(TensorWrapper<int>* input_token_len,       // input token sequence length in each batch
                            TensorWrapper<int>* padding_offset,        // calculated padding offset for each token sequence
                            TensorWrapper<int>* prefix_seq_len_sum);   // prefix valid token sequence length