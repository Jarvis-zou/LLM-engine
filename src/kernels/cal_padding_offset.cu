#include "src/kernels/cal_padding_offset.cuh"

__global__ void cal_padding_offset(const int    batch_size,
                                   const int    max_seq_len,
                                   const int*   input_token_len,
                                   int*         padding_offset,
                                   int*         prefix_seq_len_sum) 
{   
    int global_idx = 0;
    int offset_prefix_sum = 0;
    int token_len_prefix_sum = 0;

    for (int i = 0; i < batch_size; i++) {
        int currSeqValidTokenLen = input_token_len[i];
        prefix_seq_len_sum[i] = token_len_prefix_sum;
        
        for (int j = 0; j < currSeqValidTokenLen; j++) {
            padding_offset[global_idx] = offset_prefix_sum;
            global_idx++;
        }
        offset_prefix_sum += max_seq_len - currSeqValidTokenLen;
        token_len_prefix_sum += currSeqValidTokenLen;
    }
    prefix_seq_len_sum[batch_size] = token_len_prefix_sum;
}

void launchCalPaddingOffset(TensorWrapper<int>* input_token_len,      // input token sequence length in each batch
                            TensorWrapper<int>* padding_offset,       // calculated padding offset for each token sequence
                            TensorWrapper<int>* prefix_seq_len_sum)   // prefix valid token sequence length
{
    const int batch_size = padding_offset->shape[0];
    const int max_seq_len = padding_offset->shape[1];
    LLM_CHECK_WITH_INFO(batch_size == input_token_len->shape[0],
                        "Incompatible shapes: input_token_len.shape[0] (" +
                        std::to_string(input_token_len->shape[0]) +
                        ") must be equal to batch_size (" +
                        std::to_string(batch_size) + ")!");
    LLM_CHECK_WITH_INFO(batch_size + 1 == prefix_seq_len_sum->shape[0],
                        "Incompatible shapes: prefix_seq_len_sum.shape[0] (" +
                        std::to_string(prefix_seq_len_sum->shape[0]) +
                        ") must be equal to batch_size + 1 (" +
                        std::to_string(batch_size + 1) + ")!");
    cal_padding_offset<<<1, 1>>>(batch_size,
                                                  max_seq_len,
                                                  input_token_len->data,
                                                  padding_offset->data,
                                                  prefix_seq_len_sum->data);
}