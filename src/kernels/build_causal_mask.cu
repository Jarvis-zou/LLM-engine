#include "build_causal_mask.cuh"

template<typename T>
__global__ void buildCausalMask(T* mask,                     // shape = [batch_size, batch_max_q_len, batch_max_k_len]
                                const int* batch_q_len,
                                const int* batch_k_len,
                                const int batch_max_q_len,
                                const int batch_max_k_len)
{
    unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const int curr_q_len = batch_q_len[bid];  // each block handles one sample
    const int curr_k_len = batch_k_len[bid];
    const int mask_size = batch_max_q_len * batch_max_k_len;
    T* curr_mask = mask + bid * mask_size;

    while (tid < mask_size) {
        const unsigned int qid = tid / batch_max_k_len;  // mask_row_id
        const unsigned int kid = tid % batch_max_k_len;  // mask_col_id
//        bool is_not_masked = qid < curr_q_len && kid < curr_k_len && (curr_k_len - curr_q_len) <= kid <= qid + (curr_k_len - curr_q_len);  // mask history context
        bool is_not_masked = qid < curr_q_len && kid < curr_k_len && kid <= qid + (curr_k_len - curr_q_len);  // no mask history context
        curr_mask[tid] = static_cast<T>(is_not_masked);

        tid += blockDim.x;  // loop
    }
}

template<typename T>
void launchBuildCausalMask(TensorWrapper<T>* mask,
                           TensorWrapper<int>* batch_q_len,
                           TensorWrapper<int>* batch_k_len)
{
    const int batch_size = mask->shape[0];
    const int max_q_len = mask->shape[1];
    const int max_k_len = mask->shape[2];
    buildCausalMask<T><<<batch_size, 1024>>>(mask->data,
                                             batch_q_len->data,
                                             batch_k_len->data,
                                             max_q_len,
                                             max_k_len);

}

template void launchBuildCausalMask(TensorWrapper<float>* mask,
                                    TensorWrapper<int>* batch_q_len,
                                    TensorWrapper<int>* batch_k_len);

template void launchBuildCausalMask(TensorWrapper<half>* mask,
                                    TensorWrapper<int>* batch_q_len,
                                    TensorWrapper<int>* batch_k_len);