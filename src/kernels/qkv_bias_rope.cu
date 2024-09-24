#include "qkv_bias_rope.cuh"


template<typename T>
__global__ void add_fused_QKV_Bias_Transpose_RoPE(T*         q_buf,
                                                  T*         k_buf,
                                                  T*         v_buf,
                                                  T*         QKV,
                                                  const int* padding_offset,
                                                  const int* history_length,
                                                  const int* input_seq_len,               // length of each sequence in this batch
                                                  const int  batch_size,
                                                  const int  max_seq_len,                 // max sequence length in this batch
                                                  const int  num_tokens,                  // total tokens in this batch
                                                  const int  num_heads_q,
                                                  const int  num_heads_kv,
                                                  const int  head_dim,
                                                  const int  rotary_embedding_dim,        // dim involved in rotation
                                                  float      inv_freq,                    // default = 10000 in llama
                                                  int        max_position_embeddings,     // max limit of input sequence length
                                                  bool       use_dynamic_ntk)
{
    const unsigned int token_id = blockIdx.x;
    const unsigned int head_id = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int token_padding_offset = padding_offset[token_id];
    const unsigned int dst_token_id = token_id + token_padding_offset;
    const unsigned int batch_id = dst_token_id / max_seq_len;
    const unsigned int local_token_id = dst_token_id % max_seq_len;     // relative offset in each sequence(batch)

    const unsigned int num_heads_QKV = num_heads_q + 2 * num_heads_kv;

    // qkv token offset in QKV fused matrix
    const unsigned int q_id = token_id * num_heads_QKV * head_dim + head_id * head_dim + tid;
    const unsigned int k_id = token_id * num_heads_QKV * head_dim + head_id * head_dim + num_heads_q * head_dim + tid;
    const unsigned int v_id = token_id * num_heads_QKV * head_dim + head_id * head_dim + num_heads_q * head_dim + num_heads_kv * head_dim + tid;
}

template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T>* q_buf,
                                           TensorWrapper<T>* k_buf,
                                           TensorWrapper<T>* v_buf,
                                           TensorWrapper<T>* QKV,
                                           BaseWeight<T>& qkv,
                                           TensorWrapper<int>* padding_offset,
                                           TensorWrapper<int>* history_length,
                                           TensorWrapper<int>* input_length,
                                           LLaMAAttentionStaticParams& params)
{
    const int num_tokens = QKV->shape[0];
    const int num_heads_QKV = QKV->shape[1];
    const int head_dim = QKV->shape[2];

    const int batch_size = q_buf->shape[0];
    const int num_heads_q = q_buf->shape[1];
    const int max_seq_len = q_buf->shape[2];

    const int num_heads_kv = (num_heads_QKV - num_heads_q) / 2;

    std::cout << num_heads_q << std::endl;
    std::cout << num_heads_kv << std::endl;


//    add_fused_QKV_Bias_Transpose_RoPE<T><<<grid, block>>>();
}

template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<float>*       q_buf,
                                                    TensorWrapper<float>*       k_buf,
                                                    TensorWrapper<float>*       v_buf,
                                                    TensorWrapper<float>*       QKV,
                                                    BaseWeight<float>&          qkv,
                                                    TensorWrapper<int>*         padding_offset,
                                                    TensorWrapper<int>*         history_length,
                                                    TensorWrapper<int>*         input_length,
                                                    LLaMAAttentionStaticParams& params);
template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<half>*        q_buf,
                                                    TensorWrapper<half>*        k_buf,
                                                    TensorWrapper<half>*        v_buf,
                                                    TensorWrapper<half>*        QKV,
                                                    BaseWeight<half>&           qkv,
                                                    TensorWrapper<int>*         padding_offset,
                                                    TensorWrapper<int>*         history_length,
                                                    TensorWrapper<int>*         input_length,
                                                    LLaMAAttentionStaticParams& params);
