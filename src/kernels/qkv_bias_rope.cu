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
                                                  const int  num_heads,
                                                  const int  num_head_kv,
                                                  const int  head_dim,
                                                  const int  rotary_embedding_dim,        // dim involved in rotation
                                                  float      inv_freq,                    // default = 10000 in llama
                                                  int        max_position_embeddings,     // max limit of input sequence length
                                                  bool       use_dynamic_ntk)
{
    const unsigned int token_id = blockIdx.x;
    const unsigned int head_id = blockIdx.y
}

template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T>* q_buf,
                                           TensorWrapper<T>* k_buf,
                                           TensorWrapper<T>* v_buf,
                                           TensorWrapper<T>* QKV,
                                           BaseWeight<T>* qkv,
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
    const int seq_len = q_buf->shape[2];

    const int num_heads_kv = (num_heads_QKV - num_heads_q) / 2;

    dim3 grid(num_tokens, num_heads_q);
    dim3 block(head_dim);


}
