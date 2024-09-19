#include "src/kernels/qkv_bias_rope.cuh"
#include "src/utils/tensor.h"
#include "src/weights/llama/attention_weights.h"

int main() {
    const int batch_size = 1;
    const int max_seq_len = 32;                       // max length of sequence in batch
    const int num_tokens = batch_size * max_seq_len;  // total token data in the whole batch data
    const int num_heads_q = 32;                       // number of attention heads for query
    const int num_heads_kv = 32;                      // number of attention heads for key & value
    const int head_dim = 128;                         // head_dim = hidden_size / num_heads = 4096 / 32 = 128
    const int rotary_embedding_dim = 128;             // total number of embedding dims involved in rope operation
    const int inv_freq = 10000;
    const int max_position_embedding = 2048;          // max limit of input sequence length

    int* h_padding_offset;
    int* h_history_len;
    int* h_input_seq_len;
    float* h_q;
    float* h_k;
    float* h_v;
    float* h_QKV;
    float* h_qkv_bias;

    h_padding_offset = (int*)malloc(sizeof(int) * batch_size * max_seq_len);
    h_history_len = (int*)malloc(sizeof(int) * batch_size);
    h_input_seq_len = (int*)malloc(sizeof(int) * batch_size);
    h_q = (float*)malloc(sizeof(float) * batch_size * max_seq_len * num_heads_q * head_dim);   // output query matrix
    h_k = (float*)malloc(sizeof(float) * batch_size * max_seq_len * num_heads_kv * head_dim);  // output key matrix
    h_v = (float*)malloc(sizeof(float) * batch_size * max_seq_len * num_heads_kv * head_dim);  // output value matrix
    h_QKV = (float*)malloc(sizeof(float) * num_tokens * (num_heads_q + 2 * num_heads_kv) * head_dim);  // input
    h_qkv_bias = (float*)malloc(sizeof(float) * (num_heads_q + 2 * num_heads_kv) * head_dim);
    for (int i = 0; i < num_tokens * (num_heads_q + 2 * num_heads_kv) * head_dim; i++) {
        h_QKV[i] = 32.0f;
    }
    for (int i = 0; i < (num_heads_q + 2 * num_heads_kv) * head_dim; i++) {
        h_qkv_bias[i] = 2.0f;
    }
    for (int i = 0; i < batch_size * max_seq_len; i++) {
        h_padding_offset[i] = 0;
    }
    for (int i = 0; i < batch_size; i++) {
        h_history_len[i] = 0;
        h_input_seq_len[i] = 32;
    }

    int* d_padding_offset;
    int* d_history_len;
    int* d_input_seq_len;
    float* d_q;
    float* d_k;
    float* d_v;
    float* d_QKV;
    float* d_qkv_bias;

    cudaMalloc((void**)&d_padding_offset, sizeof(int) * batch_size * max_seq_len);
    cudaMalloc((void**)&d_history_len, sizeof(int) * batch_size);
    cudaMalloc((void**)&d_input_seq_len, sizeof(int) * batch_size);
    cudaMalloc((void**)&d_q, sizeof(float) * batch_size * max_seq_len * num_heads_q * head_dim);
    cudaMalloc((void**)&d_k, sizeof(float) * batch_size * max_seq_len * num_heads_kv * head_dim);
    cudaMalloc((void**)&d_v, sizeof(float) * batch_size * max_seq_len * num_heads_kv * head_dim);
    cudaMalloc((void**)&d_QKV, sizeof(float) * num_tokens * (num_heads_q + 2 * num_heads_kv) * head_dim);
    cudaMalloc((void**)&d_qkv_bias, sizeof(float) * (num_heads_q + 2 * num_heads_kv) * head_dim);

    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * batch_size * max_seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_history_len, h_history_len, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_seq_len, h_input_seq_len, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_QKV, h_QKV, sizeof(float) * num_tokens * (num_heads_q + 2 * num_heads_kv) * head_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * (num_heads_q + 2 * num_heads_kv) * head_dim, cudaMemcpyHostToDevice);

    auto* q_buf = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {batch_size, num_heads_q, max_seq_len, head_dim}, d_q);
    auto* k_buf = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {batch_size, num_heads_kv, max_seq_len, head_dim}, d_k);
    auto* v_buf = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {batch_size, num_heads_kv, max_seq_len, head_dim}, d_v);

    LLaMAAttentionWeights<float> att_weights;
    att_weights.qkv.bias = d_qkv_bias;
    auto* padding_offset_buf = new TensorWrapper<int>(Device::GPU, TensorType::INT32, {batch_size, max_seq_len}, d_padding_offset);
    auto* history_len_buf = new TensorWrapper<int>(Device::GPU, TensorType::INT32, {batch_size}, d_history_len);
    auto* input_seq_len_buf = new TensorWrapper<int>(Device::GPU, TensorType::INT32, {batch_size}, d_input_seq_len);

    LLaMAAttentionStaticParams att_params;
    att_params.rotary_embedding_dim = rotary_embedding_dim;
    att_params.inv_freq = inv_freq;
    att_params.max_position_embeddings = max_position_embedding;
    att_params.use_dynamic_ntk = false;

}

































