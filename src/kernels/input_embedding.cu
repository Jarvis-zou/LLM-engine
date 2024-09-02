#include "src/kernels/input_embedding.cuh"
#include <stdio.h>

template<typename T>
__global__ void load_embeddings(const int* input_ids,               // input token ids for accessing token embeddings
                                 T* token_embeddings,               // fetched token embeddings
                                 const T* embedding_table,          // pre-stored token embeddings
                                 const int max_context_token_num,   // max input token limit
                                 const int hidden_size)             // token embedding size
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int max_table_size = max_context_token_num * hidden_size;
    while (idx < max_table_size) {
        int id = input_ids[idx / hidden_size];                                          // which token embedding this thread is fetching
        token_embeddings[idx] = embedding_table[id * hidden_size + idx % hidden_size];  // each thread fetches one dimension in one token embedding
        idx += blockDim.x * gridDim.x;                                                  // loop processing until all tokem embeddings are fetched
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,
                          TensorWrapper<T>* token_embeddings,
                          EmbeddingWeight<T>* embedding_table)
{
    const int blockSize = 256;
    const int gridSize = 2048;
    const int max_context_token_num = token_embeddings->shape[0];  // num of tokens
    const int hidden_size = token_embeddings->shape[1];             // embedding size
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "Input and Output has different size when fetching embeddings!");
    load_embeddings<T><<<gridSize, blockSize>>>(input_ids->data, 
                                                token_embeddings->data, 
                                                embedding_table->weights, 
                                                max_context_token_num, 
                                                hidden_size);
}

template void launchInputEmbedding(TensorWrapper<int>* input_ids,
                                   TensorWrapper<float>* token_embeddings,
                                   EmbeddingWeight<float>* embedding_table);

template void launchInputEmbedding(TensorWrapper<int>* input_ids,
                                   TensorWrapper<half>* token_embeddings,
                                   EmbeddingWeight<half>* embedding_table);