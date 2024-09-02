#include "src/utils/tensor.h"
#include "src/weights/llama/embedding_weights.h"

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,
                          TensorWrapper<T>* token_embeddings,
                          EmbeddingWeight<T>* embedding_table);