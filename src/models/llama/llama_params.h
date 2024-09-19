#pragma once

// Static params for RoPE config
struct LLaMAAttentionStaticParams {
    int   rotary_embedding_dim;
    float inv_freq;
    int   max_position_embeddings;
    bool  use_dynamic_ntk;
};

//
struct LLaMAAttentionDynamicParams {
    int batch_size;
    int num_tokens;
    int max_q_len;
    int max_k_len;
    int num_layers;
    bool is_ctx = false;  // is context mode
};
