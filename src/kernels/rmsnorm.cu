#include "src/kernels/rmsnorm.cuh"

template<typename T>
__device__ T warpReduceSum(T val) {
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val) {
    const int tid = threadIdx.x;                    // thread_id in block
    const int wid = tid / 32;                       // warp_id in block
    const int lane_id = tid % 32;                   // thread_id(lane_id) in warp
    const int num_warps = (blockDim.x + 31) / 32;
    static __shared__ T warpSum[32];                // assume blocksize <= 1024

    val = warpReduceSum<T>(val);
    if (lane_id == 0) {
        warpSum[wid] = val;
    }
    __syncthreads();

    T sum = tid < num_warps ? warpSum[tid] : (T)0;  // assign one thread for each warp to read result
    sum = warpReduceSum<T>(sum);

    return sum;
}

template<typename T>
__global__ void RMSNorm(T* decoder_out,
                        T* decoder_residual,
                        T* gamma,
                        float eps,
                        int hidden_size)
{
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;  // float4
    float thread_sum = 0.0f;

    // each block handles one row of token embedding vector
    Vec_t* embedding= reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_size);
    Vec_t* residual = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_size);

    for (int i = threadIdx.x; i < hidden_size / vec_size; i += blockDim.x) {
        Vec_t vec = embedding[i];  // load 4 float values or 2 half values
        if (decoder_residual != nullptr) { residual[i] = vec; }

        // square sum
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
    }

    // sum(all x^2) in this row(block)
    thread_sum = blockReduceSum<float>(thread_sum);

    // normalization
    __shared__ float inverse_rms_mean;
    if (threadIdx.x == 0) {
        inverse_rms_mean = rsqrtf((float)(thread_sum / hidden_size) + eps);
    }
    __syncthreads();

    Vec_t* scale = reinterpret_cast<Vec_t*>(gamma);
    for (int i = threadIdx.x; i < hidden_size / vec_size; i += blockDim.x) {
        Vec_t vec = embedding[i];  // load 4 float values or 2 half values

        // square sum
        embedding[i].x = vec.x * inverse_rms_mean * scale[i].x;
        embedding[i].y = vec.y * inverse_rms_mean * scale[i].y;
        embedding[i].z = vec.x * inverse_rms_mean * scale[i].z;
        embedding[i].w = vec.w * inverse_rms_mean * scale[i].w;
    }
}


template<>
__global__ void RMSNorm<half>(half* decoder_out,
                              half* decoder_residual,
                              half* gamma,
                              float eps,
                              int hidden_size)
{
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;  // half2
    float thread_sum = 0.0f;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
    }

    // each block handles one row of token embedding vector
    Vec_t* embedding= reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_size);
    Vec_t* residual = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_size);

    for (int i = threadIdx.x; i < hidden_size / vec_size; i += blockDim.x) {
        Vec_t vec = embedding[i];  // load 4 float values or 2 half values
        if (decoder_residual != nullptr) { residual[i] = vec; }

        // square sum
        thread_sum += __half2float(vec.x) * __half2float(vec.x);
        thread_sum += __half2float(vec.y) * __half2float(vec.y);
    }

    // sum(all x^2) in this row(block)
    thread_sum = blockReduceSum<float>(thread_sum);

    // normalization
    __shared__ float inverse_rms_mean;
    if (threadIdx.x == 0) {
        inverse_rms_mean = rsqrtf((float)(thread_sum / hidden_size) + eps);
    }
    __syncthreads();

    Vec_t* scale = reinterpret_cast<Vec_t*>(gamma);  // half type
    for (int i = threadIdx.x; i < hidden_size / vec_size; i += blockDim.x) {
        Vec_t vec = embedding[i];  // load 4 float values or 2 half values

        // square sum
        embedding[i].x = __float2half(__half2float(vec.x) * inverse_rms_mean) * scale[i].x;
        embedding[i].y = __float2half(__half2float(vec.y) * inverse_rms_mean) * scale[i].y;
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("embedding[0].x=%f, embedding[0].y=%f\n", __half2float(embedding[i].x), __half2float(embedding[i].y));
        }
    }
}

template<typename T>
void launchRMSNorm(TensorWrapper<T>* decoder_out,              // current decoder layer output, shape = [num_tokens, hidden_size]
                   TensorWrapper<T>* decoder_residual,         // former decoder layer output, shape = [num_token, hidden_size]
                   LayerNormWeight<T>& layer_norm_weights,     // using gamma for scaling weight
                   float eps,                                  // RMS hyper-param
                   bool is_last)                       // flag for last layer of decoder
{
    const int num_tokens = decoder_out->shape[0];
    const int hidden_size = decoder_out->shape[1];
    const int vec_size = Vec<T>::size;
    const int num_threads = hidden_size / vec_size;

    dim3 grid(num_tokens);  // each block handle one row of vector
    dim3 block;
    if (num_threads > 1024) {   // My device has 1024 block_size limit
        block = dim3(1024);
    }
    else {
        block = dim3(num_threads);
    }

    RMSNorm<T><<<grid, block>>>(decoder_out->data,
                             decoder_residual->data,
                             layer_norm_weights.gamma,
                             eps,
                             hidden_size);
}

template void launchRMSNorm(TensorWrapper<float>* decoder_out,
                            TensorWrapper<float>* decoder_residual,
                            LayerNormWeight<float>& layer_norm_weights,
                            float eps,
                            bool is_last);
                            
template void launchRMSNorm(TensorWrapper<half>* decoder_out,
                            TensorWrapper<half>* decoder_residual,
                            LayerNormWeight<half>& layer_norm_weights,
                            float eps,
                            bool is_last);