#include <algorithm>
#include <iostream>
#include <cstdlib>
#include "src/kernels/build_causal_mask.cuh"

void CPUbuildCasualMask(float* mask,
                        const int* q_lens,  // query len, shape = [batch size]
                        const int* k_lens,  // context len, shape = [batch size]
                        int max_q_len,
                        int max_k_len,
                        int batch_size) {
    for(int b = 0; b < batch_size; b++){
        int start = b * max_q_len * max_k_len;
        int q = q_lens[b];
        int k = k_lens[b];
        for(int i = 0; i < max_q_len; i++) {
            for(int j = 0; j < max_k_len; j++) {
                if(j <= i + (k - q) && i < q && j < k) {
                    mask[start + i * max_k_len + j] = 1.0f;
                } else {
                    mask[start + i * max_k_len + j] = 0.0f;
                }
            }
        }
    }

}

template<typename T>
bool checkResult(float* cpu_res, T* gpu_res, const int size) {
    float gpu_res_fp32;
    for (int i = 0; i < size; i++) {
        gpu_res_fp32 = std::is_same<T, half>::value ? __half2float(gpu_res[i]) : (float)gpu_res[i];
        if (fabsf(cpu_res[i] - gpu_res_fp32) > 1e-3) {
            printf("Wrong result at index[%d], cpu_res=%f, gpu_res=%f\n", i, cpu_res[i], gpu_res_fp32);
            return false;
        }
    }
    return true;
}


int main() {
    // init cpu data
    const int batch_size = 4;
    const int batch_max_q_len = 512;
    const int batch_max_k_len = 1024;
    const int mask_size = batch_size * batch_max_k_len * batch_max_q_len;

    auto* h_batch_q_len = (int*)malloc(batch_size * sizeof(int));
    auto* h_batch_k_len = (int*)malloc(batch_size * sizeof(int));
    auto* h_mask_fp32 = (float*)malloc(mask_size * sizeof(float));
    auto* h_mask_fp16 = (half*)malloc(mask_size * sizeof(half));

    for(int i = 0; i < batch_size; i++) {
        h_batch_q_len[i] = 256;
        h_batch_k_len[i] = 512;
    }

    // init GPU data
    int* d_batch_q_len;
    int* d_batch_k_len;
    float* d_mask_fp32;
    half* d_mask_fp16;

    cudaMalloc((void**)&d_batch_q_len, batch_size * sizeof(int));
    cudaMalloc((void**)&d_batch_k_len, batch_size * sizeof(int));
    cudaMalloc((void**)&d_mask_fp32, mask_size * sizeof(float));
    cudaMalloc((void**)&d_mask_fp16, mask_size * sizeof(half));

    CHECK(cudaMemcpy(d_batch_q_len, h_batch_q_len, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_batch_k_len, h_batch_k_len, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask_fp32, h_mask_fp32, mask_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask_fp16, h_mask_fp16, mask_size * sizeof(half), cudaMemcpyHostToDevice));

    // init test data

    auto* mask_fp32 = new TensorWrapper<float>(Device::GPU,
                                               TensorType::FP32,
                                               {batch_size, batch_max_q_len, batch_max_k_len},
                                               d_mask_fp32);

    auto* mask_fp16 = new TensorWrapper<half>(Device::GPU,
                                               TensorType::FP16,
                                               {batch_size, batch_max_q_len, batch_max_k_len},
                                               d_mask_fp16);

    auto* q_len = new TensorWrapper<int>(Device::GPU,
                                         TensorType::INT32,
                                         {batch_size},
                                         d_batch_q_len);

    auto* k_len = new TensorWrapper<int>(Device::GPU,
                                         TensorType::INT32,
                                         {batch_size},
                                         d_batch_k_len);

    // start kernels
    launchBuildCausalMask<float>(mask_fp32,
                                 q_len,
                                 k_len);

    launchBuildCausalMask<half>(mask_fp16,
                                q_len,
                                k_len);


    // Get result
    CHECK(cudaMemcpy(h_mask_fp32, d_mask_fp32, mask_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_mask_fp16, d_mask_fp16, mask_size * sizeof(half), cudaMemcpyDeviceToHost));

    // check result
    float* cpu_mask = (float*)malloc(mask_size * sizeof(float));
    CPUbuildCasualMask(cpu_mask,
                       h_batch_q_len,
                       h_batch_k_len,
                       batch_max_q_len,
                       batch_max_k_len,
                       batch_size);

    bool fp32_right = checkResult<float>(cpu_mask, h_mask_fp32, mask_size);
    bool fp16_right = checkResult<half>(cpu_mask, h_mask_fp16, mask_size);
    if (fp32_right && fp16_right) {
        std::cout << "Test Passed!" << std::endl;
    }

    // clean
    free(h_batch_q_len);
    free(h_batch_k_len);
    free(h_mask_fp32);
    free(h_mask_fp16);
    free(cpu_mask);

    cudaFree(d_batch_q_len);
    cudaFree(d_batch_k_len);
    cudaFree(d_mask_fp32);
    cudaFree(d_mask_fp16);

    delete(mask_fp32);
    delete(mask_fp16);
    delete(q_len);
    delete(k_len);
}