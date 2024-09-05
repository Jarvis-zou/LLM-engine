#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include "src/kernels/rmsnorm.cuh"
#include "src/utils/macro.h"
#include "src/utils/tensor.h"
// `rmxnorm_test 0` to test fp32 GPU kernel
// `rmxnorm_test 1` to test fp16 GPU kernel

void CPUfusedResandRMSNorm(float* h_decoder_out,
                           const float* h_gamma,
                           const float eps,
                           const int hidden_size,
                           const int num_tokens
                           )
{
    for (int row = 0; row < num_tokens; row++) {
        float inverse_rms_mean = 0.0f;
        float row_sum = 0.0f;

        // x ^ 2
        for (int i = 0; i < hidden_size; i++) {
            float token_embedding = h_decoder_out[row * hidden_size + i];  // offset
            row_sum += token_embedding * token_embedding;
        }

        // Sum(x ^ 2)
        inverse_rms_mean = rsqrtf((row_sum / hidden_size) + eps);

        for (int i = 0; i < hidden_size; i++) {
            h_decoder_out[row * hidden_size + i] = h_decoder_out[row * hidden_size + i] * inverse_rms_mean * h_gamma[i];
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
    const int num_tokens = 64;
    const int hidden_size = 4096;  // llama-7b embedding_dimensions
    const int data_size = num_tokens * hidden_size;
    float eps = 1e-6;
    bool test_fp32 = false;

    if (test_fp32) {
        // Allocate CPU mem
        auto* h_decoder_out = (float*)malloc(data_size * sizeof(float));
        auto* h_gamma = (float*)malloc(data_size * sizeof(float));
        
        // Init data
        for (int i = 0; i < data_size; i++) {
            h_decoder_out[i] = (float)(i % 2 + 1);
        }
        for (int i = 0; i < hidden_size; i++) {
            h_gamma[i] = (float)(i % 2 + 1);
        }

        // Allocate GPU mem
        float* d_decoder_out;
        float* d_decoder_residual;
        float* d_gamma;
        
        cudaMalloc((void**)&d_decoder_out, data_size * sizeof(float));
        cudaMalloc((void**)&d_decoder_residual, data_size * sizeof(float));
        cudaMalloc((void**)&d_gamma, hidden_size * sizeof(float));

        // Move data to GPU
        CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, data_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

        // Init GPU input
        auto* decoder_out = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {num_tokens, hidden_size}, d_decoder_out);
        auto* decoder_residual = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {num_tokens, hidden_size}, d_decoder_residual);
        LayerNormWeight<float>layer_weight{};
        layer_weight.gamma = d_gamma;

        // launch kernel
        launchRMSNorm<float>(decoder_out, decoder_residual, layer_weight, eps, false);

        // Move result to CPU
        CHECK(cudaMemcpy(h_decoder_out, d_decoder_out, data_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Check result
        auto* cpu_decoder_out = (float*)malloc(data_size * sizeof(float));
        auto* cpu_gamma = (float*)malloc(hidden_size * sizeof(float));
        for (int i = 0; i < data_size; i++) {
            cpu_decoder_out[i] = (float)(i % 2 + 1);
        }
        for (int i = 0; i < hidden_size; i++) {
            cpu_gamma[i] = (float)(i % 2 + 1);
        }
        CPUfusedResandRMSNorm(cpu_decoder_out,
                              cpu_gamma,
                              eps,
                              hidden_size,
                              num_tokens);
        
        bool res_right = checkResult<float>(cpu_decoder_out, h_decoder_out, data_size);
        if (res_right) {
            std::cout << "Test Passed!" << std::endl;
        }

        // clean
        free(h_decoder_out);
        free(h_gamma);
        free(cpu_decoder_out);
        free(cpu_gamma);

        cudaFree(d_decoder_out);
        cudaFree(d_decoder_residual);
        cudaFree(d_gamma);

        delete(decoder_out);
        delete(decoder_residual);
    }
    else {
        // Allocate CPU mem
        auto* h_decoder_out = (half*)malloc(data_size * sizeof(half));
        auto* h_gamma = (half*)malloc(data_size * sizeof(half));

        // Init data
        for (int i = 0; i < data_size; i++) {
            h_decoder_out[i] = (half)(i % 2 + 1);
        }
        for (int i = 0; i < hidden_size; i++) {
            h_gamma[i] = (half)(i % 2 + 1);
        }

        // Allocate GPU mem
        half* d_decoder_out;
        half* d_decoder_residual;
        half* d_gamma;

        cudaMalloc((void**)&d_decoder_out, data_size * sizeof(half));
        cudaMalloc((void**)&d_decoder_residual, data_size * sizeof(half));
        cudaMalloc((void**)&d_gamma, hidden_size * sizeof(half));

        // Move data to GPU
        CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, data_size * sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(half), cudaMemcpyHostToDevice));

        // Init GPU input
        auto* decoder_out = new TensorWrapper<half>(Device::GPU, TensorType::FP16, {num_tokens, hidden_size}, d_decoder_out);
        auto* decoder_residual = new TensorWrapper<half>(Device::GPU, TensorType::FP16, {num_tokens, hidden_size}, d_decoder_residual);
        LayerNormWeight<half>layer_weight{};
        layer_weight.gamma = d_gamma;

        // launch kernel
        launchRMSNorm<half>(decoder_out, decoder_residual, layer_weight, eps, false);

        // Move result to CPU
        CHECK(cudaMemcpy(h_decoder_out, d_decoder_out, data_size * sizeof(half), cudaMemcpyDeviceToHost));

        // Check result
        auto* cpu_decoder_out = (float*)malloc(data_size * sizeof(float));
        auto* cpu_gamma = (float*)malloc(hidden_size * sizeof(float));
        for (int i = 0; i < data_size; i++) {
            cpu_decoder_out[i] = (float)(i % 2 + 1);
        }
        for (int i = 0; i < hidden_size; i++) {
            cpu_gamma[i] = (float)(i % 2 + 1);
        }
        CPUfusedResandRMSNorm(cpu_decoder_out,
                              cpu_gamma,
                              eps,
                              hidden_size,
                              num_tokens);

        bool res_right = checkResult<half>(cpu_decoder_out, h_decoder_out, data_size);
        if (res_right) {
            std::cout << "Test Passed!" << std::endl;
        }

        // clean
        free(h_decoder_out);
        free(h_gamma);
        free(cpu_decoder_out);
        free(cpu_gamma);

        cudaFree(d_decoder_out);
        cudaFree(d_decoder_residual);
        cudaFree(d_gamma);

        delete(decoder_out);
        delete(decoder_residual);
    }
}

