#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/kernels/input_embedding.cuh"  // Assuming the kernel and launcher are declared here

void testFP16() {
    // [Test5]: Embedding Kernel Test for FP16
    try {
        // Define dimensions
        const int max_context_token_num = 2;  // Example: 2 tokens
        const int hidden_size = 4;            // Example: 4 dimensions per embedding

        // Initialize input data (token IDs)
        int host_input_ids[max_context_token_num] = {1, 0};  // Example input token IDs

        // Initialize embedding table on the host with FP16 values
        __half host_embedding_table[2 * hidden_size] = {
            __float2half(0.1f), __float2half(0.2f), __float2half(0.3f), __float2half(0.4f),  // Embedding for token ID 0
            __float2half(0.5f), __float2half(0.6f), __float2half(0.7f), __float2half(0.8f)   // Embedding for token ID 1
        };

        // Expected output embeddings in FP16
        __half expected_output[max_context_token_num * hidden_size] = {
            __float2half(0.5f), __float2half(0.6f), __float2half(0.7f), __float2half(0.8f),  // Embedding for token ID 1
            __float2half(0.1f), __float2half(0.2f), __float2half(0.3f), __float2half(0.4f)   // Embedding for token ID 0
        };

        // Allocate device memory
        int* dev_input_ids;
        __half* dev_embedding_table;
        __half* dev_token_embeddings;

        cudaMalloc((void**)&dev_input_ids, max_context_token_num * sizeof(int));
        cudaMalloc((void**)&dev_embedding_table, 2 * hidden_size * sizeof(__half));
        cudaMalloc((void**)&dev_token_embeddings, max_context_token_num * hidden_size * sizeof(__half));

        // Copy data to device
        cudaMemcpy(dev_input_ids, host_input_ids, max_context_token_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_embedding_table, host_embedding_table, 2 * hidden_size * sizeof(__half), cudaMemcpyHostToDevice);

        // Prepare TensorWrapper for launchInputEmbedding function
        TensorWrapper<int> input_ids_wrapper = TensorWrapper<int>(Device::GPU, TensorType::INT32, std::vector<int>{max_context_token_num}, dev_input_ids);
        TensorWrapper<__half> output_wrapper = TensorWrapper<__half>(Device::GPU, TensorType::FP16, std::vector<int>{max_context_token_num, hidden_size}, dev_token_embeddings);
        
        // Initialize EmbeddingWeight for FP16
        EmbeddingWeight<__half> embedding_table_wrapper;
        embedding_table_wrapper.shape = {2, hidden_size};  // Shape of the embedding table: 2 tokens, each with hidden_size dimensions
        embedding_table_wrapper.type = WeightType::FP16_W;  // Assuming WeightType is an enum or similar that describes the type of weight
        embedding_table_wrapper.weights = dev_embedding_table;  // Set the device memory pointer for weights
        embedding_table_wrapper.bias = nullptr;  // No bias for embeddings, can be set to nullptr

        // Launch the kernel via the launcher function
        launchInputEmbedding<__half>(&input_ids_wrapper, &output_wrapper, &embedding_table_wrapper);

        // Copy output from device to host
        __half host_output[max_context_token_num * hidden_size];
        cudaMemcpy(host_output, output_wrapper.data, max_context_token_num * hidden_size * sizeof(__half), cudaMemcpyDeviceToHost);

        // Compare the output with the expected result
        bool test_passed = true;
        for (int i = 0; i < max_context_token_num * hidden_size; ++i) {
            float host_output_float = __half2float(host_output[i]);
            float expected_output_float = __half2float(expected_output[i]);
            printf("host_output=%f, expected=%f\n", host_output_float, expected_output_float);
            if (abs(host_output_float - expected_output_float) > 1e-3) {  // Use a larger tolerance for FP16
                test_passed = false;
                break;
            }
        }

        std::cout << (test_passed ? "[Test5] Passed: Embedding Kernel Test for FP16" : "[Test5] Failed: Embedding Kernel Test for FP16") << std::endl;

        // Free device memory
        cudaFree(dev_input_ids);
        cudaFree(dev_embedding_table);
        cudaFree(dev_token_embeddings);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return;
}

void testFP32() {
    // [Test4]: Embedding Kernel Test
    try {
        // Define dimensions
        const int max_context_token_num = 2;  // Example: 2 tokens
        const int hidden_size = 4;            // Example: 4 dimensions per embedding

        // Initialize input data (token IDs)
        int host_input_ids[max_context_token_num] = {1, 0};  // Example input token IDs

        // Initialize embedding table on the host
        float host_embedding_table[2 * hidden_size] = {
            0.1f, 0.2f, 0.3f, 0.4f,  // Embedding for token ID 0
            0.5f, 0.6f, 0.7f, 0.8f   // Embedding for token ID 1
        };

        // Expected output embeddings
        float expected_output[max_context_token_num * hidden_size] = {
            0.5f, 0.6f, 0.7f, 0.8f,  // Embedding for token ID 1
            0.1f, 0.2f, 0.3f, 0.4f   // Embedding for token ID 0
        };

        // Allocate device memory
        int* dev_input_ids;
        float* dev_embedding_table;
        float* dev_token_embeddings;

        cudaMalloc((void**)&dev_input_ids, max_context_token_num * sizeof(int));
        cudaMalloc((void**)&dev_embedding_table, 2 * hidden_size * sizeof(float));
        cudaMalloc((void**)&dev_token_embeddings, max_context_token_num * hidden_size * sizeof(float));

        // Copy data to device
        cudaMemcpy(dev_input_ids, host_input_ids, max_context_token_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_embedding_table, host_embedding_table, 2 * hidden_size * sizeof(float), cudaMemcpyHostToDevice);

        // Prepare TensorWrapper for launchInputEmbedding function
        TensorWrapper<int> input_ids_wrapper = TensorWrapper<int>(Device::GPU, TensorType::INT32, std::vector<int>{max_context_token_num}, dev_input_ids);
        TensorWrapper<float> output_wrapper = TensorWrapper<float>(Device::GPU, TensorType::FP32, std::vector<int>{max_context_token_num, hidden_size}, dev_token_embeddings);
        

        // Initialize EmbeddingWeight
        EmbeddingWeight<float> embedding_table_wrapper;
        embedding_table_wrapper.shape = {2, hidden_size};  // Shape of the embedding table: 2 tokens, each with hidden_size dimensions
        embedding_table_wrapper.type = WeightType::FP32_W;  // Assuming WeightType is an enum or similar that describes the type of weight
        embedding_table_wrapper.weights = dev_embedding_table;  // Set the device memory pointer for weights
        embedding_table_wrapper.bias = nullptr;  // No bias for embeddings, can be set to nullptr

        // Launch the kernel via the launcher function
        launchInputEmbedding<float>(&input_ids_wrapper, &output_wrapper, &embedding_table_wrapper);

        // Copy output from device to host
        float host_output[max_context_token_num * hidden_size];
        cudaMemcpy(host_output, output_wrapper.data, max_context_token_num * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Compare the output with the expected result
        bool test_passed = true;
        for (int i = 0; i < max_context_token_num * hidden_size; ++i) {
            printf("host_output=%f, expected=%f\n", host_output[i], expected_output[i]);
            if (abs(host_output[i] - expected_output[i]) > 1e-5) {
                test_passed = false;
                break;
            }
        }

        std::cout << (test_passed ? "[Test4] Passed: Embedding Kernel Test" : "[Test4] Failed: Embedding Kernel Test") << std::endl;

        // Free device memory
        cudaFree(dev_input_ids);
        cudaFree(dev_embedding_table);
        cudaFree(dev_token_embeddings);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return;
}

int main(int argc, char **argv) {
    testFP16();
    testFP32();
}