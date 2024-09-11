#include "src/kernels/linear.cuh"

void linear_cpu(float* input, float* weight, float* output, int m, int k, int n) {
    for (int i = 0; i < m * n; i++) {
        output[i] = 0.0f;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                output[i * n + j] += input[i * k + l] * weight[l * n + j];
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
    const int seq_len = 13;
    const int hidden_size = 4096;
    const int vocab_size = 32;
    const int res_size = seq_len * hidden_size;  // res.shape = [seq_len, hidden_size]

    float* h_input;
    float* h_weight;
    float* h_res;

    h_input = (float*)malloc(seq_len* hidden_size * sizeof(float));
    h_weight = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    h_res = (float*)malloc(res_size * sizeof(float));


    for (int i = 0; i < res_size; i++) {
        h_input[i] = (float)(i % 3);   //  [1, 2, 1, 2, ..., 1, 2]
    }

    for (int i = 0; i < hidden_size * hidden_size; i++) {
        h_weight[i] = (float)(i % 3);  //  [1, 2, 1, 2, ..., 1, 2]
    }

    float* d_input;
    float* d_weight;
    float* d_res;

    cudaMalloc((void**)&d_input, seq_len* hidden_size * sizeof(float));
    cudaMalloc((void**)&d_weight, hidden_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_res, res_size * sizeof(float));

    CHECK(cudaMemcpy(d_input, h_input, seq_len* hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, h_weight, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    auto* input = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {seq_len, hidden_size}, d_input);
    auto* weight = new BaseWeight<float>;
    weight->shape = {hidden_size, hidden_size};
    weight->type = WeightType::FP32_W;
    weight->weights = d_weight;
    weight->bias = nullptr;
    auto* res = new TensorWrapper<float>(Device::GPU, TensorType::FP32, {seq_len, hidden_size}, d_res);

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasLtCreate(&cublaslt_handle);
    cublasCreate_v2(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    cublas_wrapper->setFP32GemmConfig();

    launchLinearGemm<float>(input,
                            weight,
                            res,
                            cublas_wrapper,
                            false,
                            false);

    CHECK(cudaMemcpy(h_res, d_res, res_size * sizeof(float), cudaMemcpyDeviceToHost));

    float* cpu_res = (float*)malloc(res_size * sizeof(float));
    linear_cpu(h_input, h_weight, cpu_res, seq_len, hidden_size, hidden_size);
    bool res_right = checkResult<float>(cpu_res, h_res, res_size);
    if (res_right) {
        std::cout << "Test Passed!" << std::endl;
    }

    free(h_input);
    free(h_weight);
    free(h_res);
    free(cpu_res);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_res);
}