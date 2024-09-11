#include "src/kernels/linear.cuh"

template<typename T>
void launchLinearGemm(TensorWrapper<T>* input,
                      BaseWeight<T>*    weight,
                      TensorWrapper<T>* res,
                      cublasWrapper*    cublas_wrapper,
                      bool              transa,
                      bool              transb)
{   
    // In cublas, all matrices A, B and C are col-major, y = x * w -> y^T = w^T * x^T = weight^T * input^T
    const int Am = weight->shape[1];
    const int Ak = weight->shape[0];
    const int Bk = input->shape.size() == 3 ? input->shape[1] * input->shape[2] : input->shape[1];
    const int Bn = input->shape[0]; 
    const int Cm = res->shape.size() == 3 ? res->shape[1] * res->shape[2] : res->shape[1];
    const int Cn = res->shape[0];

    const int lda = Am;
    const int ldb = Bk;
    const int ldc = Cm;

    float alpha_val = 1.0f;
    float beta_val = 0.0f;
    const float* alpha = &alpha_val;
    const float* beta = &beta_val;

    if (!transa && !transb) {
        LLM_CHECK_WITH_INFO(Ak == Bk, 
            "Matrix multiplication dimension mismatch: A's column count (A.shape[1] = " + std::to_string(Ak) + 
            ") does not match B's row count (B.shape[0] = " + std::to_string(Bk) + "). Ensure that the number of columns in A is equal to the number of rows in B.");
    }

    cublasOperation_t transA = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transa ? CUBLAS_OP_T : CUBLAS_OP_N;

    const void* A = static_cast<const void*>(weight->weights);
    const void* B = static_cast<const void*>(input->data);
    void* C = static_cast<void*>(res->data);

    cublas_wrapper->Gemm(transA,
                         transB,
                         transb ? Ak : Am,
                         Cn,
                         Bk,
                         alpha,
                         A,
                         lda,
                         B,
                         ldb,
                         beta,
                         C,
                         ldc);
}

template<typename T>
void launchLinearGemmStridedBatched(TensorWrapper<T>* input1,
                                    TensorWrapper<T>* input2,
                                    TensorWrapper<T>* res,
                                    cublasWrapper*    cublas_wrapper,
                                    bool              transa,
                                    bool              transb)
{
    const int Am = input2->shape[2];
    const int Ak = input2->shape[3];
    const int Bk = input1->shape[2];
    const int Bn = input1->shape[3];
    const int Cm = res->shape[2];
    const int Cn = res->shape[3];

    const int lda = Ak;
    const int ldb = Bn;
    const int ldc = Cn;

    float alpha_val = 1.0f;
    float beta_val = 0.0f;
    const float* alpha = &alpha_val;
    const float* beta = &beta_val;

    const int64_t strideA = Am * Ak;
    const int64_t strideB = Bk * Bn;
    const int64_t strideC = Cm * Cn;

    const int batchCount = input1->shape[0] * input1->shape[1];

    if (!transa && !transb) {
        LLM_CHECK_WITH_INFO(Ak == Bk, 
            "Matrix multiplication dimension mismatch: A's column count (A.shape[1] = " + std::to_string(Ak) + 
            ") does not match B's row count (B.shape[0] = " + std::to_string(Bk) + "). Ensure that the number of columns in A is equal to the number of rows in B.");
    }

    cublasOperation_t transA = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transa ? CUBLAS_OP_T : CUBLAS_OP_N;

    const void* A = static_cast<const void*>(input2->data);
    const void* B = static_cast<const void*>(input1->data);
    void* C = static_cast<void*>(res->data);

    cublas_wrapper->GemmStridedBatched(transA,
                                       transB,
                                       Cn,
                                       Cm,
                                       Bn,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount);

}

template void launchLinearGemm(TensorWrapper<float>* input,
                               BaseWeight<float>*    weight,
                               TensorWrapper<float>* res,
                               cublasWrapper*        cublas_wrapper,
                               bool                  transa,
                               bool                  transb);

template void launchLinearGemm(TensorWrapper<half>* input,
                               BaseWeight<half>*    weight,
                               TensorWrapper<half>* res,
                               cublasWrapper*       cublas_wrapper,
                               bool                 transa,
                               bool                 transb);

template void launchLinearGemmStridedBatched(TensorWrapper<float>* input1,
                                             TensorWrapper<float>* input2,
                                             TensorWrapper<float>* res,
                                             cublasWrapper*        cublas_wrapper,
                                             bool                  transa,
                                             bool                  transb);

template void launchLinearGemmStridedBatched(TensorWrapper<half>* input1,
                                             TensorWrapper<half>* input2,
                                             TensorWrapper<half>* res,
                                             cublasWrapper*       cublas_wrapper,
                                             bool                 transa,
                                             bool                 transb);