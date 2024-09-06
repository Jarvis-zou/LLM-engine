#include "src/kernels/cublas_utils.cuh"


cublasWrapper::cublasWrapper(cublasHandle_t   cublas_handle_t,
                             cublasLtHandle_t cublaslt_handle_t) :
                             cublas_handle_t_(cublas_handle_t),
                             cublaslt_handle_t_(cublaslt_handle_t) {}

cublasWrapper::~cublasWrapper() {}

void cublasWrapper::setFP32GemmConfig() {
    Atype_       = CUDA_R_32F;
    Btype_       = CUDA_R_32F;
    Ctype_       = CUDA_R_32F;
    computeType_ = CUBLAS_COMPUTE_32F;
    algo_        = CUBLAS_GEMM_DEFAULT;
}

void cublasWrapper::setFP16GemmConfig() {
    Atype_       = CUDA_R_16F;
    Btype_       = CUDA_R_16F;
    Ctype_       = CUDA_R_16F;
    computeType_ = CUBLAS_COMPUTE_32F;  // inference does't need high precision
    algo_        = CUBLAS_GEMM_DEFAULT;
}

void cublasWrapper::Gemm(cublasOperation_t transa,
                         cublasOperation_t transb,
                         const int         m,
                         const int         n,
                         const int         k,
                         const float       *f_alpha,
                         const void        *A,
                         const int         lda,
                         const void        *B,
                         const int         ldb,
                         const float       *f_beta,
                         void              *C,
                         const int         ldc)
{   
    // fp16 check
    bool is_fp16_compute = (computeType_ == CUBLAS_COMPUTE_16F);
    half h_alpha = is_fp16_compute ? __float2half(*f_alpha) : half();
    half h_beta  = is_fp16_compute ? __float2half(*f_beta)  : half();

    const void* alpha = is_fp16_compute ? reinterpret_cast<const void*>(&h_alpha) : reinterpret_cast<const void*>(f_alpha);
    const void* beta  = is_fp16_compute ? reinterpret_cast<const void*>(&h_beta)  : reinterpret_cast<const void*>(f_beta);

    CHECK_CUBLAS(cublasGemmEx(cublas_handle_t_,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              Atype_,
                              lda,
                              B,
                              Btype_,
                              ldb,
                              beta,
                              C,
                              Ctype_,
                              ldc,
                              computeType_,
                              algo_));
}

void cublasWrapper::GemmStridedBatched(cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       const int         m,
                                       const int         n,
                                       const int         k,
                                       const float       *f_alpha,
                                       const void        *A,
                                       const int         lda,
                                       const int64_t     strideA,
                                       const void        *B,
                                       const int         ldb,
                                       const int64_t     strideB,
                                       const float       *f_beta,
                                       void              *C,
                                       const int         ldc,
                                       const int64_t     strideC,
                                       const int         batchCount)
{
    // fp16 check
    bool is_fp16_compute = (computeType_ == CUBLAS_COMPUTE_16F);
    half h_alpha = is_fp16_compute ? __float2half(*f_alpha) : half();
    half h_beta  = is_fp16_compute ? __float2half(*f_beta)  : half();

    const void* alpha = is_fp16_compute ? reinterpret_cast<const void*>(&h_alpha) : reinterpret_cast<const void*>(f_alpha);
    const void* beta  = is_fp16_compute ? reinterpret_cast<const void*>(&h_beta)  : reinterpret_cast<const void*>(f_beta);

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(cublas_handle_t_,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            Atype_,
                                            lda,
                                            strideA,
                                            B,
                                            Btype_,
                                            ldb,
                                            strideB,
                                            beta,
                                            C,
                                            Ctype_,
                                            ldc,
                                            strideC,
                                            batchCount,
                                            computeType_,
                                            algo_));
}