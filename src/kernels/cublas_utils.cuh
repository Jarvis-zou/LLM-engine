#pragma once
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdint.h>
#include "src/utils/macro.h"

class cublasWrapper {
private:
    cublasHandle_t      cublas_handle_t_;
    cublasLtHandle_t    cublaslt_handle_t_;

    cudaDataType_t      Atype_;
    cudaDataType_t      Btype_;
    cudaDataType_t      Ctype_;

    cublasComputeType_t computeType_;
    cublasGemmAlgo_t    algo_;

public:
    cublasWrapper(cublasHandle_t   cublas_handle_t,
                  cublasLtHandle_t cublaslt_handle_t);
                //   BaseAllocator*   allocator);  // need self-defined memory allocator if use cublaslt
    ~cublasWrapper();

    void setFP32GemmConfig();
    void setFP16GemmConfig();

    void Gemm(cublasOperation_t transa,
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
              const int         ldc);
    
    void GemmStridedBatched(cublasOperation_t transa,
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
                            const int         batchCount);
};