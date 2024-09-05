#include "src/utils/tensor.h"
#include "src/kernels/cal_padding_offset.cuh"

int main() {
    // hyper-param
    const int batch_size = 4;
    const int max_seq_len = 5;

    // init CPU var
    int* h_input_token_len;               // array represents valid token length for each sequence in batch (host)
    int* h_padding_offset;
    int* h_prefix_seq_len_sum;    

    // allocate memory on CPU
    h_input_token_len = (int*)malloc(batch_size * sizeof(int));
    h_padding_offset = (int*)malloc(batch_size * max_seq_len * sizeof(int));
    h_prefix_seq_len_sum = (int*)malloc((batch_size + 1) * sizeof(int));      // +1 for total valid token length in this batch


    // init GPU var
    int* d_input_token_len;  
    int* d_padding_offset;
    int* d_prefix_seq_len_sum;    

    // allocate memory on GPU
    cudaMalloc((void**)&d_input_token_len, batch_size * sizeof(int));
    cudaMalloc((void**)&d_padding_offset, batch_size * max_seq_len * sizeof(int));
    cudaMalloc((void**)&d_prefix_seq_len_sum, (batch_size + 1) * sizeof(int));


    // init test data
    for (int i = 0; i < batch_size; i++) {
        h_input_token_len[i] = i + 1;
    }
    TensorWrapper<int>* input_token_len = new TensorWrapper<int>(Device::GPU, TensorType::INT32, {batch_size}, d_input_token_len);
    TensorWrapper<int>* padding_offset = new TensorWrapper<int>(Device::GPU, TensorType::INT32, {batch_size, max_seq_len}, d_padding_offset);
    TensorWrapper<int>* prefix_seq_len_sum = new TensorWrapper<int>(Device::GPU, TensorType::INT32, {batch_size + 1}, d_prefix_seq_len_sum);

    // Memcpy from host to device
    cudaMemcpy(d_input_token_len, h_input_token_len, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, batch_size * max_seq_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_seq_len_sum, h_prefix_seq_len_sum, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Init kernel
    launchCalPaddingOffset(input_token_len,
                           padding_offset,
                           prefix_seq_len_sum);

    // Memcpy from device to host
    cudaMemcpy(h_padding_offset, d_padding_offset, batch_size * max_seq_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prefix_seq_len_sum, d_prefix_seq_len_sum, (batch_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);


    // check kernel result
    for(int i = 0; i < batch_size * max_seq_len; i++) {
        printf("padding_offset = %d\n", h_padding_offset[i]);
    }
    for(int i = 0; i < batch_size + 1; i++){
        printf("prefix_seq_len_sum =%d\n", h_prefix_seq_len_sum[i]);
    }

    // clean
    free(h_input_token_len);
    free(h_padding_offset);
    free(h_prefix_seq_len_sum);

    cudaFree(d_input_token_len);
    cudaFree(d_padding_offset);
    cudaFree(d_prefix_seq_len_sum);

    delete input_token_len;
    delete padding_offset;
    delete prefix_seq_len_sum;
}