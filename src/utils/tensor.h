#pragma once
#include <vector>
#include <cstdint>
#include <numeric>
#include <cuda_fp16.h>

enum class Device {
    CPU_PINNED;
    CPU;
    GPU;
};

enum class TensorType {
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};

template<typename T>
inline TensorType getTensorType() {
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return TensorType::FP32;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return TensorType::FP16;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return TensorType::INT8;
    }
    else if (std::is_same<T, int32_t>::value || std::is_same<T, const int32_t>::value) {
        return TensorType::INT32;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return TensorType::BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const bool>::value) {
        return TensorType::BYTES;
    }
    else { return TensorType::UNSUPPORTED; }
}

template<typename T>
struct Tensor {
        Device device;
        TensorType dtype;
        std::vector<int> shape;

        Tensor() = default;

        Tensor(const Device device_, const TensorType dtype_, const std::vector<int> shape_)
        : device(device_), dtype(dtype_), shape(shape_) {}

        // multiply all dimensions of shape
        virtual int size() const {
            if (shape.empty()) { return 0; }
            return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
        }




    };