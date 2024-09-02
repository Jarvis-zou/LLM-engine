#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <numeric>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda_fp16.h>
#include "src/utils/string_utils.h"
#include "src/utils/macro.h"

enum class Device {
    CPU_PINNED,
    CPU,
    GPU,
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
class TensorWrapper;

struct Tensor {
        Device device;
        TensorType dtype;
        std::vector<int> shape;

        Tensor() = default;

        Tensor(const Device device, const TensorType dtype, const std::vector<int> shape)
        : device(device), dtype(dtype), shape(shape) {}

        // multiply all dimensions of shape, ep: size = B x C x W x H
        virtual int size() const {
            LLM_CHECK_WITH_INFO(!shape.empty(), "Tensor shape empty!");
            return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
        }

        template<typename T>
        TensorWrapper<T>* as(){
            return static_cast<TensorWrapper<T>*>(this);
        }

        inline std::string getDevice() const {
            std::unordered_map<Device, std::string> deviceType = {
                {Device::CPU, "CPU"},
                {Device::CPU_PINNED, "CPU_PINNED"},
                {Device::GPU, "GPU"}
            };

            return deviceType.at(device);
        }

        inline std::string getDtype() const {
            std::unordered_map<TensorType, std::string> dataTypes = {
                {TensorType::FP32, "FP32"},
                {TensorType::FP16, "FP16"},
                {TensorType::INT8, "INT8"},
                {TensorType::INT32, "INT32"},
                {TensorType::BOOL, "BOOL"},
                {TensorType::BYTES, "BYTES"},
                {TensorType::UNSUPPORTED, "UNSUPPORTED"}
            };

            return dataTypes.at(dtype);
        }

        // Get basic tensor config debug-info
        virtual inline std::string info() {
            return fmtstr("Tensor [device=%s, dtype=%s, shape=%s, data=%p]",
                                  getDevice().c_str(),
                                  getDtype().c_str(),
                                  vec2str(shape).c_str());
        }
};

template<typename T>
class TensorWrapper : public Tensor {
public:
    T* data;

    // Basic Constructor from Tensor class
    TensorWrapper(Device device, TensorType dtype, std::vector<int> shape) : Tensor(device, dtype, shape) {}

    // TensorWrapper Constructor
    TensorWrapper(Device device, TensorType dtype, std::vector<int> shape, T* data): Tensor(device, dtype, shape), data(data) {
            TensorType in_dtype = getTensorType<T>();  // check input dtype with pre-set type T
            LLM_CHECK_WITH_INFO(in_dtype == dtype, "Unmatched data type when construct TensorWrapper!");
            };
    
    int size() const override {
        // double check on input data
        LLM_CHECK_WITH_INFO(!shape.empty(), "Tensor shape empty!");
        LLM_CHECK_WITH_INFO(data != nullptr, "Input data is empty!");

        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }

    inline T getVal(int offset = 0) const {
        LLM_CHECK(device != Device::CPU);  // only can fetch data using [offset] on CPU, offset not allowed when data is on GPU memory
        return data[offset];
    }

    inline T* dataPtr() const {
        LLM_CHECK(device != Device::CPU);
        return (T*)data;
    }

    inline T* dataPtrAt(int offset) const {
        LLM_CHECK(device != Device::CPU);
        return (T*)data + offset;
    }

   virtual std::string info() {
    return fmtstr("Tensor [device=%s, dtype=%s, shape=%s, data=%p]",
                    getDevice().c_str(),
                          getDtype().c_str(),
                          vec2str(shape).c_str(),
                          data);
   }
}; 

struct TensorMap {
    std::unordered_map<std::string, Tensor*> tensor_map;

    TensorMap() = default;
    
    // Insert one Tensor into tensor_map each time
    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for (auto& pair : tensor_map) {
            if (isValid(pair.second)) { 
                insert(pair.first, pair.second); 
            }
            else {
                LLM_CHECK_WITH_INFO(isValid(pair.second), pair.first + "is not a valid tensor, tensor init failed");
            }
        }
    }

    inline bool isValid(const Tensor* tensor) { return tensor->size() > 0; }

    inline void insert(const std::string& name, Tensor* tensor) { tensor_map[name] = tensor; }

    inline void insert(std::pair<std::string, Tensor*> p) { tensor_map.insert(p); }  // Safe insert method, value will not be overrided

    inline void remove(const std::string& name) { tensor_map.erase(name); }

    inline bool hasKey(const std::string& name) { return tensor_map.find(name) != tensor_map.end(); }

    inline void replace(const std::string& name, Tensor* tensor) { 
        LLM_CHECK_WITH_INFO(hasKey(name), "No matched key in the map!");
        tensor_map[name] = tensor;
    }

    inline int size() { return tensor_map.size(); }

    inline Tensor* at(const std::string name) {
        LLM_CHECK_WITH_INFO(hasKey(name), "No matched key in tensor map!");
        return tensor_map.at(name);
    }

    inline Tensor* operator[](const std::string name) {
        LLM_CHECK_WITH_INFO(hasKey(name), "No matched key in tensor map!");
        return tensor_map.at(name);
    }

    inline std::vector<std::string> keys() {
        std::vector<std::string> keyVec;
        for (auto& pair : tensor_map) {
            keyVec.push_back(pair.first);
        }
        return keyVec;
    }

    inline void printKeys() {
        std::vector<std::string> keyNames = keys();
        vec2str(keyNames);
    }

    ~TensorMap() { tensor_map.clear(); }
};