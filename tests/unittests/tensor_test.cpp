#pragma once
#include "src/utils/tensor.h"

int main(int argc, char **argv) {
     // [Test1]: Tensor.size() / Tensor.info()
     try {
         Tensor tensor = Tensor(Device::GPU, TensorType::FP32, std::vector<int>{1, 1});
         std::cout << tensor.info() << std::endl;
     } catch (const std::exception& e) {
         std::cerr << e.what() << std::endl;
     }

     // [Test2]: TensorWrapper.size() / TensorWrapper.info()
     try {
         float* input = new float[3]{1.0f, 2.0f, 3.0f};
         TensorWrapper<float> tensorwrapper = TensorWrapper<float>(Device::GPU, TensorType::FP32, std::vector<int>{1, 3}, input);
         std::cout << tensorwrapper.info() << std::endl;
     } catch (const std::exception& e) {
         std::cerr << e.what() << std::endl;
     }
    
    // [Test3]: TensorMap
//    TensorMap testMap;
//    Tensor* tensor1 = new Tensor(Device::GPU, TensorType::FP32, std::vector<int>{1, 10});
//    Tensor* tensor2 = new Tensor(Device::CPU, TensorType::FP16, std::vector<int>{1, 5});
//
//    testMap = {
//        {"tensor1", tensor1},
//        {"tensor2", tensor2},
//    };
//
//    // std::cout << (testMap.hasKey("tensor1") ? "Find tensor1" : "Can't find tensor1") << std::endl;
//    // std::cout << (testMap.hasKey("tensor3") ? "Find tensor3" : "Can't find tensor3") << std::endl;
//    std::cout << vec2str(testMap.keys()) << std::endl;
//    std::cout << testMap.size() << std::endl;
//    testMap.printKeys();
//    Tensor* res = testMap.at("tensor1");
//    res->info();
//
//    try {
//        testMap.replace("tensor4", tensor1);
//    } catch (const std::exception& e) {
//        std::cerr << e.what() << std::endl;
//    }


};