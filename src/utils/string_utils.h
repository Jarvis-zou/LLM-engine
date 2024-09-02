# pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args)
{
    // Disable format-security warning in this function.
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

template<typename T>
std::string arr2str(const T* arr, size_t size) {
    std::cout << size << std::endl;
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        ss << arr[i];
        if (i != size - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

template<typename T>
inline std::string vec2str(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i != vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}