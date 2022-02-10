#pragma once

#include <cstdio>
#include <cassert>
#include <string>
#include <memory>
#include <chrono>

#include <cuda_runtime.h>

#include <NvInfer.h>

// error handling

void Error(const char *fmt, ...);

// wall clock

class Timer {
public:
    Timer();
    ~Timer();
public:
    void Reset();
    void Start();
    void Stop();
    float Elapsed();
private:
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    float elapsed;
};

// CUDA helpers

void CallCuda(cudaError_t stat);
void *Malloc(int size);
void Free(void *ptr);
void Memget(void *dst, const void *src, int size);
void Memput(void *dst, const void *src, int size);

template<typename T>
class CudaBuffer {
public:
    CudaBuffer(): 
        m_size(0), m_data(nullptr) { }
    ~CudaBuffer() {
        Done();
    }
public:
    void Init(int size) {
        assert(m_data == nullptr);
        m_size = size;
        m_data = static_cast<T *>(Malloc(size * sizeof(T)));
    }
    void Done() {
        if (m_data != nullptr) {
            Free(m_data);
            m_size = 0;
            m_data = nullptr;
        }
    }
    int Size() const {
        return m_size;
    }
    const T *Data() const {
        return m_data;
    }
    T *Data() {
        return m_data;
    }
    void Get(float *host) const {
        Memget(host, m_data, m_size * sizeof(T));
    }
    void Put(const float *host) {
        Memput(m_data, host, m_size * sizeof(T));
    }
private:
    int m_size;
    T *m_data;
};

// general helpers

void Softmax(int count, float *data);
void TopK(int count, const float *data, int k, int *pos, float *val);

// TensorRT helpers

std::string FormatDims(const nvinfer1::Dims &dims);

// logger

class Logger: public nvinfer1::ILogger {
public:
    Logger();
    ~Logger();
public:
    nvinfer1::ILogger::Severity SeverityLevel() const;
    void SetSeverityLevel(nvinfer1::ILogger::Severity level);
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
private:
    static const char *GetSeverityString(nvinfer1::ILogger::Severity severity);
private:
    nvinfer1::ILogger::Severity m_severityLevel;
};

// smart pointer alias

template<typename T>
using UniquePtr = std::unique_ptr<T>;


