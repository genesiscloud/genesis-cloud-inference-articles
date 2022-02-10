
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <string>
#include <chrono>

#include <cuda_runtime.h>

#include <NvInfer.h>

#include "common.h"

// error handling

void Error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

// wall clock

Timer::Timer(): elapsed(0.0f) { }

Timer::~Timer() { }

void Timer::Reset() {
    elapsed = 0.0f;
}

void Timer::Start() {
    start = std::chrono::steady_clock::now();
}

void Timer::Stop() {
    end = std::chrono::steady_clock::now();
    elapsed +=
        std::chrono::duration_cast<
            std::chrono::duration<float, std::milli>>(end - start).count();
}

float Timer::Elapsed() {
    return elapsed;
}

// CUDA helpers

void CallCuda(cudaError_t stat) {
    if (stat != cudaSuccess) {
        Error("%s", cudaGetErrorString(stat));
    }
}

void *Malloc(int size) {
    void *ptr = nullptr;
    CallCuda(cudaMalloc(&ptr, size));
    return ptr;
}

void Free(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void Memget(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void Memput(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

// general helpers

void Softmax(int count, float *data) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += std::exp(data[i]);
    }
    for (int i = 0; i < count; i++) {
        data[i] = std::exp(data[i]) / sum;
    }
}

void TopK(int count, const float *data, int k, int *pos, float *val) {
    for (int i = 0; i < k; i++) {
        pos[i] = -1;
        val[i] = 0.0f;
    }
    for (int p = 0; p < count; p++) {
        float v = data[p];
        int j = -1;
        for (int i = 0; i < k; i++) {
            if (pos[i] < 0 || val[i] < v) {
                j = i;
                break;
            }
        }
        if (j >= 0) {
            for (int i = k - 1; i > j; i--) {
                pos[i] = pos[i-1];
                val[i] = val[i-1];
            }
            pos[j] = p;
            val[j] = v;
        }
    }
}

// TensorRT helpers

std::string FormatDims(const nvinfer1::Dims &dims) {
    std::string result;
    char buf[64];
    int nbDims = static_cast<int>(dims.nbDims);
    for (int i = 0; i < nbDims; i++) {
        if (i > 0) {
            result += " ";
        }
        sprintf(buf, "%d", static_cast<int>(dims.d[i]));
        result += buf;
    }
    return result;
}

// logger

Logger::Logger():
        m_severityLevel(nvinfer1::ILogger::Severity::kWARNING) { }

Logger::~Logger() { }

nvinfer1::ILogger::Severity Logger::SeverityLevel() const {
    return m_severityLevel;
}

void Logger::SetSeverityLevel(nvinfer1::ILogger::Severity level) {
    m_severityLevel = level;
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    if (severity > m_severityLevel) {
        return;
    }
    fprintf(stderr, "%s: %s\n", GetSeverityString(severity), msg);
}

const char *Logger::GetSeverityString(nvinfer1::ILogger::Severity severity) {
    using T = nvinfer1::ILogger::Severity;
    switch (severity) {
    case T::kINTERNAL_ERROR:
        return "INTERNAL_ERROR";
    case T::kERROR:
        return "ERROR";
    case T::kWARNING:
        return "WARNING";
    case T::kINFO:
        return "INFO";
    case T::kVERBOSE:
        return "VERBOSE";
    default:
        return "?";
    }
}


