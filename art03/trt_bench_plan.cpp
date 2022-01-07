
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include <cuda_runtime.h>

#include <NvInfer.h>

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

class WallClock {
public:
    WallClock();
    ~WallClock();
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

WallClock::WallClock(): elapsed(0.0f) { }

WallClock::~WallClock() { }

void WallClock::Reset() {
    elapsed = 0.0f;
}

void WallClock::Start() {
    start = std::chrono::steady_clock::now();
}

void WallClock::Stop() {
    end = std::chrono::steady_clock::now();
    elapsed +=
        std::chrono::duration_cast<
            std::chrono::duration<float, std::milli>>(end - start).count();
}

float WallClock::Elapsed() {
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

// logger

class Logger: public nvinfer1::ILogger {
public:
    Logger();
    ~Logger();
public:
    nvinfer1::ILogger::Severity SeverityLevel() const;
    void SetSeverityLevel(nvinfer1::ILogger::Severity level);
    void log(nvinfer1::ILogger::Severity severity, const char *msg) override;
private:
    static const char *GetSeverityString(nvinfer1::ILogger::Severity severity);
private:
    nvinfer1::ILogger::Severity m_severityLevel;
};

Logger::Logger():
        m_severityLevel(nvinfer1::ILogger::Severity::kWARNING) { }

Logger::~Logger() { }

nvinfer1::ILogger::Severity Logger::SeverityLevel() const {
    return m_severityLevel;
}

void Logger::SetSeverityLevel(nvinfer1::ILogger::Severity level) {
    m_severityLevel = level;
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) {
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

// deleter

struct Deleter {
    template<typename T>
    void operator()(T* obj) const {
        if (obj != nullptr) {
            obj->destroy();
        }
    }
};

template<typename T>
using UniquePtr = std::unique_ptr<T, Deleter>;

// wrapper class for inference engine

class Engine {
public:
    Engine();
    ~Engine();
public:
    void Init(const std::vector<char> &plan);
    void Done();
    void StartInfer(const std::vector<float> &input);
    void RunInfer();
    void EndInfer(std::vector<float> &output);
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IRuntime> m_runtime;
    UniquePtr<nvinfer1::ICudaEngine> m_engine;
    UniquePtr<nvinfer1::IExecutionContext> m_context;
    CudaBuffer<float> m_inputBuffer;
    CudaBuffer<float> m_outputBuffer;
};

Engine::Engine(): m_active(false) { }

Engine::~Engine() {
    Done();
}

void Engine::Init(const std::vector<char> &plan) {
    assert(!m_active);
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (m_runtime == nullptr) {
        Error("Error creating infer runtime");
    }
    m_engine.reset(m_runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (m_engine == nullptr) {
        Error("Error deserializing CUDA engine");
    }
}

void Engine::Done() {
    if (!m_active) {
        return;
    }
    m_context.reset();
    m_engine.reset();
    m_runtime.reset(); 
    m_active = false;
}

void Engine::StartInfer(const std::vector<float> &input) {
    assert(m_active);
    m_context.reset(m_engine->createExecutionContext());
    if (m_context == nullptr) {
        Error("Error creating execution context");
    }
    m_inputBuffer.Init(3 * 224 * 224);
    assert(m_inputBuffer.Size() == input.size());
    m_inputBuffer.Put(input.data());
    m_outputBuffer.Init(1000);
}

void Engine::RunInfer() {
    void *bindings[2];
    bindings[0] = m_inputBuffer.Data();
    bindings[1] = m_outputBuffer.Data();
    bool ok = m_context->executeV2(bindings);
    if (!ok) {
        Error("Error executing inference");
    }
}

void Engine::EndInfer(std::vector<float> &output) {
    output.resize(m_outputBuffer.Size());
    m_outputBuffer.Get(output.data());
}

// I/O utilities

void ReadPlan(const char *path, std::vector<char> &plan) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg();
    plan.resize(size);
    ifs.seekg(0, ifs.beg);
    ifs.read(plan.data(), size);
    ifs.close();
}

void GenerateInput(std::vector<float> &input) {
    int size = 3 * 224 * 224;
    input.resize(size);
    float *p = input.data();
    std::srand(1234);
    for (int i = 0; i < size; i++) {
        p[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

void PrintOutput(const std::vector<float> &output) {
    int top5p[5];
    float top5v[5];
    TopK(static_cast<int>(output.size()), output.data(), 5, top5p, top5v);
    printf("Top-5 results\n");
    for (int i = 0; i < 5; i++) {
        int label = top5p[i];
        float prob = 100.0f * top5v[i];
        printf("  [%d] %d %.2f%%\n", i, label, prob);
    }
}

// main program

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: bench_plan <plan_path>\n");
        return 1;
    }
    const char *planPath = argv[1];

    std::vector<char> plan;
    ReadPlan(planPath, plan);

    std::vector<float> input;
    GenerateInput(input);

    std::vector<float> output;

    Engine engine;
    engine.Init(plan);
    engine.StartInfer(input);

    for (int i = 0; i < 10; i++) {
        engine.RunInfer();
    }

    int repeat = 100;
    WallClock clock;
    clock.Start();
    for (int i = 0; i < repeat; i++) {
        engine.RunInfer();
    }
    clock.Stop();
    float t = clock.Elapsed();
    printf("Model %s: elapsed time %f ms / %d = %f\n", planPath, t, repeat, t / float(repeat));

    engine.EndInfer(output);
    engine.Done();

    Softmax(static_cast<int>(output.size()), output.data());
    PrintOutput(output);

    return 0;
}

