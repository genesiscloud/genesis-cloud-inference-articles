
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

#include <NvInfer.h>

#include "common.h"

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

// deleter

struct Deleter {
    template<typename T>
    void operator()(T *obj) const {
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
    m_active = true;
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
        fprintf(stderr, "Usage: trt_bench_plan <plan_path>\n");
        return 1;
    }
    const char *planPath = argv[1];

    int repeat = 100;

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

