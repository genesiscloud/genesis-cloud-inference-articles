
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

#include <NvInfer.h>

#include "common.h"

// general helpers

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
    void Infer(const std::vector<float> &input, std::vector<float> &output);
    void DiagBindings();
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IRuntime> m_runtime;
    UniquePtr<nvinfer1::ICudaEngine> m_engine;
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
    m_engine.reset();
    m_runtime.reset(); 
    m_active = false;
}

void Engine::Infer(const std::vector<float> &input, std::vector<float> &output) {
    assert(m_active);
    UniquePtr<nvinfer1::IExecutionContext> context;
    context.reset(m_engine->createExecutionContext());
    if (context == nullptr) {
        Error("Error creating execution context");
    }
    CudaBuffer<float> inputBuffer;
    inputBuffer.Init(3 * 224 * 224);
    assert(inputBuffer.Size() == input.size());
    inputBuffer.Put(input.data());
    CudaBuffer<float> outputBuffer;
    outputBuffer.Init(1000);
    void *bindings[2];
    bindings[0] = inputBuffer.Data();
    bindings[1] = outputBuffer.Data();
    bool ok = context->executeV2(bindings);
    if (!ok) {
        Error("Error executing inference");
    }
    output.resize(outputBuffer.Size());
    outputBuffer.Get(output.data());
}

void Engine::DiagBindings() {
    int nbBindings = static_cast<int>(m_engine->getNbBindings());
    printf("Bindings: %d\n", nbBindings);
    for (int i = 0; i < nbBindings; i++) {
        const char *name = m_engine->getBindingName(i);
        bool isInput = m_engine->bindingIsInput(i);
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        std::string fmtDims = FormatDims(dims);
        printf("  [%d] \"%s\" %s [%s]\n", i, name, isInput ? "input" : "output", fmtDims.c_str());
    }
}

// I/O utilities

void ReadClasses(const char *path, std::vector<std::string> &classes) {
    std::string line;
    std::ifstream ifs(path, std::ios::in);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
    ifs.close();
}

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

void ReadInput(const char *path, std::vector<float> &input) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    size_t size = 3 * 224 * 224;
    input.resize(size);
    ifs.read(reinterpret_cast<char *>(input.data()), size * sizeof(float));
    ifs.close();
}

void PrintOutput(const std::vector<float> &output, const std::vector<std::string> &classes) {
    int top5p[5];
    float top5v[5];
    TopK(static_cast<int>(output.size()), output.data(), 5, top5p, top5v);
    printf("Top-5 results\n");
    for (int i = 0; i < 5; i++) {
        std::string label = classes[top5p[i]];
        float prob = 100.0f * top5v[i];
        printf("  [%d] %s %.2f%%\n", i, label.c_str(), prob);
    }
}

// main program

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: trt_infer_plan <plan_path> <input_path>\n");
        return 1;
    }
    const char *planPath = argv[1];
    const char *inputPath = argv[2];

    std::vector<std::string> classes;
    ReadClasses("imagenet_classes.txt", classes);

    std::vector<char> plan;
    ReadPlan(planPath, plan);

    std::vector<float> input;
    ReadInput(inputPath, input);

    std::vector<float> output;

    Engine engine;
    engine.Init(plan);
    engine.DiagBindings();
    engine.Infer(input, output);
    engine.Done();

    Softmax(static_cast<int>(output.size()), output.data());
    PrintOutput(output, classes);

    return 0;
}

