
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>

#include <NvInfer.h>

#include "common.h"

// wrapper class for inference engine

class Engine {
public:
    Engine();
    ~Engine();
public:
    void Init(const std::vector<char> &plan);
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

Engine::~Engine() { }

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

    printf("Start %s\n", planPath);

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
    // record for automated extraction
    printf("#%s;%f\n", planPath, t / float(repeat)); 

    engine.EndInfer(output);

    Softmax(static_cast<int>(output.size()), output.data());
    PrintOutput(output);

    return 0;
}


