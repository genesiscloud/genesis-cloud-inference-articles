
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <chrono>

#include "onnxruntime_cxx_api.h"

//
//    Argument parsing
//

bool Atoi(const char *s, int &v) {
    char *p;
    long t = strtol(s, &p, 10);
    if (*p != '\0') {
        return false;
    }
    int r = int(t);
    if (long(r) != t) {
        return false;
    }
    v = r;
    return true;
}

//
//    Timer
//

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

// construction/destruction

Timer::Timer(): elapsed(0.0f) { }

Timer::~Timer() { }

// interface

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

//
//    Input generator
//

void GenerateInput(int64_t *input, int volume) {
    for (int i = 0; i < volume; i++) {
        float value = static_cast<double>(std::rand()) / RAND_MAX;
        input[i] = static_cast<int64_t>(100.0 * value);
    }
}

//
//    ORT functions
//

Ort::Value CreateTensor(int64_t *data, size_t size, const int64_t *shape, size_t rank) {
    auto info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<int64_t>(info, data, size, shape, rank);
    return tensor;
}

void RunInference(
        Ort::Session &session,
        int numInputNodes,
        const char **inputNodeNames, 
        Ort::Value *inputTensors,
        int numOutputNodes,
        const char **outputNodeNames) {
    auto outputTensors = 
        session.Run(
            Ort::RunOptions{nullptr}, 
            inputNodeNames, 
            inputTensors, 
            numInputNodes, 
            outputNodeNames, 
            numOutputNodes);
}

//
//    Main program
//

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: onnx_bench <batch_size> <seq_len> [opt]\n");
        return 1;
    }

    // parse command line arguments

    int batchSize;
    if (!Atoi(argv[1], batchSize)) {
        fprintf(stderr, "Invalid batch size: '%s'\n", argv[1]);
        return 1;
    }
    int seqLen;
    if (!Atoi(argv[2], seqLen)) {
        fprintf(stderr, "Invalid sequence length: '%s'\n", argv[2]);
        return 1;
    }

    const char *optMode = nullptr;
    if (argc == 4) {
        optMode = argv[3];
        if (!strstr(optMode, "opt")) {
            fprintf(stderr, "Invalid optimization mode: '%s'\n", argv[3]);
            return 1;
        }
    }

    int warmup = 10;
    int measures = 100;

    char modelPath[128];
    if (optMode != nullptr) {
        sprintf(modelPath, "./onnx/bert_base_uncased_b%d_s%d_%s.onnx", batchSize, seqLen, optMode);
    } else {
        sprintf(modelPath, "./onnx/bert_base_uncased_b%d_s%d.onnx", batchSize, seqLen);
    }

    // create ORT session

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

    Ort::Session session(env, modelPath, sessionOptions);

    // fetch names of input and output nodes

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    assert(numInputNodes == 3);
    std::vector<const char *> inputNodeNames(numInputNodes);
    for (int i = 0; i < numInputNodes; i++) {
        const char *inputName = session.GetInputName(i, allocator);
        inputNodeNames[i] = inputName;
    }

    size_t numOutputNodes = session.GetOutputCount();
    assert(numOutputNodes == 2);
    std::vector<const char *> outputNodeNames(numOutputNodes);
    for (int i = 0; i < numOutputNodes; i++) {
        const char *outputName = session.GetOutputName(i, allocator);
        outputNodeNames[i] = outputName;
    }

    // set input data values

    int volume = batchSize * seqLen;
    std::vector<int64_t> inputIds(volume);
    std::vector<int64_t> tokenTypeIds(volume, 0);
    std::vector<int64_t> attentionMask(volume, 1);

    std::srand(1234);
    GenerateInput(inputIds.data(), volume);

    // create input tensor objects from data values

    const int64_t shape[] = {batchSize, seqLen};
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(CreateTensor(inputIds.data(), volume, shape, 2));
    inputTensors.push_back(CreateTensor(tokenTypeIds.data(), volume, shape, 2));
    inputTensors.push_back(CreateTensor(attentionMask.data(), volume, shape, 2));

    // warm up runs

    for (int i = 0; i < warmup; i++) {
        RunInference(
            session,
            numInputNodes, 
            inputNodeNames.data(), 
            inputTensors.data(), 
            numOutputNodes, 
            outputNodeNames.data());
    }

    // measured runs

    Timer timer;
    timer.Start();
    for (int i = 0; i < measures; i++) {
        RunInference(
            session,
            numInputNodes, 
            inputNodeNames.data(), 
            inputTensors.data(), 
            numOutputNodes, 
            outputNodeNames.data());
    }
    timer.Stop();
    float t = timer.Elapsed();
    printf("Model %s: elapsed time %f ms / %d = %f\n", modelPath, t, measures, t / float(measures));
    // record for automated extraction
    printf("#%s;%f\n", modelPath, t / float(measures)); 

    // release buffers allocated by ORT allocator
 
    for(const char *nodeName: inputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }
    for(const char *nodeName: outputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }

    return 0;
}


