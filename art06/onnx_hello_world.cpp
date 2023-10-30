
#include <cstdio>
#include <cstdint>
#include <vector>

#include "onnxruntime_cxx_api.h"

Ort::Value CreateTensor(int64_t *data, size_t size, const int64_t *shape, size_t rank) {
    auto info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<int64_t>(info, data, size, shape, rank);
    return tensor;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

    const char *modelPath = "./onnx/bert_base_uncased_b1_s16.onnx";

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

    Ort::Session session(env, modelPath, sessionOptions);

    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t numInputNodes = session.GetInputCount();
    std::vector<const char *> inputNodeNames(numInputNodes);
    std::vector<int64_t> inputNodeDims;

    printf("Number of inputs = %zu\n", numInputNodes);

    // iterate over all input nodes
    for (int i = 0; i < numInputNodes; i++) {
        // print input node names
        const char *inputName = session.GetInputName(i, allocator);
        printf("Input %d: name=%s\n", i, inputName);
        inputNodeNames[i] = inputName;

        // print input node types
        Ort::TypeInfo typeInfo = session.GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        printf("Input %d: type=%d\n", i, type);

        // print input shapes/dims
        inputNodeDims = tensorInfo.GetShape();
        printf("Input %d: num_dims=%zu\n", i, inputNodeDims.size());
        for (size_t j = 0; j < inputNodeDims.size(); j++) {
            printf("Input %d: dim %zu=%jd\n", i, j, inputNodeDims[j]);
        }
    }

    // print number of model input nodes
    size_t numOutputNodes = session.GetOutputCount();
    std::vector<const char *> outputNodeNames(numOutputNodes);
    std::vector<int64_t> outputNodeDims;  

    printf("Number of outputs = %zu\n", numOutputNodes);

    // iterate over all output nodes
    for (int i = 0; i < numOutputNodes; i++) {
        // print output node names
        const char *outputName = session.GetOutputName(i, allocator);
        printf("Output %d: name=%s\n", i, outputName);
        outputNodeNames[i] = outputName;

        // print output node types
        Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        printf("Output %d: type=%d\n", i, type);

        // print output shapes/dims
        outputNodeDims = tensorInfo.GetShape();
        printf("Output %d: num_dims=%zu\n", i, outputNodeDims.size());
        for (size_t j = 0; j < outputNodeDims.size(); j++) {
            printf("Output %d: dim %zu=%jd\n", i, j, outputNodeDims[j]);
        }
    }

    int64_t inputIds[16] = {101, 7592, 2088, 102};
    int64_t tokenTypeIds[16] = {0, 0, 0, 0};
    int64_t attentionMask[16] = {1, 1, 1, 1};

    // create input tensor objects from data values
    const int64_t shape[] = {1, 16};
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(CreateTensor(inputIds, 16, shape, 2));
    inputTensors.push_back(CreateTensor(tokenTypeIds, 16, shape, 2));
    inputTensors.push_back(CreateTensor(attentionMask, 16, shape, 2));

    // run inference, get back output tensors
    auto outputTensors = 
        session.Run(
            Ort::RunOptions{nullptr}, 
            inputNodeNames.data(), 
            inputTensors.data(), 
            3, 
            outputNodeNames.data(), 
            2);

    // get pointer to output tensor float values
    float *output = outputTensors[0].GetTensorMutableData<float>();

    // print some output values; assume output shape [1, 16, 768]
    for (int i = 0; i < 16; i++) {
        int k = i * 768;
        printf("output[%d, *] %g %g %g ... %g %g %g\n",
            i, 
            output[k], output[k + 1], output[k + 2], 
            output[k + 765], output[k + 766], output[k + 767]);
    }

    // release buffers allocated by ORT allocator
    for(const char *nodeName: inputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }
    for(const char *nodeName: outputNodeNames) {
        allocator.Free(const_cast<void *>(reinterpret_cast<const void *>(nodeName)));
    }

    printf("DONE\n");
    return 0;
}


