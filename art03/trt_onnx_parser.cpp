
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "common.h"

// wrapper class for ONNX parser

class OnnxParser {
public:
    OnnxParser();
    ~OnnxParser();
public:
    void Init();
    void Parse(const char *onnxPath, const char *planPath);
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IBuilder> m_builder;
    UniquePtr<nvinfer1::INetworkDefinition> m_network;
    UniquePtr<nvinfer1::IBuilderConfig> m_config;
    UniquePtr<nvonnxparser::IParser> m_parser;
};

OnnxParser::OnnxParser(): m_active(false) { }

OnnxParser::~OnnxParser() { }

void OnnxParser::Init() {
    assert(!m_active);
    m_builder.reset(nvinfer1::createInferBuilder(m_logger));
    if (m_builder == nullptr) {
        Error("Error creating infer builder");
    }
    auto networkFlags = 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_network.reset(m_builder->createNetworkV2(networkFlags));
    if (m_network == nullptr) {
        Error("Error creating network");
    }
    m_config.reset(m_builder->createBuilderConfig());
    if (m_config == nullptr) {
        Error("Error creating builder config");
    }
    m_config->setMaxWorkspaceSize(256 * 1024 * 1024);
    m_config->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
    m_parser.reset(nvonnxparser::createParser(*m_network, m_logger));
    if (m_parser == nullptr) {
        Error("Error creating ONNX parser");
    }
}

void OnnxParser::Parse(const char *onnxPath, const char *planPath) {
    bool ok = m_parser->parseFromFile(onnxPath, static_cast<int>(m_logger.SeverityLevel()));
    if (!ok) {
        Error("ONNX parse error");
    }
    UniquePtr<nvinfer1::IHostMemory> plan(m_builder->buildSerializedNetwork(*m_network, *m_config));
    if (plan == nullptr) {
        Error("Network serialization error");
    }
    const void *data = plan->data();
    size_t size = plan->size();
    FILE *fp = fopen(planPath, "wb");
    if (fp == nullptr) {
        Error("Failed to create file %s", planPath);
    }
    fwrite(data, 1, size, fp);
    fclose(fp);
}

// main program

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: trt_onnx_parser <input_onnx_path> <output_plan_path>\n");
        return 1;
    }
    const char *onnxPath = argv[1];
    const char *planPath = argv[2];
    printf("Generate TensorRT plan for %s\n", onnxPath);
    OnnxParser parser;
    parser.Init();
    parser.Parse(onnxPath, planPath);   
    return 0;
}

