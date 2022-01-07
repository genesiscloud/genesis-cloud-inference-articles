
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cassert>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>

// error handling

void Error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
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

// wrapper class for ONNX parser

class OnnxParser {
public:
    OnnxParser();
    ~OnnxParser();
public:
    void Init();
    void Done();
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

OnnxParser::~OnnxParser() {
    Done();
}

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
    m_parser.reset(nvonnxparser::createParser(*m_network, m_logger));
    if (m_parser == nullptr) {
        Error("Error creating ONNX parser");
    }
}

void OnnxParser::Done() {
    if (!m_active) {
        return;
    }
    m_parser.reset();
    m_config.reset();
    m_network.reset();
    m_builder.reset();
    m_active = false;
}

void OnnxParser::Parse(const char *onnxPath, const char *planPath) {
    bool ok = m_parser->parseFromFile(onnxPath, static_cast<int>(m_logger.SeverityLevel()));
    if (!ok) {
        Error("ONNX parse error");
    }
#if 1 // TODO: Revise this
    UniquePtr<nvinfer1::ICudaEngine> engine(
        m_builder->buildEngineWithConfig(*m_network, *m_config));
    if (engine == nullptr) {
        Error("Error building CUDA engine");
    }
    UniquePtr<nvinfer1::IHostMemory> plan(engine->serialize());
    if (plan == nullptr) {
        Error("Network serialization error");
    }
#else
    // requires version 8.x
    UniquePtr<nvinfer1::IHostMemory> plan(m_builder->buildSerializedNetwork(*m_network, *m_config));
    if (plan == nullptr) {
        Error("Network serialization error");
    }
#endif
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
        fprintf(stderr, "Usage: onnx_parser <input_onnx_path> <output_plan_path>\n");
        return 1;
    }
    const char *onnxPath = argv[1];
    const char *planPath = argv[2];
    OnnxParser parser;
    parser.Init();
    parser.Parse(onnxPath, planPath);   
    parser.Done();
    return 0;
}

