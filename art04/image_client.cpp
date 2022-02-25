// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include "http_client.h"

namespace tc = triton::client;

namespace {

void CheckError(const tc::Error &err, const char *msg) {
    if (!err.IsOk()) {
        std::cerr << "Error: " << msg << ": " << err << std::endl;
        exit(1);
    }
}

struct ModelInfo {
    std::string output_name;
    std::string input_name;
    std::string input_datatype; 
};

void ParseModel(const rapidjson::Document &model_metadata, ModelInfo &model_info) {
    const auto &input_itr = model_metadata.FindMember("inputs");
    const auto &output_itr = model_metadata.FindMember("outputs");

    const auto &input_metadata = *input_itr->value.Begin();
    const auto &output_metadata = *output_itr->value.Begin();

    model_info.output_name = std::string(
        output_metadata["name"].GetString(),
        output_metadata["name"].GetStringLength());
    model_info.input_name = std::string(
        input_metadata["name"].GetString(),
        input_metadata["name"].GetStringLength());
    model_info.input_datatype = std::string(
        input_metadata["datatype"].GetString(),
        input_metadata["datatype"].GetStringLength());
}

void FileToInputData(const std::string &filename, std::vector<uint8_t> &input_data) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open '" << filename << "'" << std::endl;
        exit(1);
    }
    size_t size = 3 * 224 * 224 * sizeof(float);
    input_data.resize(size);
    ifs.read(reinterpret_cast<char *>(input_data.data()), size);
    ifs.close();
}

void Postprocess(
        tc::InferResult *result,
        const std::string &filename, 
        const std::string &output_name, 
        size_t topk) {
    if (!result->RequestStatus().IsOk()) {
        std::cerr << "inference failed with error: " 
            << result->RequestStatus() << std::endl;
        exit(1);
    }

    std::vector<std::string> result_data;
    tc::Error err = result->StringData(output_name, &result_data);
    CheckError(err, "unable to get output data");

    size_t result_size = result_data.size();
    if (result_size != topk) {
        std::cerr << "unexpected number of strings in the result"
            << ", expected " << topk << ", got " << result_size << std::endl;
        exit(1);
    }

    std::cout << "Image '" << filename << "':" << std::endl;
    for (size_t c = 0; c < topk; c++) {
        std::istringstream is(result_data[c]);
        int count = 0;
        std::string token;
        while (getline(is, token, ':')) {
            if (count == 0) {
                std::cout << "    " << token;
            } else if (count == 1) {
                std::cout << " (" << token << ")";
            } else if (count == 2) {
                std::cout << " = " << token;
            }
            count++;
        }
        std::cout << std::endl;
    }
}

tc::Error ParseJson(rapidjson::Document *document, const std::string &json_str) {
    document->Parse(json_str.c_str(), json_str.size());
    if (document->HasParseError()) {
        return tc::Error(
            "failed to parse JSON at" + std::to_string(document->GetErrorOffset()) +
            ": " + std::string(GetParseError_En(document->GetParseError())));
    }
    return tc::Error::Success;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: image_client <model_name> <input_filename>" << std::endl;
        exit(1); 
    }

    std::string model_name{argv[1]};
    std::string input_filename{argv[2]};

    bool verbose = false;
    int topk = 5;
    std::string model_version = "";
    std::string url = "localhost:8000"; 
    tc::Headers http_headers; // empty

    std::vector<int64_t> shape{3, 224, 224};

    tc::Error err; 

    // Create the inference client for the server.
    std::unique_ptr<tc::InferenceServerHttpClient> http_client;
    err = tc::InferenceServerHttpClient::Create(&http_client, url, verbose);
    CheckError(err, "unable to create client for inference");

    std::string model_metadata;
    err = http_client->ModelMetadata(&model_metadata, model_name, model_version, http_headers);
    CheckError(err, "failed to get model metadata");
    rapidjson::Document model_metadata_json;
    err = ParseJson(&model_metadata_json, model_metadata);
    CheckError(err, "failed to parse model metadata");

    ModelInfo model_info;
    ParseModel(model_metadata_json, model_info);

    // Read input data
    std::vector<uint8_t> input_data;
    FileToInputData(input_filename, input_data);

    // Initialize the inputs with the data.
    tc::InferInput *input;
    err = tc::InferInput::Create(&input, model_info.input_name, shape, model_info.input_datatype);
    CheckError(err, "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr(input);

    tc::InferRequestedOutput *output;
    // Set the number of classification expected
    err = tc::InferRequestedOutput::Create(&output, model_info.output_name, topk);
    CheckError(err, "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

    std::vector<tc::InferInput *> inputs{input_ptr.get()};
    std::vector<const tc::InferRequestedOutput *> outputs{output_ptr.get()};

    // Configure context
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;

    // Prepare request
    err = input_ptr->Reset();
    CheckError(err, "failed resetting input");
    err = input_ptr->AppendRaw(input_data);
    CheckError(err, "failed setting input");
    options.request_id_ = "0";

    // Send request
    tc::InferResult *result;
    err = http_client->Infer(&result, options, inputs, outputs, http_headers);
    CheckError(err, "failed sending synchronous infer request");
    std::unique_ptr<tc::InferResult> result_ptr(result);

    // Post-process result to make prediction
    Postprocess(result_ptr.get(), input_filename, model_info.output_name, topk);

    return 0;
}

