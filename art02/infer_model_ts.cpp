
#include <cassert>
#include <iostream>
#include <fstream>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: infer_model_ts <path-to-exported-model> <path-to-input-data>" << std::endl;
        return -1;
    }

    // make sure CUDA us available; get CUDA device
    bool have_cuda = torch::cuda::is_available();
    assert(have_cuda);
    torch::Device device = torch::kCUDA;

    std::cout << "Loading model..." << std::endl;

    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1], device);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model" << std::endl;
        std::cerr << e.what_without_backtrace() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully" << std::endl;
    std::cout << std::endl;

    torch::NoGradGuard no_grad; // ensures that autograd is off
    module.eval(); // turn off dropout and other training-time layers/functions

    // read classes
    std::string line;
    std::ifstream ifsClasses("imagenet_classes.txt", std::ios::in);
    if (!ifsClasses.is_open()) {
        std::cerr << "Cannot open imagenet_classes.txt" << std::endl;
        return -1;
    }
    std::vector<std::string> classes;
    while (std::getline(ifsClasses, line)) {
        classes.push_back(line);
    }
    ifsClasses.close();

    // read input
    std::ifstream ifsData(argv[2], std::ios::in | std::ios::binary);
    if (!ifsData.is_open()) {
        std::cerr << "Cannot open " << argv[2] << std::endl;
        return -1;
    }
    size_t size = 3 * 224 * 224 * sizeof(float);
    std::vector<char> data(size);
    ifsData.read(data.data(), data.size());
    ifsData.close();

    // create input tensor on CUDA device 
    at::Tensor input = torch::from_blob(data.data(), {1, 3, 224, 224}, torch::kFloat);
    input = input.to(device);

    // create inputs
    std::vector<torch::jit::IValue> inputs{input};

    // execute model and package output as tensor
    at::Tensor output = module.forward(inputs).toTensor();

    // apply softmax and get Top-5 results
    namespace F = torch::nn::functional;
    at::Tensor softmax = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5 = softmax.topk(5);
    
    // get probabilities ans labels
    at::Tensor probs = std::get<0>(top5);
    at::Tensor labels = std::get<1>(top5);

    // print probabilities and labels
    for (int i = 0; i < 5; i++) {
        float prob = 100.0f * probs[0][i].item<float>();
        long label = labels[0][i].item<long>();
        std::cout << std::fixed << std::setprecision(2) << prob << "% " << classes[label] << std::endl; 
    }
    std::cout << std::endl;

    std::cout << "DONE" << std::endl;
    return 0;
}


