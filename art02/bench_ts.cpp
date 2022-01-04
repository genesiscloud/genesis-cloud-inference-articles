
#include <cstdio>
#include <cassert>
#include <string>
#include <iostream>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

//
//    WallClock
//

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

// construction/destruction

WallClock::WallClock(): elapsed(0.0f) { }

WallClock::~WallClock() { }

// interface

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

//
//    Main program
//

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: bench_ts <path-to-exported-model>" << std::endl;
        return -1;
    }

    std::string name(argv[1]);

    if (name.find("googlenet") != std::string::npos) {
        std::cout << "Skip inference: " << name << std::endl;
        std::cout << "DONE" << std::endl << std::endl;
        return 0;
    }

    // execute model and package output as tensor
    std::cout << "Start model " << name << std::endl;

    int repeat = 100; // make it configuravle?

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

    // ensures that autograd is off
    torch::NoGradGuard no_grad; 
    // turn off dropout and other training-time layers/functions
    module.eval(); 

    // create input
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}, device));

    // warm up
    for (int i = 0; i < 10; i++) {
        module.forward(inputs);
    }

    // benchmark
    WallClock clock;
    clock.Start();
    for (int i = 0; i < repeat; i++) {
        module.forward(inputs);
    }
    clock.Stop();
    float t = clock.Elapsed();
    std::cout << "Model " << name << ": elapsed time " << 
        t << " ms / " << repeat << " iterations = " << t / float(repeat) << std::endl; 

    // execute model and package output as tensor
    at::Tensor output = module.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
    at::Tensor top5 = std::get<1>(top5_tensor);

    std::cout << top5[0] << std::endl;

    std::cout << "DONE" << std::endl << std::endl;
    return 0;
}


