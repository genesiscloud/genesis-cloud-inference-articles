
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
//    Main program
//

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: bench_ts <torchscript-model-path>" << std::endl;
        return -1;
    }

    std::string name(argv[1]);

    std::cout << "Start model " << name << std::endl;

    int repeat = 100; 

    bool haveCuda = torch::cuda::is_available();
    assert(haveCuda);

    torch::Device device = torch::kCUDA;

    std::cout << "Loading model..." << std::endl;

    // load model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1], device);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading model" << std::endl;
        std::cerr << e.what_without_backtrace() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully" << std::endl;

    // switch off autograd, set evluation mode
    torch::NoGradGuard noGrad; 
    module.eval(); 

    // create input
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}, device));

    // warm up
    for (int i = 0; i < 10; i++) {
        module.forward(inputs);
    }

    // benchmark
    Timer timer;
    timer.Start();
    for (int i = 0; i < repeat; i++) {
        module.forward(inputs);
    }
    timer.Stop();
    float t = timer.Elapsed();
    std::cout << "Model " << name << ": elapsed time " << 
        t << " ms / " << repeat << " iterations = " << t / float(repeat) << std::endl; 
    // record for automated extraction
    std::cout << "#" << name << ";" << t / float(repeat) << std::endl;

    // execute model
    at::Tensor output = module.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    at::Tensor softmax = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5 = softmax.topk(5);
    at::Tensor labels = std::get<1>(top5);

    std::cout << labels[0] << std::endl;

    std::cout << "DONE" << std::endl << std::endl;
    return 0;
}


