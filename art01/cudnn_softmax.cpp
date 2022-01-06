
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cassert>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

// error handling

void Error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

// CUDA helpers

void CallCuda(cudaError_t stat) {
    if (stat != cudaSuccess) {
        Error("%s", cudaGetErrorString(stat));
    }
}

void *Malloc(int size) {
    void *ptr = nullptr;
    CallCuda(cudaMalloc(&ptr, size));
    return ptr;
}

void Free(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void Memget(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void Memput(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

template<typename T>
class CudaBuffer {
public:
    CudaBuffer(): 
        m_size(0), m_data(nullptr) { }
    ~CudaBuffer() {
        Done();
    }
public:
    void Init(int size) {
        assert(m_data == nullptr);
        m_size = size;
        m_data = static_cast<T *>(Malloc(size * sizeof(T)));
    }
    void Done() {
        if (m_data != nullptr) {
            Free(m_data);
            m_size = 0;
            m_data = nullptr;
        }
    }
    int Size() const {
        return m_size;
    }
    const T *Data() const {
        return m_data;
    }
    T *Data() {
        return m_data;
    }
    void Get(float *host) const {
        Memget(host, m_data, m_size * sizeof(T));
    }
    void Put(const float *host) {
        Memput(m_data, host, m_size * sizeof(T));
    }
private:
    int m_size;
    T *m_data;
};

// cuDNN helpers

void CallCudnn(cudnnStatus_t stat) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        Error("%s", cudnnGetErrorString(stat));
    }
}

cudnnHandle_t g_cudnnHandle;

void CudnnInit() {
    CallCudnn(cudnnCreate(&g_cudnnHandle));
}

void CudnnDone() {
    cudnnDestroy(g_cudnnHandle);
}

cudnnHandle_t CudnnHandle() {
    return g_cudnnHandle;
}

// wrapper class for softmax primitive

class Softmax {
public:
    Softmax();
    ~Softmax();
public:
    void Init(int n, int c, int h, int w);
    void Done();
    void Forward(const float *x, float *y);
private:
    bool m_active;
    cudnnHandle_t m_handle;
    cudnnTensorDescriptor_t m_desc;
};

Softmax::Softmax(): 
        m_active(false),
        m_handle(nullptr),
        m_desc(nullptr) { }

Softmax::~Softmax() {
    Done();
}

void Softmax::Init(int n, int c, int h, int w) {
    assert(!m_active);
    m_handle = CudnnHandle();
    CallCudnn(cudnnCreateTensorDescriptor(&m_desc));
    CallCudnn(cudnnSetTensor4dDescriptor(
        m_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n,
        c,
        h,
        w));
    m_active = true;
}

void Softmax::Done() {
    if (!m_active) {
        return;
    }
    cudnnDestroyTensorDescriptor(m_desc);
    m_active = false;
}

void Softmax::Forward(const float *x, float *y) {
    assert(m_active);
    static float one = 1.0f;
    static float zero = 0.0f;
    CallCudnn(cudnnSoftmaxForward(
        m_handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &one,
        m_desc,
        x,
        &zero,
        m_desc,
        y));
}

// Top-K helper

void TopK(int count, const float *data, int k, int *pos, float *val) {
    for (int i = 0; i < k; i++) {
        pos[i] = -1;
        val[i] = 0.0f;
    }
    for (int p = 0; p < count; p++) {
        float v = data[p];
        int j = -1;
        for (int i = 0; i < k; i++) {
            if (pos[i] < 0 || val[i] < v) {
                j = i;
                break;
            }
        }
        if (j >= 0) {
            for (int i = k - 1; i > j; i--) {
                pos[i] = pos[i-1];
                val[i] = val[i-1];
            }
            pos[j] = p;
            val[j] = v;
        }
    }
}

// handling input and output

void SetInput(CudaBuffer<float> &b) {
    int size = b.Size();
    std::vector<float> h(size);
    float *p = h.data();
    std::srand(1234);
    for (int i = 0; i < size; i++) {
        p[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    b.Put(p);
}

void GetOutput(const CudaBuffer<float> &b) {
    int size = b.Size();
    std::vector<float> h(size);
    float *p = h.data();
    b.Get(p);
    int top5p[5];
    float top5v[5];
    TopK(size, p, 5, top5p, top5v);
    for (int i = 0; i < 5; i++) {
        printf("[%d] pos %d val %g\n", i, top5p[i], top5v[i]);
    }
}

// main program

int main() {
    CudnnInit();
    Softmax softmax;
    int n = 1;
    int c = 1000;
    int h = 1;
    int w = 1;
    softmax.Init(n, c, h, w);
    int size = n * c * h * w;
    CudaBuffer<float> x;
    CudaBuffer<float> y;
    x.Init(size);
    y.Init(size);
    SetInput(x);
    softmax.Forward(x.Data(), y.Data());
    GetOutput(y);
    y.Done();
    x.Done();
    softmax.Done();
    CudnnDone();
}

