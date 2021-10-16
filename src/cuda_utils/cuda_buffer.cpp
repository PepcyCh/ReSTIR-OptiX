#include "cuda_utils/cuda_buffer.hpp"

#include <cassert>

#include "cuda_runtime.h"

#include "misc/check_macros.hpp"

CUdeviceptr CudaBuffer::DevicePtr() const {
    return reinterpret_cast<CUdeviceptr>(ptr_);
}

size_t CudaBuffer::Size() const {
    return size_;
}

void CudaBuffer::Resize(size_t size) {
    if (ptr_) {
        Free();
    }
    Alloc(size);
}

void CudaBuffer::Alloc(size_t size) {
    assert(ptr_ == nullptr);
    size_ = size;
    CUDA_CHECK(cudaMalloc(&ptr_, size));
}

void CudaBuffer::Free() {
    CUDA_CHECK(cudaFree(ptr_));
    size_ = 0;
    ptr_ = nullptr;
}

void CudaBuffer::Upload(const void *data, size_t size) const {
    assert(ptr_ != nullptr);
    assert(size <= size_);
    CUDA_CHECK(cudaMemcpy(ptr_, data, size, cudaMemcpyHostToDevice));
}

void CudaBuffer::AllocAndUpload(const void *data, size_t size) {
    Alloc(size);
    Upload(data, size);
}

void CudaBuffer::Download(void *receiver, size_t size) const {
    assert(ptr_ != nullptr);
    assert(size <= size_);
    CUDA_CHECK(cudaMemcpy(receiver, ptr_, size, cudaMemcpyDeviceToHost));
}