#pragma once

#include "cuda.h"

class CudaBuffer {
public:
    CUdeviceptr DevicePtr() const;

    template <typename T>
    T *TypedPtr() const {
        return static_cast<T *>(ptr_);
    }

    size_t Size() const;

    void Resize(size_t size);

    void Alloc(size_t size);

    void Free();

    void Upload(const void *data, size_t size) const;

    void AllocAndUpload(const void *data, size_t size);

    void Download(void *receiver, size_t size) const;

private:
    size_t size_ = 0;
    void *ptr_ = nullptr;
};