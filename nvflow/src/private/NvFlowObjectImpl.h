#ifndef NVFLOWOBJECT_H
#define NVFLOWOBJECT_H
#include <cstdlib>
#include <cstdint>

struct NvFlowObject {
    virtual uint32_t addRef() = 0;

    virtual uint32_t release() = 0;

    virtual uint64_t getGPUBytesUsed() = 0;
};  // namespace NvFlow

#endif /* NVFLOWOBJECT_H */
