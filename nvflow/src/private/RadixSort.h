#ifndef NVFLOW_RADIXSORT_H
#define NVFLOW_RADIXSORT_H
#include "NvFlowObjectImpl.h"
#include "NvFlowContextImpl.h"

namespace NvFlow {

struct RadixSortBuffer {
    NvFlowBuffer *key;
    NvFlowBuffer *val;
};

struct RadixSortDesc {
    unsigned int maxSortBlocks;
};

struct RadixSortParams {
    unsigned int numSortBlocks;
};

struct RadixSort : NvFlowObject {
    virtual RadixSortBuffer getBuffer() = 0;
    virtual void sort(NvFlowContext *context, const RadixSortParams *params) = 0;
};

RadixSort *createRadixSort(NvFlowContext *context, const RadixSortDesc *desc);

struct RadixSortCPUBuffer {
    NvFlowUint2 *keyVal;
};

struct RadixSortCPUParams {
    unsigned int numElements;
};

struct RadixSortCPUDesc {
    unsigned int maxElements;
};

struct RadixSortCPU : NvFlowObject {
    virtual RadixSortCPUBuffer getBuffer() = 0;
    virtual void sort(NvFlowContext *context, const RadixSortCPUParams *params) = 0;
};

RadixSortCPU *createRadixSortCPU(const RadixSortCPUDesc *desc);

}  // namespace NvFlow

#endif /* NVFLOW_RADIXSORT_H */
