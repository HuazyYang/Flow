#ifndef NVFLOW_SPARSETEXTUREMEMORYPOOL_H
#define NVFLOW_SPARSETEXTUREMEMORYPOOL_H
#include "Types.h"
#include "Object.h"
#include "NvFlowContextImpl.h"

struct NvFlowContext;

namespace NvFlow {

struct SparseTextureMemoryMappingDesc;
struct SparseTextureMemoryMapping;

NvFlowDim computeSparseTexturePoolGridDim(NvFlowFormat format, const NvFlowDim &virtualDim,
                                          float residentScale);

struct SparseTextureMemoryPoolDesc {
    NvFlowDim poolGridDim;
    NvFlowFormat format;
    bool enableVTR;
};

struct SparseTextureMemoryPoolStatus {
    uint32_t textureMemoryCount;
};

struct SparseTextureMemoryHandle {
    uint64_t uid;
};

struct SparseTextureMemoryBlockConfig {
    NvFlowDim blockDim;
    NvFlowDim linearBlockDim;
    NvFlowDim linearBlockOffset;
    NvFlowDim poolGridDim;
    NvFlowDim poolDim;

    NvFlowFormat format;
    uint32_t formatSizeInBytes;
    bool enableVTR;
};

struct SparseTextureMemoryMappingDesc {
    NvFlowDim virtualDim;
};

struct SparseTextureMemoryPool : NvFlowObject {
    virtual SparseTextureMemoryPoolDesc getDesc() = 0;
    virtual SparseTextureMemoryPoolStatus getStatus() = 0;
};

struct SparseTextureMemoryPoolInternal : SparseTextureMemoryPool {
    virtual SparseTextureMemoryMapping *createMemoryMapping(
        NvFlowContext *context, const SparseTextureMemoryMappingDesc *desc) = 0;

    virtual SparseTextureMemoryHandle acquireTextureMemory(NvFlowContext *context) = 0;
    virtual SparseTextureMemoryHandle acquireTextureMemoryNoAllocate(
        NvFlowContext *context) = 0;

    virtual uint32_t addRefTextureMemory(SparseTextureMemoryHandle handle) = 0;
    virtual uint32_t releaseTextureMemory(SparseTextureMemoryHandle handle) = 0;

    virtual SparseTextureMemoryBlockConfig getBlockConfig() = 0;
};

SparseTextureMemoryPool *createSparseTextureMemoryPool(
    NvFlowContext *context, const SparseTextureMemoryPoolDesc *desc);

struct SparseTextureMemoryMapping : NvFlowObject {
    virtual void pushMapping(uint64_t version, uint32_t *blockTableImage, uint32_t rowPitch,
                             uint32_t depthPitch, uint32_t blockTableImageBytes) = 0;
    virtual void updateMapping(NvFlowContext *context, uint32_t maxUpdateTextures) = 0;
    virtual bool canCommitMapping(NvFlowContext *context, uint64_t version) = 0;
    virtual NvFlowResult commitMapping(NvFlowContext *context, uint64_t version) = 0;
    virtual int64_t frontVersion() = 0;
    virtual NvFlowResourceRW *getResourceRW(NvFlowContext *context,
                                            SparseTextureMemoryHandle handle) = 0;
};

}  // namespace NvFlow

#endif /* NVFLOW_SPARSETEXTUREMEMORYPOOL_H */
