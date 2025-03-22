#ifndef NVFLOW_SPARSETEXTUREPOOL_H
#define NVFLOW_SPARSETEXTUREPOOL_H
#include "SparseMappable.h"
#include <nvflow/NvFlowShader.h>

namespace NvFlow {

struct SparseTextureMemoryPool;
struct SparseTexturePool;
struct SparseMapping;

struct SparseFadeField {
    void *userData;
    NvFlowResource *(*getFadeField)(void *userdata, bool isVelocityField,
                                    uint32_t layerIdx);
};

struct SparseLayeredBlockListView {
    NvFlowUint2 *dataCPU;
    uint32_t numBlocks;
};

struct SparseWriteBlockLayeredView {
    NvFlowShaderPointParams params;
    SparseLayeredBlockListView layeredBlockList;
};

struct SparseBlockMappingHandle {
    NvFlowResource *blockTable;
    NvFlowResource *blockList;
    uint32_t numBlocks;
};

struct SparseReadPointLayerView {
    NvFlowResource *data;
    SparseBlockMappingHandle mapping;
};

struct SparseReadPointLayeredView {
    NvFlowShaderPointParams pointParams;
    NvFlowShaderLinearParams downsampleParams;
    SparseLayeredBlockListView layeredBlockList;
};

struct SparseReadLinearLayeredView {
    NvFlowShaderLinearParams params;
    SparseLayeredBlockListView layeredBlockList;
};

struct SparseReadPointHandle {
    SparseTexturePool *pool;
    uint64_t uid;
    uint32_t numLayers;

    SparseReadPointLayeredView layeredView();
    SparseReadPointLayerView layerView(uint32_t layerIdx);
};

struct SparseReadLinearLayerView {
    NvFlowResource *data;
    SparseBlockMappingHandle mapping;
};

struct SparseReadLinearHandle {
    SparseTexturePool *pool;
    uint64_t uid;
    uint32_t numLayers;

    SparseReadLinearLayeredView layeredView();
    SparseReadLinearLayerView layerView(uint32_t layerIdx);
};

struct SparseWritePointLayeredView {
    NvFlowShaderPointParams params;
    SparseLayeredBlockListView layeredBlockList;
};

struct SparseWritePointLayerView {
    NvFlowResourceRW *data;
    SparseBlockMappingHandle mapping;
};

struct SparseWritePointHandle {
    SparseTexturePool *pool;
    uint64_t uid;
    uint32_t numLayers;

    SparseWritePointLayeredView layeredView();
    SparseWritePointLayerView layerView(uint32_t layerIdx);
};

struct SparseWriteLinearLayeredView {
    NvFlowShaderLinearParams params;
    SparseLayeredBlockListView layeredBlockList;
};

struct SparseWriteLinearLayerView {
    NvFlowResourceRW *data;
    SparseBlockMappingHandle mapping;
};

struct SparseWriteLinearHandle {
    SparseTexturePool *pool;
    uint64_t uid;
    uint32_t numLayers;

    SparseWriteLinearLayeredView layeredView();
    SparseWriteLinearLayerView layerView(uint32_t layerIdx);
};

struct SparseTextureHandle {
    SparseTexturePool *pool;
    uint64_t uid;

    SparseReadLinearHandle readLinearHandle(NvFlowContext *context);
    void readLinearHandleRelease(NvFlowContext *context);
    SparseReadPointHandle readPointHandle(NvFlowContext *context);
    void readPointHandleRelease(NvFlowContext *context);
    SparseWriteLinearHandle writeLinearHandle(NvFlowContext *context);
    SparseWritePointHandle writePointHandle(NvFlowContext *context);

    void addRefTexture();
    void releaseTexture();
};

struct SparseTexturePoolDesc {
    SparseTextureMemoryPool *memoryPool;
    NvFlowDim virtualDim;
    SparseMapping *initialMapping;
};

struct SparseTexturePoolConfig {
    uint32_t maxBlocks;
    NvFlowDim gridDim;
    NvFlowDim poolGridDim;
    NvFlowDim virtualDim;
};

struct SparseTextureFront {
    SparseTexturePool *pool;
    SparseTextureHandle front;

    void init(NvFlowContext *context, SparseTexturePool *poolIn);

    SparseTextureHandle acquireTexture(NvFlowContext *context);

    SparseTexturePoolDesc getDesc();

    SparseTexturePoolConfig getConfig();

    void swap(const SparseTextureHandle &newFront);
};

struct SparseTexturePool : NvFlowObject, SparseMappable {
    virtual SparseTextureHandle acquireTexture(NvFlowContext *context) = 0;
    virtual SparseTextureHandle acquireTextureNoAllocate(NvFlowContext *context) = 0;
    virtual uint32_t addRefTexture(SparseTextureHandle handle) = 0;
    virtual uint32_t releaseTexture(SparseTextureHandle handle) = 0;
    virtual SparseReadPointHandle readPointHandle(NvFlowContext *context,
                                                  SparseTextureHandle handle) = 0;
    virtual SparseReadPointLayeredView readPointLayeredView(
        SparseReadPointHandle handle) = 0;
    virtual SparseReadPointLayerView readPointLayerView(SparseReadPointHandle handle,
                                                        uint32_t layerIdx) = 0;
    virtual void readPointHandleRelease(NvFlowContext *context, SparseTextureHandle handle) = 0;
    virtual SparseReadLinearHandle readLinearHandle(NvFlowContext *context,
                                                    SparseTextureHandle handle) = 0;
    virtual SparseReadLinearLayeredView readLinearLayeredView(
        SparseReadLinearHandle handle) = 0;
    virtual SparseReadLinearLayerView readLinearLayerView(SparseReadLinearHandle handle,
                                                          uint32_t layerIdx) = 0;
    virtual void readLinearHandleRelease(NvFlowContext *context,
                                         SparseTextureHandle handle) = 0;

    virtual SparseWritePointHandle writePointHandle(NvFlowContext *context,
                                                    SparseTextureHandle handle) = 0;
    virtual SparseWritePointLayeredView writePointLayeredView(
        SparseWritePointHandle handle) = 0;
    virtual SparseWritePointLayerView writePointLayerView(SparseWritePointHandle handle,
                                                          uint32_t layerIdx) = 0;
    virtual SparseWriteLinearHandle writeLinearHandle(NvFlowContext *context,
                                                      SparseTextureHandle handle) = 0;
    virtual SparseWriteLinearLayeredView writeLinearLayeredView(
        SparseWriteLinearHandle handle) = 0;
    virtual SparseWriteLinearLayerView writeLinearLayerView(SparseWriteLinearHandle handle,
                                                            uint32_t layerIdx) = 0;
    virtual SparseTexturePoolConfig getConfig() = 0;
    virtual SparseTexturePoolDesc getDesc() = 0;
};

SparseTexturePool *createSparseTexturePool(NvFlowContext *context,
                                           const SparseTexturePoolDesc *desc);

struct SparseTexturePoolInternal : SparseTexturePool {};

}  // namespace NvFlow

#endif /* NVFLOW_SPARSETEXTUREPOOL_H */
