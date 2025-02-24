#ifndef SPARSEMAPPING_H
#define SPARSEMAPPING_H
#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ClientHelper.h"
#include "VectorCached.h"

namespace NvFlow {

struct SparseMapping;
struct SparseMappingHandle;
struct SparseMappingLayer;
struct SparseMappingLayerHandle;

struct SparseMappingLayerHandle {
    NvFlowTexture3D *mask;
    NvFlowDim dim;
    bool enable;
};

struct SparseMappingLayer {
    bool enable;
};

struct SparseMappingLayerTarget {
    bool enableTarget;
};

struct SparseMappingMask {
    NvFlowDim dim;
};

struct SparseMappingLayerMask {
    NvFlowTexture3D *mask;
    NvFlowTexture3D *accumFront;
    NvFlowTexture3D *accumBack;
    bool maskDirty;
};

struct SparseMappingHandle {
    SparseMapping *mapping;
    uint64_t uid;
    uint32_t numLayers;
};

struct SparseMappingDesc {
    NvFlowDim maskDim;
    uint32_t initialNumLayers;
};

struct SparseMapping : NvFlowObject {
    virtual SparseMappingHandle *mapAccum(SparseMappingHandle *result,
                                          NvFlowContext *ctx) = 0;

    virtual SparseMappingLayerHandle *mapAccumLayer(SparseMappingLayerHandle *result,
                                                    SparseMappingHandle mapping,
                                                    uint32_t idx);

    virtual void unmapAccumLayer(SparseMappingHandle mapping, uint32_t idx) = 0;

    virtual void unmapAccumBackLayer(SparseMappingHandle mapping, uint32_t idx) = 0;

    virtual void unmapAccum(NvFlowContext *ctx) = 0;

    virtual void swapAccum(NvFlowContext *ctx) = 0;

    virtual void clearAccum(NvFlowContext *ctx) = 0;

    virtual void shiftAccum(NvFlowContext *ctx) = 0;

    virtual SparseMappingHandle *mapMask(SparseMappingHandle *result,
                                         NvFlowContext *ctx) = 0;

    virtual SparseMappingLayerHandle *mapMaskLayer(SparseMappingLayerHandle *result,
                                                   SparseMappingHandle mapping,
                                                   uint32_t idx);

    virtual void unmapMaskLayer(SparseMappingHandle mapping, uint32_t idx) = 0;

    virtual void unmapMask(NvFlowContext *ctx) = 0;

    virtual void addLayer() = 0;

    virtual void enableLayer(uint32_t idx) = 0;

    virtual void disableLayer(uint32_t idx) = 0;

    virtual uint32_t getNumLayers() = 0;

    virtual NvFlowDim getMaskDim(NvFlowDim *result) = 0;
};

struct SparseMappingInternal : SparseMapping {
    virtual SparseMappingHandle *mapMaskScaled(SparseMappingHandle *result,
                                               NvFlowContext *ctx, NvFlowDim dim) = 0;

    virtual SparseMappingLayerHandle *mapMAskLayerScaled(SparseMappingLayerHandle *result,
                                                         NvFlowContext *ctx,
                                                         SparseMappingHandle mapping,
                                                         uint32_t idx) = 0;

    virtual void unmapMaskLayerScaled(NvFlowContext *ctx, SparseMappingHandle mapping,
                                      uint32_t idx) = 0;

    virtual void unmapMaskScaled(NvFlowContext *ctx) = 0;
};

struct SparseMappingImpl : Object, SparseMappingInternal {
    // Implement
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()
    uint64_t getGPUBytesUsed() override;

    SparseMappingHandle *mapAccum(SparseMappingHandle *result, NvFlowContext *ctx) override;

    SparseMappingLayerHandle *mapAccumLayer(SparseMappingLayerHandle *result,
                                            SparseMappingHandle mapping, uint32_t idx) override;

    void unmapAccumLayer(SparseMappingHandle mapping, uint32_t idx) override;

    void unmapAccumBackLayer(SparseMappingHandle mapping, uint32_t idx) override;

    void unmapAccum(NvFlowContext *ctx) override;

    void swapAccum(NvFlowContext *ctx) override;

    void clearAccum(NvFlowContext *ctx) override;

    void shiftAccum(NvFlowContext *ctx) override;

    SparseMappingHandle *mapMask(SparseMappingHandle *result, NvFlowContext *ctx) override;

    SparseMappingLayerHandle *mapMaskLayer(SparseMappingLayerHandle *result,
                                           SparseMappingHandle mapping, uint32_t idx) override;

    void unmapMaskLayer(SparseMappingHandle mapping, uint32_t idx) override;

    void unmapMask(NvFlowContext *ctx) override;

    void addLayer() override;

    void enableLayer(uint32_t idx) override;

    void disableLayer(uint32_t idx) override;

    uint32_t getNumLayers() override;

    NvFlowDim getMaskDim(NvFlowDim *result) override;

    SparseMappingHandle *mapMaskScaled(SparseMappingHandle *result, NvFlowContext *ctx,
                                       NvFlowDim dim) override;

    SparseMappingLayerHandle *mapMAskLayerScaled(SparseMappingLayerHandle *result,
                                                 NvFlowContext *ctx,
                                                 SparseMappingHandle mapping,
                                                 uint32_t idx) override;

    void unmapMaskLayerScaled(NvFlowContext *ctx, SparseMappingHandle mapping,
                              uint32_t idx) override;

    void unmapMaskScaled(NvFlowContext *ctx) override;

    // Details
    SparseMappingImpl(NvFlowContext *ctx, const SparseMappingDesc *desc);

    ~SparseMappingImpl();

    SparseMappingDesc m_desc;
    uint64_t m_mapAccumVersion = 0;
    uint64_t m_mapMaskVersion = 0;
    unsigned int m_numLayersTarget = 0;
    unsigned int m_numLayers = 0;
    AutoPtr<NvFlowConstantBuffer> m_constantBuffer;
    AutoPtr<NvFlowComputeShader> m_sparseClearCS;
    AutoPtr<NvFlowComputeShader> m_sparseScaleCS;
    AutoPtr<NvFlowComputeShader> m_sparseShiftCS;
    VectorCached<SparseMappingLayerTarget, 8> m_layerTarget;
    VectorCached<SparseMappingLayer, 8> m_layer;
    VectorCached<SparseMappingMask, 8> m_mask;
    VectorCached2D<SparseMappingLayerMask, 8, 8> m_layerMask;
};

}  // namespace NvFlow

#endif /* SPARSEMAPPING_H */
