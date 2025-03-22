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

struct SparseMappingHandle {
    SparseMapping *handle;
    uint64_t uid;
    uint32_t numLayers;

    SparseMappingLayerHandle mapAccumLayer(uint32_t layerIdx);
    SparseMappingLayerHandle mapMaskLayer(uint32_t layerIdx);

    void unmapAccumLayer(uint32_t layerIdx);
    void unmapMaskLayer(uint32_t layerIdx);
};

struct SparseMappingLayer {
    bool enable;
};

struct SparseMappingLayerTarget {
    bool enableTarget = 1;
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

struct SparseMappingDesc {
    NvFlowDim maskDim;
    uint32_t initialNumLayers;
};

struct SparseMapping : NvFlowObject {
    virtual SparseMappingHandle mapAccum(NvFlowContext *context) = 0;

    virtual SparseMappingLayerHandle mapAccumLayer(SparseMappingHandle handle,
                                                   uint32_t layerIdx) = 0;

    virtual void unmapAccumLayer(SparseMappingHandle handle, uint32_t layerIdx) = 0;

    virtual void unmapAccumBackLayer(SparseMappingHandle handle, uint32_t layerIdx) = 0;

    virtual void unmapAccum(NvFlowContext *context) = 0;

    virtual void swapAccum(NvFlowContext *context) = 0;

    virtual void clearAccum(NvFlowContext *context) = 0;

    virtual void shiftAccum(NvFlowContext *context, const NvFlowInt3 &offset) = 0;

    virtual SparseMappingHandle mapMask(NvFlowContext *context) = 0;

    virtual SparseMappingLayerHandle mapMaskLayer(SparseMappingHandle handle,
                                                  uint32_t layerIdx) = 0;

    virtual void unmapMaskLayer(SparseMappingHandle handle, uint32_t layerIdx) = 0;

    virtual void unmapMask(NvFlowContext *context) = 0;

    virtual void addLayer() = 0;

    virtual void enableLayer(uint32_t layerIdx) = 0;

    virtual void disableLayer(uint32_t layerIdx) = 0;

    virtual uint32_t getNumLayers() = 0;

    virtual NvFlowDim getMaskDim() = 0;
};

SparseMapping *createSparseMapping(NvFlowContext *context, const SparseMappingDesc *desc);

struct SparseMappingInternal : SparseMapping {
    virtual SparseMappingHandle mapMaskScaled(NvFlowContext *context, NvFlowDim dim) = 0;

    virtual SparseMappingLayerHandle mapMaskLayerScaled(NvFlowContext *context,
                                                        SparseMappingHandle handle,
                                                        uint32_t layerIdx) = 0;

    virtual void unmapMaskLayerScaled(NvFlowContext *context, SparseMappingHandle handle,
                                      uint32_t layerIdx) = 0;

    virtual void unmapMaskScaled(NvFlowContext *context) = 0;
};

}  // namespace NvFlow

#endif /* SPARSEMAPPING_H */
