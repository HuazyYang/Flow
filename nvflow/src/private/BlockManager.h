#ifndef BLOCKMANAGER_H
#define BLOCKMANAGER_H

#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ClientHelper.h"
#include "VectorCached.h"
#include "SparseMapping.h"
#include "SparseMappable.h"
#include "SparseTexturePool.h"

namespace NvFlow {

struct BlockManagerDesc {
    bool enableVTR;
    bool lowLatencyMapping;
    // padding byte
    // padding byte
    NvFlowDim velocityVirtualDim;
    NvFlowDim densityVirtualDim;
};

struct BlockManagerPerLayerParams {
    float velocityWeight;
    float smokeWeight;
    float tempWeight;
    float fuelWeight;
    float velocityThreshold;
    float smokeThreshold;
    float tempThreshold;
    float fuelThreshold;
};

struct BlockManagerParams {
    void (*getPerLayer)(BlockManagerPerLayerParams *, void *, unsigned int);
    void (*reportVelocityLayerNumBlocks)(void *, unsigned int, unsigned int);
    void (*reportDensityLayerNumBlocks)(void *, unsigned int, unsigned int);
    void (*reportSummaryUpdate)(void *);
    void *userdata;
    bool bigEffectMode;
    float bigEffectPredictTime;
    NvFlowFloat3 gridHalfSize;
    NvFlowFloat3 gridLocation;
    NvFlowFloat3 gridTargetLocation;
};

struct BlockManager : NvFlowObject {
    virtual SparseMappingHandle map(NvFlowContext *context) = 0;
    virtual void unmap(NvFlowContext *context) = 0;
    virtual NvFlowDim getDim() = 0;
    virtual SparseMapping *getSparseMapping() = 0;
    virtual SparseFadeField *getFadeField() = 0;
    virtual NvFlowResult commit(SparseTextureFront *velocity, SparseTextureFront *density,
                                NvFlowContext *context, SparseMappable *const *fields,
                                uint32_t numFields) = 0;
    virtual bool updateLocation(NvFlowFloat3 *newGridLocation,
                                NvFlowFloat3 *oldGridLocation) = 0;
    virtual void update(NvFlowContext *context, SparseTextureFront *velocity,
                        SparseTextureFront *density, SparseTextureFront *coarseDensity,
                        SparseMappable *const *fields, uint32_t numFields,
                        const BlockManagerParams *params) = 0;
};

BlockManager *createBlockManager(NvFlowContext *context, const BlockManagerDesc *desc);

}  // namespace NvFlow

#endif /* BLOCKMANAGER_H */
