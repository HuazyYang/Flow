#ifndef NVFLOW_ADVECT_H
#define NVFLOW_ADVECT_H
#include "Object.h"
#include "NvFlowImpl.h"
#include "SparseTexturePool.h"
#include "BlockManager.h"

struct NvFlowContext;

namespace NvFlow {

struct AdvectDesc {
    bool enableVTR;
    SparseTextureFront *velocity;
};

struct AdvectPerLayerParams {
    NvFlowFloat4 blendFactor;
    NvFlowFloat4 blendThreshold;
    NvFlowFloat4 damping;
    NvFlowFloat4 fade;
    float ignitionTemp;
    float burnPerTemp;
    float fuelPerBurn;
    float tempPerBurn;
    float smokePerBurn;
    float divergencePerBurn;
    float buoyancyPerTemp;
    float coolingRate;
    uint32_t materialIdx;
};

const struct AdvectParams {
    float deltaTime;
    NvFlowFloat3 valueCellSize;
    NvFlowFloat3 gravity;
    NvFlowUint4 emitterCount;
    NvFlowFloat4x4 gridToWorld;
    NvFlowFloat3 gridHalfSize;
    NvFlowFloat3 gridNewLocation;
    NvFlowFloat3 gridOldLocation;
    bool singlePassAdvection;
    void (*getPerLayer)(AdvectPerLayerParams *, void *, uint32_t);
    void *userdata;
};

struct Advect : NvFlowObject {
    virtual void advect(NvFlowContext *context, SparseTextureFront *value,
                        SparseTextureFront *velocity, const AdvectParams *params) = 0;

    virtual void advectCombustDensity(NvFlowContext *context, SparseTextureFront *density,
                                      SparseTextureFront *velocity,
                                      SparseTextureFront *coarseDensity,
                                      SparseFadeField *fadeField,
                                      NvFlowResource *emitterParameters,
                                      const AdvectParams *params) = 0;

    virtual void advectCombustVelocity(NvFlowContext *context, SparseTextureFront *velocity,
                                       SparseTextureFront *density,
                                       SparseTextureFront *coarseDensity,
                                       SparseFadeField *fadeField,
                                       NvFlowResource *emitterParameters,
                                       const AdvectParams *params) = 0;
};

Advect *createAdvect(NvFlowContext *context, const AdvectDesc *desc);

}  // namespace NvFlow

#endif /* NVFLOW_ADVECT_H */
