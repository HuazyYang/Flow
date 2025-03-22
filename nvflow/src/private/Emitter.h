#ifndef EMITTER_H
#define EMITTER_H
#include "NvFlowObjectImpl.h"
#include "FlowMath.h"
#include "NvFlowContextImpl.h"
#include "NvFlowImpl.h"

namespace NvFlow {

struct BlockManager;
struct SparseTextureFront;

struct EmitterPerLayerParams {
    uint32_t materialIdx;
};

struct EmitterAllocParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowFloat4x4 worldToGrid;
    void (*getPerLayer)(EmitterPerLayerParams *, void *, uint32_t);
    void *userdata;
};

struct EmitterLookups {
    const NvFlowGridMaterialHandle *emitMaterials;
    unsigned int numEmitMaterials;
    NvFlowShapeSDF **sdfs;
    unsigned int numSdfs;
};

struct EmitterData {
    const NvFlowShapeDesc *shapes;
    uint32_t numShapes;
    uint32_t numShapeRefs;
    const NvFlowGridEmitParams *params;
    uint32_t numParams;
    EmitterLookups lookups;
};

struct EmitterVelocityParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowFloat4x4 worldToGrid;
    NvFlowDim virtualDim;
    void (*getPerLayer)(EmitterPerLayerParams *, void *, uint32_t);
    void *userdata;
};

struct EmitterDensityParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowFloat4x4 worldToGrid;
    NvFlowDim virtualDim;
    void (*getPerLayer)(EmitterPerLayerParams *, void *, uint32_t);
    void *userdata;
};

struct Emitter : NvFlowObject {
    virtual void allocate(NvFlowContext *context, BlockManager *blockManager,
                          const EmitterAllocParams *allocParams,
                          const EmitterData *data) = 0;
    virtual void allocateShape(NvFlowContext *context, BlockManager *BlockManager,
                               const EmitterAllocParams *allocParams,
                               const EmitterData *data) = 0;
    virtual void emitVelocityParameters(NvFlowContext *context,
                                        const EmitterVelocityParams *opParams,
                                        const EmitterData *emitData,
                                        NvFlowResource **parameterResources,
                                        NvFlowUint4 *parameterCount) = 0;

    virtual void emitDensityParameters(NvFlowContext *context,
                                       const EmitterDensityParams *opParams,
                                       const EmitterData *emitData,
                                       NvFlowResource **parameterResources,
                                       NvFlowUint4 *parameterCount) = 0;

    virtual void emitVelocity(NvFlowContext *context, SparseTextureFront *,
                              const EmitterVelocityParams *velocityParams,
                              const EmitterData *data) = 0;
    virtual void emitDensity(NvFlowContext *context, SparseTextureFront *,
                             SparseTextureFront *,
                             const EmitterDensityParams *densityParams,
                             const EmitterData *data) = 0;
};

struct EmitterDesc {
    bool enableVTR;
};

Emitter *createEmitter(NvFlowContext *context, const EmitterDesc *desc);

}  // namespace NvFlow

#endif /* EMITTER_H */
