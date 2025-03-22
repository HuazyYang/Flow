#ifndef NVFLOW_SDFGEN_H
#define NVFLOW_SDFGEN_H
#include "NvFlowContextImpl.h"
#include "NvFlowImpl.h"

struct NvFlowSDFGen : NvFlowObject {
    virtual void reset(NvFlowContext *context) = 0;
    virtual void voxelize(NvFlowContext *context, const NvFlowSDFGenMeshParams *params) = 0;
    virtual void update(NvFlowContext *context) = 0;
    virtual NvFlowTexture3D *shape(NvFlowContext *context) = 0;
};

namespace NvFlow {
NvFlowSDFGen *FlowCreateSDFGen(NvFlowContext *context, const NvFlowSDFGenDesc *desc);
};

#endif /* NVFLOW_SDFGEN_H */
