#ifndef NVFLOW_VOLUMESHADOW_H
#define NVFLOW_VOLUMESHADOW_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"

struct NvFlowVolumeShadow : NvFlowObject {
    virtual void update(NvFlowContext *context, NvFlowGridExport *gridExport,
                        const NvFlowVolumeShadowParams *params) = 0;
    virtual NvFlowGridExport *getGridExport(NvFlowContext *context) = 0;
    virtual void debugRender(NvFlowContext *context,
                             const NvFlowVolumeShadowDebugRenderParams *params) = 0;
    virtual void getStats(NvFlowVolumeShadowStats *stats) = 0;
};

namespace NvFlow {

NvFlowVolumeShadow *FlowCreateVolumeShadow(NvFlowContext *context,
                                           const NvFlowVolumeShadowDesc *desc);

};

#endif /* NVFLOW_VOLUMESHADOW_H */
