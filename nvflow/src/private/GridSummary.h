#ifndef NVFLOW_GRIDSUMMARY_H
#define NVFLOW_GRIDSUMMARY_H
#include "NvFlowObjectImpl.h"
#include "NvFlowImpl.h"

struct NvFlowContext;

struct NvFlowGridSummaryStateCPU : NvFlowObject {
    virtual uint32_t getNumLayers() = 0;
    virtual NvFlowGridMaterialHandle getLayerMaterial(uint32_t layerIdx) = 0;
    virtual void getSummaries(const NvFlowGridSummaryResult **results, uint32_t *numResults,
                              uint32_t layerIdx) = 0;
};

struct NvFlowGridSummary : NvFlowObject {
    virtual NvFlowGridSummaryStateCPU *createStateCPU() = 0;
    virtual void update(NvFlowContext *context,
                        const NvFlowGridSummaryUpdateParams *params) = 0;
    virtual void debugRender(NvFlowContext *context,
                             const NvFlowGridSummaryDebugRenderParams *debugParams) = 0;
};

namespace NvFlow {

NvFlowGridSummary *FlowCreateGridSummary(NvFlowContext *context,
                                         const NvFlowGridSummaryDesc *desc);
}

#endif /* NVFLOW_GRIDSUMMARY_H */
