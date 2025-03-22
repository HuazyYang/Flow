#ifndef NVFLOW_CROSSSECTION_H
#define NVFLOW_CROSSSECTION_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"

struct NvFlowCrossSection : NvFlowObject {
    virtual void render(NvFlowContext *context, const NvFlowCrossSectionParams *params) = 0;
};

namespace NvFlow {

NvFlowCrossSection *FlowCreateCrossSection(NvFlowContext *context,
                                           const NvFlowCrossSectionDesc *desc);
}

#endif /* NVFLOW_CROSSSECTION_H */
