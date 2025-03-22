#ifndef NVFLOW_RENDERMATERIALPOOL_H
#define NVFLOW_RENDERMATERIALPOOL_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"

struct NvFlowRenderMaterialPool : NvFlowObject {
    virtual NvFlowRenderMaterialHandle getDefaultRenderMaterial() = 0;
    virtual NvFlowRenderMaterialHandle createRenderMaterial(
        NvFlowContext *context, const NvFlowRenderMaterialParams *params) = 0;

    virtual void releaseRenderMaterial(NvFlowRenderMaterialHandle handle) = 0;
    virtual void renderMaterialUpdate(NvFlowRenderMaterialHandle handle,
                                      const NvFlowRenderMaterialParams *params) = 0;
    virtual NvFlowColorMapData renderMaterialColorMap(
        NvFlowContext *context, NvFlowRenderMaterialHandle handle) = 0;
    virtual void renderMaterialColorUnmap(NvFlowContext *context,
                                          NvFlowRenderMaterialHandle handle) = 0;
    virtual NvFlowResource *getColorMap(NvFlowRenderMaterialHandle handle) = 0;

    virtual const NvFlowRenderMaterialParams *getMaterialParams(
        NvFlowRenderMaterialHandle handle) = 0;

    virtual NvFlowRenderMaterialHandle getRenderMaterialHandle(
        NvFlowGridMaterialHandle handle) = 0;

    virtual uint32_t getRenderMaterialHandles(NvFlowRenderMaterialHandle *results,
                                              uint32_t maxDstHandles,
                                              NvFlowGridMaterialHandle handle) = 0;
};

namespace NvFlow {

NvFlowRenderMaterialPool *FlowCreateRenderMaterialPool(
    NvFlowContext *context, const NvFlowRenderMaterialPoolDesc *desc);

}

#endif /* NVFLOW_RENDERMATERIALPOOL_H */
