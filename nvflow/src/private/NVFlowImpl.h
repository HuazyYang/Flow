#ifndef NVFLOWIMPL_H
#define NVFLOWIMPL_H
#include <nvflow/NvFlow.h>
#include "NvFlowObjectImpl.h"

struct NvFlowGrid : NvFlowObject {
    virtual void reset(const NvFlowGridResetDesc *) = 0;
    virtual NvFlowResult querySupport(NvFlowContext *, NvFlowSupport *) = 0;
    virtual NvFlowResult queryTime(NvFlowQueryTime *, NvFlowQueryTime *) = 0;
    virtual void GPUMemUsage(NvFlowUint64 *) = 0;
    virtual void update(NvFlowContext *, float) = 0;
    virtual void setTargetLocation(NvFlowFloat3) = 0;
    virtual void setParams(const NvFlowGridParams *) = 0;
    virtual NvFlowGridMaterialHandle *getDefaultMaterial(
        NvFlowGridMaterialHandle *result) = 0;
    virtual NvFlowGridMaterialHandle *createMaterial(NvFlowGridMaterialHandle *result,
                                                     const NvFlowGridMaterialParams *) = 0;
    virtual void releaseMaterial(NvFlowGridMaterialHandle) = 0;
    virtual void setMaterialParams(NvFlowGridMaterialHandle,
                                   const NvFlowGridMaterialParams *) = 0;
    virtual void emit(const NvFlowShapeDesc *, NvFlowUint, const NvFlowGridEmitParams *,
                      NvFlowUint) = 0;
    virtual void updateEmitMaterials(NvFlowGridMaterialHandle *, NvFlowUint) = 0;
    virtual void updateEmitSDFs(NvFlowShapeSDF **, NvFlowUint) = 0;
    virtual void emitCustomRegisterAllocFunc(
        void (*)(void *, const NvFlowGridEmitCustomAllocParams *), void *) = 0;
    virtual void emitCustomRegisterEmitFunc(
        NvFlowGridTextureChannel,
        void (*)(void *, NvFlowUint *, const NvFlowGridEmitCustomEmitParams *), void *) = 0;
    virtual void emitCustomGetLayerParams(const NvFlowGridEmitCustomEmitParams *,
                                          NvFlowUint,
                                          NvFlowGridEmitCustomEmitLayerParams *) = 0;
    virtual NvFlowGridExport *getGridExport(NvFlowContext *) = 0;
};

#endif /* NVFLOWIMPL_H */
