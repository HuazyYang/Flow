#ifndef NVFLOW_SHAPESDF_H
#define NVFLOW_SHAPESDF_H
#include "Object.h"
#include "NvFlowObjectImpl.h"
#include "NvFlowImpl.h"
#include "ClientHelper.h"

struct NvFlowContext;

struct NvFlowShapeSDF : NvFlowObject {
    virtual NvFlowShapeSDFData map(NvFlowContext *context) = 0;
    virtual void unmap(NvFlowContext *context) = 0;
};

namespace NvFlow {

NvFlowShapeSDF *FlowCreateShape(NvFlowContext *context, const NvFlowShapeSDFDesc *desc);
NvFlowShapeSDF *FlowCreateShape(NvFlowContext *context, NvFlowTexture3D *texture);

struct Shape : Object, NvFlowShapeSDF {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    NvFlowShapeSDFData map(NvFlowContext *context) override;
    void unmap(NvFlowContext *context) override;

    // Details
    Shape(NvFlowContext *context, NvFlowTexture3D *texture);
    Shape(NvFlowContext *context, const NvFlowShapeSDFDesc *desc);
    ~Shape();

    NvFlowShapeSDFDesc m_desc;
    NvFlowTexture3D *m_sdf;
};

}  // namespace NvFlow

#endif /* NVFLOW_SHAPESDF_H */
