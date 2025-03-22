#include "ShapeSDF.h"
#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ClientHelper.h"

namespace NvFlow {

uint64_t Shape::getGPUBytesUsed() {
    return 0;
}

NvFlowShapeSDFData Shape::map(NvFlowContext *context) {
    NvFlowShapeSDFData result;
    auto mappedData = NvFlowTexture3DMap(context, m_sdf);
    result.data = (float *)mappedData.data;
    result.rowPitch = mappedData.rowPitch / sizeof(float);
    result.depthPitch = mappedData.depthPitch / sizeof(float);
    result.dim = m_desc.resolution;
    return result;
}

void Shape::unmap(NvFlowContext *context) {
    NvFlowTexture3DUnmap(context, m_sdf);
}

Shape::Shape(NvFlowContext *context, NvFlowTexture3D *texture) {
    NvFlowTexture3DDesc texture_desc;
    NvFlowTexture3DGetDesc(texture, &texture_desc);
    m_desc.resolution = texture_desc.dim;

    NvFlowTexture3DDesc texDesc;
    texDesc.format = eNvFlowFormat_r32_float;
    texDesc.dim = m_desc.resolution;
    texDesc.uploadAccess = 1;
    texDesc.downloadAccess = 0;
    m_sdf = NvFlowCreateTexture3D(context, &texDesc);
    NvFlowContextCopyTexture3D(context, m_sdf, texture);
}

Shape::Shape(NvFlowContext *context, const NvFlowShapeSDFDesc *desc) {
    m_desc = *desc;

    NvFlowTexture3DDesc texDesc = {};
    texDesc.format = eNvFlowFormat_r32_float;
    texDesc.dim = desc->resolution;
    texDesc.uploadAccess = 1;
    texDesc.downloadAccess = 0;
    m_sdf = NvFlowCreateTexture3D(context, &texDesc);
}

Shape::~Shape() {
    NvFlowReleaseTexture3D(m_sdf);
}

NvFlowShapeSDF *FlowCreateShape(NvFlowContext *context, const NvFlowShapeSDFDesc *desc) {
    return new Shape(context, desc);
}

NvFlowShapeSDF *FlowCreateShape(NvFlowContext *context, NvFlowTexture3D *texture) {
    return new Shape(context, texture);
}

}  // namespace NvFlow