#include "RenderMaterialPool.h"
#include "Object.h"
#include "NvFlowContextImpl.h"
#include "VectorCached.h"
#include "ClientHelper.h"

namespace NvFlow {

struct RenderMaterialPoolImpl : Object, NvFlowRenderMaterialPool {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    NvFlowRenderMaterialHandle getDefaultRenderMaterial() override;
    NvFlowRenderMaterialHandle createRenderMaterial(
        NvFlowContext *context, const NvFlowRenderMaterialParams *params) override;

    void releaseRenderMaterial(NvFlowRenderMaterialHandle handle) override;
    void renderMaterialUpdate(NvFlowRenderMaterialHandle handle,
                              const NvFlowRenderMaterialParams *params) override;
    NvFlowColorMapData renderMaterialColorMap(NvFlowContext *context,
                                              NvFlowRenderMaterialHandle handle) override;
    void renderMaterialColorUnmap(NvFlowContext *context,
                                  NvFlowRenderMaterialHandle handle) override;
    NvFlowResource *getColorMap(NvFlowRenderMaterialHandle handle) override;

    const NvFlowRenderMaterialParams *getMaterialParams(
        NvFlowRenderMaterialHandle handle) override;

    NvFlowRenderMaterialHandle getRenderMaterialHandle(
        NvFlowGridMaterialHandle gridMaterial) override;

    uint32_t getRenderMaterialHandles(NvFlowRenderMaterialHandle *results,
                                      uint32_t maxDstHandles,
                                      NvFlowGridMaterialHandle handle) override;

    // Details
    struct PerMaterial {
        bool valid;
        NvFlowTexture1D *colorMapFP32;
        NvFlowTexture1D *colorMapFP16;
        NvFlowRenderMaterialParams materialParams;
    };

    RenderMaterialPoolImpl(NvFlowContext *context,
                           const NvFlowRenderMaterialPoolDesc *desc);
    ~RenderMaterialPoolImpl();

    PerMaterial *handleToPtr(const NvFlowRenderMaterialHandle &handle);

    bool materialsEqual(const NvFlowGridMaterialHandle &left,
                        const NvFlowGridMaterialHandle &right);

    NvFlowRenderMaterialPoolDesc m_desc;
    NvFlowComputeShader *m_colorMapConvertCS;
    VectorCached<PerMaterial, 8> m_materials;
};

#include "colorMapConvertCS.hlsl.h"

RenderMaterialPoolImpl::RenderMaterialPoolImpl(NvFlowContext *context,
                                               const NvFlowRenderMaterialPoolDesc *desc)
    : m_desc{}, m_colorMapConvertCS{0} {
    m_desc = *desc;

    NvFlowComputeShaderDesc shaderDesc = {};
    shaderDesc.cs = g_colorMapConvertCS;
    shaderDesc.cs_length = sizeof(g_colorMapConvertCS);
    shaderDesc.label = L"colorMapConvertCS";
    m_colorMapConvertCS = NvFlowCreateComputeShader(context, &shaderDesc);

    NvFlowRenderMaterialParams defaultMaterialParams;
    NvFlowRenderMaterialParamsDefaults(&defaultMaterialParams);
    createRenderMaterial(context, &defaultMaterialParams);
}

RenderMaterialPoolImpl::~RenderMaterialPoolImpl() {
    for (auto &material : m_materials) {
        SafeRelease(material.colorMapFP16);
        SafeRelease(material.colorMapFP32);
    }
    SafeRelease(m_colorMapConvertCS);
}

RenderMaterialPoolImpl::PerMaterial *RenderMaterialPoolImpl::handleToPtr(
    const NvFlowRenderMaterialHandle &handle) {
    if (handle.pool == this && handle.uid < m_materials.size()) {
        return &m_materials[handle.uid];
    }
    if (handle.uid) return nullptr;
    return &m_materials[0];
}

bool RenderMaterialPoolImpl::materialsEqual(const NvFlowGridMaterialHandle &left,
                                            const NvFlowGridMaterialHandle &right) {
    return (left.grid == right.grid && left.uid == right.uid) || (!left.uid && !right.uid);
}

uint64_t RenderMaterialPoolImpl::getGPUBytesUsed() {
    return 0;
}

NvFlowRenderMaterialHandle RenderMaterialPoolImpl::getDefaultRenderMaterial() {
    NvFlowRenderMaterialHandle result;
    result.pool = this;
    result.uid = 0;
    return result;
}

NvFlowRenderMaterialHandle RenderMaterialPoolImpl::createRenderMaterial(
    NvFlowContext *context, const NvFlowRenderMaterialParams *params) {
    uint32_t allocIdx;
    for (allocIdx = 0; allocIdx < m_materials.size() && m_materials[allocIdx].valid;
         ++allocIdx)
        ;

    if (allocIdx == m_materials.size()) {
        allocIdx = m_materials.allocateBack();
        auto &material = m_materials[allocIdx];
        material.colorMapFP16 = 0;
        material.colorMapFP32 = 0;
    }

    auto &material = m_materials[allocIdx];
    material.valid = 1;

    material.materialParams = *params;

    NvFlowTexture1DDesc texDesc;
    if (!material.colorMapFP32) {
        texDesc.format = eNvFlowFormat_r32g32b32a32_float;
        texDesc.dim = m_desc.colorMapResolution;
        texDesc.uploadAccess = 1;
        material.colorMapFP32 = NvFlowCreateTexture1D(context, &texDesc);
    }

    if (!material.colorMapFP16) {
        texDesc.format = eNvFlowFormat_r16g16b16a16_float;
        texDesc.dim = m_desc.colorMapResolution;
        texDesc.uploadAccess = 0;
        material.colorMapFP16 = NvFlowCreateTexture1D(context, &texDesc);
    }

    NvFlowRenderMaterialHandle result;
    result.pool = this;
    result.uid = allocIdx;
    return result;
}

void RenderMaterialPoolImpl::releaseRenderMaterial(NvFlowRenderMaterialHandle handle) {
    auto material = handleToPtr(handle);
    if (material) material->valid = 0;
}

void RenderMaterialPoolImpl::renderMaterialUpdate(
    NvFlowRenderMaterialHandle handle, const NvFlowRenderMaterialParams *params) {
    auto ptr = handleToPtr(handle);
    if (ptr) ptr->materialParams = *params;
}

NvFlowColorMapData RenderMaterialPoolImpl::renderMaterialColorMap(
    NvFlowContext *context, NvFlowRenderMaterialHandle handle) {
    NvFlowColorMapData mappedData;
    auto ptr = handleToPtr(handle);
    if (ptr) {
        mappedData.data = (NvFlowFloat4 *)NvFlowTexture1DMap(context, ptr->colorMapFP32);
        mappedData.dim = m_desc.colorMapResolution;
    } else {
        ZeroMemory(&mappedData, sizeof(mappedData));
    }
    return mappedData;
}

void RenderMaterialPoolImpl::renderMaterialColorUnmap(NvFlowContext *context,
                                                      NvFlowRenderMaterialHandle handle) {
    auto ptr = handleToPtr(handle);
    if (ptr) {
        auto colorMap = ptr->colorMapFP32;
        NvFlowTexture1DUnmap(context, colorMap);

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_colorMapConvertCS;
        dparams.gridDim.x = (m_desc.colorMapResolution + 127) / 0x80;
        dparams.gridDim.y = 1;
        dparams.gridDim.z = 1;
        dparams.readOnly[0] = NvFlowTexture1DGetResource(ptr->colorMapFP32);
        dparams.readWrite[0] = NvFlowTexture1DGetResourceRW(ptr->colorMapFP16);
        NvFlowContextDispatch(context, &dparams);
    }
}

NvFlowResource *RenderMaterialPoolImpl::getColorMap(NvFlowRenderMaterialHandle handle) {
    auto material = handleToPtr(handle);
    if (material)
        return NvFlowTexture1DGetResource(material->colorMapFP16);
    else
        return nullptr;
}

const NvFlowRenderMaterialParams *RenderMaterialPoolImpl::getMaterialParams(
    NvFlowRenderMaterialHandle handle) {
    auto material = handleToPtr(handle);
    if (material) return &material->materialParams;
    return nullptr;
}

NvFlowRenderMaterialHandle RenderMaterialPoolImpl::getRenderMaterialHandle(
    NvFlowGridMaterialHandle handle) {
    uint32_t materialIdx;
    for (materialIdx = 0;; ++materialIdx) {
        if (materialIdx >= m_materials.size()) {
            NvFlowRenderMaterialHandle result;
            result.pool = 0;
            result.uid = 0;
            return result;
        }

        auto &material = m_materials[materialIdx];
        if (material.valid && materialsEqual(handle, material.materialParams.material)) {
            break;
        }
    }

    NvFlowRenderMaterialHandle result;
    result.pool = this;
    result.uid = materialIdx;
    return result;
}

uint32_t RenderMaterialPoolImpl::getRenderMaterialHandles(
    NvFlowRenderMaterialHandle *dstArray, uint32_t maxDstHandles,
    NvFlowGridMaterialHandle handle) {
    uint32_t count = 0;
    for (uint32_t materialIdx = 0; materialIdx < m_materials.size(); ++materialIdx) {
        auto &material = m_materials[materialIdx];
        if (material.valid) {
            if (materialsEqual(handle, material.materialParams.material)) {
                if (dstArray) {
                    dstArray[count].pool = this;
                    dstArray[count].uid = materialIdx;
                }
                ++count;

                if (dstArray) {
                    if (count >= maxDstHandles) break;
                }
            }
        }
    }

    return count;
}

NvFlowRenderMaterialPool *FlowCreateRenderMaterialPool(
    NvFlowContext *context, const NvFlowRenderMaterialPoolDesc *desc) {
    return new RenderMaterialPoolImpl(context, desc);
}

};  // namespace NvFlow