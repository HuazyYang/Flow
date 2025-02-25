#include "SparseMapping.h"
#include "Context.h"
#include <algorithm>

namespace NvFlow {

#include "sparseClearCS.hlsl.h"
#include "sparseScaleCS.hlsl.h"
#include "sparseShiftCS.hlsl.h"

uint32_t SparseMappingImpl::getNumLayers() {
    return m_numLayers;
}

NvFlowDim SparseMappingImpl::getMaskDim(NvFlowDim* result) {
    return m_desc.maskDim;
}

SparseMappingHandle* SparseMappingImpl::mapMaskScaled(SparseMappingHandle* result,
                                                      NvFlowContext* ctx, NvFlowDim dim) {
    return nullptr;
}

SparseMappingLayerHandle* SparseMappingImpl::mapMaskLayerScaled(
    SparseMappingLayerHandle* result, NvFlowContext* ctx, SparseMappingHandle mapping,
    uint32_t idx) {
    return nullptr;
}

void SparseMappingImpl::unmapMaskLayerScaled(NvFlowContext* ctx,
                                             SparseMappingHandle mapping, uint32_t idx) {}

void SparseMappingImpl::unmapMaskScaled(NvFlowContext* ctx) {}

SparseMappingImpl::SparseMappingImpl(NvFlowContext* ctx, const SparseMappingDesc* desc) {
    m_desc = *desc;

    auto createShader = [ctx](const BYTE* cs, uint64_t cs_length, const wchar_t* label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(ctx, &desc);
    };

    m_sparseClearCS =
        TakeOver(createShader(g_sparseClearCS, countof(g_sparseClearCS), L"sparseClearCS"));

    m_sparseScaleCS =
        TakeOver(createShader(g_sparseScaleCS, countof(g_sparseScaleCS), L"sparseScaleCS"));

    m_sparseShiftCS =
        TakeOver(createShader(g_sparseShiftCS, sizeof(g_sparseShiftCS), L"sparseShiftCS"));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = 32;
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(ctx, &cbDesc);
    m_mask.push_back({m_desc.maskDim});

    m_layerMask.allocateBackY();

    for (int layerIdx = 0; layerIdx < m_desc.initialNumLayers; ++layerIdx) {
        addLayer();

        SparseMappingHandle result;
        mapMask(&result, ctx);

        unmapMask(ctx);
    }
}

SparseMappingImpl::~SparseMappingImpl() {}

NvFlowTexture3D* SparseMappingImpl::createMask(NvFlowContext* ctx, const NvFlowDim& dim) {
    NvFlowTexture3DDesc maskTexDesc = {};
    maskTexDesc.format = eNvFlowFormat_r32_uint;
    maskTexDesc.dim = dim;
    maskTexDesc.uploadAccess = false;
    maskTexDesc.downloadAccess = false;
    auto maskTex = NvFlowCreateTexture3D(ctx, &maskTexDesc);
    return maskTex;
}

void SparseMappingImpl::clearMask(NvFlowContext* ctx, NvFlowTexture3D* maskTex) {
    NvFlowTexture3DDesc texDesc;
    NvFlowTexture3DGetDesc(maskTex, &texDesc);
    NvFlowDim gridDim = (texDesc.dim + 7) / 8;
    NvFlowDispatchParams dispatchParams;
    dispatchParams.shader = m_sparseClearCS;
    dispatchParams.gridDim = gridDim;
    dispatchParams.rootConstantBuffer = nullptr;
    dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(maskTex);
    NvFlowContextDispatch(ctx, &dispatchParams);
}

void SparseMappingImpl::syncLayers(NvFlowContext* ctx) {
    while (m_layer.size() < m_numLayersTarget)
        addLayer(ctx);

    m_numLayers = m_numLayersTarget;

    for (uint32_t layerIdx = 0; layerIdx < m_layer.size(); ++layerIdx) {
        auto& layer = m_layer[layerIdx];
        auto& layerTarget = m_layerTarget[layerIdx];
        if (!layerTarget.enableTarget && layer.enable) {
            auto& rootLayerMask = m_layerMask[0][layerIdx];
            clearMask(ctx, rootLayerMask.mask);

            for (uint32_t maskIdx = 1; maskIdx < m_layerMask.sizeY(); ++maskIdx) {
                auto& layerMask = m_layerMask[maskIdx][layerIdx];
                layerMask.maskDirty = 1;
            }
        }
        layer.enable = layerTarget.enableTarget;
    }
}

uint64_t SparseMappingImpl::getGPUBytesUsed() {
    uint64_t totalBytes = 0;
    for (uint32_t i = 0; i < m_layerMask.sizeY(); ++i) {
        auto r = m_layerMask[i];
        for (uint32_t j = 0; i < m_layerMask.sizeX(); ++i) {
            if (r[j].mask) {
                auto maskObj = NvFlowTexture3DGetContextObject(r[j].mask);
                totalBytes += NvFlowContextObjectGetGPUBytesUsed(maskObj);
            }
        }
    }
    return totalBytes;
}

SparseMappingHandle* SparseMappingImpl::mapAccum(SparseMappingHandle* result,
                                                 NvFlowContext* ctx) {
    return nullptr;
}

SparseMappingLayerHandle* SparseMappingImpl::mapAccumLayer(SparseMappingLayerHandle* result,
                                                           SparseMappingHandle mapping,
                                                           uint32_t idx) {
    return nullptr;
}

void SparseMappingImpl::unmapAccumLayer(SparseMappingHandle mapping, uint32_t idx) {}

void SparseMappingImpl::unmapAccumBackLayer(SparseMappingHandle mapping, uint32_t idx) {}

void SparseMappingImpl::unmapAccum(NvFlowContext* ctx) {}

void SparseMappingImpl::swapAccum(NvFlowContext* ctx) {}

void SparseMappingImpl::clearAccum(NvFlowContext* ctx) {
    SparseMappingHandle mapped;

    mapAccum(&mapped, ctx);

    for (uint32_t layerIdx = 0; layerIdx < mapped.numLayers; ++layerIdx) {

        SparseMappingLayerHandle mappedLayer;
        mapAccumLayer(&mappedLayer, mapped, layerIdx);

        if (mappedLayer.enable) {
            clearMask(ctx, mappedLayer.mask);
        }

        unmapAccumLayer(mapped, layerIdx);
    }

    unmapAccum(ctx);
}

void SparseMappingImpl::shiftAccum(NvFlowContext* ctx) {}

SparseMappingHandle* SparseMappingImpl::mapMask(SparseMappingHandle* result,
                                                NvFlowContext* ctx) {
    ++m_mapMaskVersion;
    syncLayers(ctx);
    result->mapping = this;
    result->numLayers = m_layerTarget.size();
    result->uid = m_mapMaskVersion;
    return result;
}

SparseMappingLayerHandle* SparseMappingImpl::mapMaskLayer(SparseMappingLayerHandle* result,
                                                          SparseMappingHandle handle,
                                                          uint32_t layerIdx) {
    if (handle.uid == m_mapMaskVersion && layerIdx < m_layerTarget.size()) {
        auto& layer = m_layer[layerIdx];
        auto& rootLayerMask = m_layerMask[0][layerIdx];

        result->mask = rootLayerMask.mask;
        result->dim = m_mask[0].dim;
        result->enable = layer.enable;
        return result;
    } else {
        memset(result, 0, sizeof(SparseMappingLayerHandle));
    }
    return result;
}

void SparseMappingImpl::unmapMaskLayer(SparseMappingHandle handle, uint32_t layerIdx) {
    if (handle.uid == m_mapMaskVersion && layerIdx < m_layerTarget.size()) {
        auto& layer = m_layer[layerIdx];
        if (layer.enable) {
            for (uint32_t maskIdx = 1; maskIdx < m_layerMask.sizeY(); ++maskIdx) {
                auto& layerMask = m_layerMask[maskIdx][layerIdx];
                layerMask.maskDirty = true;
            }
        }
    }
}

void SparseMappingImpl::unmapMask(NvFlowContext* ctx) {
    ++m_mapMaskVersion;
}

void SparseMappingImpl::addLayer() {
    ++m_numLayersTarget;
    while (m_layerTarget.size() < m_numLayersTarget) {
        uint32_t layerIdx = m_layerTarget.allocateBack();
        auto& layerTarget = m_layerTarget[layerIdx];
        layerTarget.enableTarget = false;
    }
}

void SparseMappingImpl::addLayer(NvFlowContext* ctx) {
    uint32_t layerAllocIdx = m_layer.allocateBack();
    uint32_t layerMaskAllocIdx = m_layerMask.allocateBackX();

    NVFLOW_ASSERT(layerAllocIdx == layerMaskAllocIdx);

    auto& layer = m_layer[layerAllocIdx];
    layer.enable = 1;

    for (uint32_t maskIdx = 0; maskIdx < m_layerMask.sizeY(); ++maskIdx) {
        auto r = m_layerMask[maskIdx];
        auto& layerMask = r[layerMaskAllocIdx];
        layerMask.mask = nullptr;
        layerMask.maskDirty = 1;
    }

    auto& rootLayerMask = m_layerMask[0][layerMaskAllocIdx];
    rootLayerMask.maskDirty = 0;

    auto& Mask = m_mask[0];

    rootLayerMask.mask = createMask(ctx, Mask.dim);
    rootLayerMask.accumFront = createMask(ctx, Mask.dim);
    rootLayerMask.accumBack = createMask(ctx, Mask.dim);
}

void SparseMappingImpl::enableLayer(uint32_t idx) {}

void SparseMappingImpl::disableLayer(uint32_t idx) {}

}  // namespace NvFlow