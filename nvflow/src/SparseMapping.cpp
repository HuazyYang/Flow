#include "SparseMapping.h"
#include "Context.h"
#include <algorithm>

namespace NvFlow {

struct SparseMappingImpl : Object, SparseMappingInternal {
    // Implement
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()
    uint64_t getGPUBytesUsed() override;

    SparseMappingHandle mapAccum(NvFlowContext* context) override;

    SparseMappingLayerHandle mapAccumLayer(SparseMappingHandle handle,
                                           uint32_t layerIdx) override;

    void unmapAccumLayer(SparseMappingHandle handle, uint32_t layerIdx) override;

    SparseMappingLayerHandle mapAccumBackLayer(SparseMappingHandle handle,
                                               uint32_t layerIdx);

    void unmapAccumBackLayer(SparseMappingHandle handle, uint32_t layerIdx) override;

    void unmapAccum(NvFlowContext* context) override;

    void swapAccum(NvFlowContext* context) override;

    void clearAccum(NvFlowContext* context) override;

    void shiftAccum(NvFlowContext* context, const NvFlowInt3& offset) override;

    SparseMappingHandle mapMask(NvFlowContext* context) override;

    SparseMappingLayerHandle mapMaskLayer(SparseMappingHandle handle,
                                          uint32_t layerIdx) override;

    void unmapMaskLayer(SparseMappingHandle handle, uint32_t layerIdx) override;

    void unmapMask(NvFlowContext* context) override;

    void addLayer() override;

    void addLayer(NvFlowContext* context);

    void enableLayer(uint32_t layerIdx) override;

    void disableLayer(uint32_t layerIdx) override;

    uint32_t getNumLayers() override;

    NvFlowDim getMaskDim() override;

    SparseMappingHandle mapMaskScaled(NvFlowContext* context, NvFlowDim dim) override;

    SparseMappingLayerHandle mapMaskLayerScaled(NvFlowContext* context,
                                                SparseMappingHandle handle,
                                                uint32_t layerIdx) override;

    void unmapMaskLayerScaled(NvFlowContext* context, SparseMappingHandle handle,
                              uint32_t layerIdx) override;

    void unmapMaskScaled(NvFlowContext* context) override;

    // Details
    SparseMappingImpl(NvFlowContext* context, const SparseMappingDesc* desc);
    ~SparseMappingImpl();

    NvFlowTexture3D* createMask(NvFlowContext* context, const NvFlowDim& mask);

    void clearMask(NvFlowContext* context, NvFlowTexture3D* maskTex);

    void shiftMask(NvFlowContext* context, NvFlowTexture3D* destTex,
                   NvFlowTexture3D* srcTex, const NvFlowInt3& offset);

    void syncLayers(NvFlowContext* context);

    SparseMappingDesc m_desc;
    uint64_t m_mapAccumVersion = 0;
    uint64_t m_mapMaskVersion = 0;
    unsigned int m_numLayersTarget = 0;
    unsigned int m_numLayers = 0;
    NvFlowConstantBuffer* m_constantBuffer = 0;
    NvFlowComputeShader* m_sparseClearCS = 0;
    NvFlowComputeShader* m_sparseScaleCS = 0;
    NvFlowComputeShader* m_sparseShiftCS = 0;
    VectorCached<SparseMappingLayerTarget, 8> m_layerTarget;
    VectorCached<SparseMappingLayer, 8> m_layer;
    VectorCached<SparseMappingMask, 8> m_mask;
    VectorCached2D<SparseMappingLayerMask, 8, 8> m_layerMask;
};

#include "sparseClearCS.hlsl.h"
#include "sparseScaleCS.hlsl.h"
#include "sparseShiftCS.hlsl.h"

struct SparseMappingScaleShaderParams {
    NvFlowUint4 factor;
    NvFlowUint4 factorBits;
};

struct SparseMappingShiftShaderParams {
    NvFlowInt4 maskOffset;
};

uint32_t SparseMappingImpl::getNumLayers() {
    return m_numLayersTarget;
}

NvFlowDim SparseMappingImpl::getMaskDim() {
    return m_desc.maskDim;
}

SparseMappingHandle SparseMappingImpl::mapMaskScaled(NvFlowContext* context,
                                                     NvFlowDim dim) {
    SparseMappingHandle result;
    uint32_t maskIdx;

    for (maskIdx = 0; maskIdx < m_mask.size(); ++maskIdx) {
        NvFlowDim maskDim = m_mask[maskIdx].dim;

        if (maskDim == dim) break;
    }

    if (maskIdx == m_mask.size()) {
        m_mask.allocateBack();
        auto layerMaskIdx = m_layerMask.allocateBackY();
        m_mask[maskIdx].dim = dim;

        for (uint32_t layerIdx = 0; layerIdx < m_layerMask.sizeX(); ++layerIdx) {
            auto& layerMask = m_layerMask[layerMaskIdx][layerIdx];
            layerMask.mask = nullptr;
            layerMask.maskDirty = 1;
        }
    }

    result.handle = this;
    result.uid = maskIdx;
    result.numLayers = m_numLayers;

    return result;
}

SparseMappingLayerHandle SparseMappingImpl::mapMaskLayerScaled(NvFlowContext* context,
                                                               SparseMappingHandle handle,
                                                               uint32_t layerIdx) {
    SparseMappingLayerHandle result;

    if (handle.uid >= m_mask.size() || layerIdx >= m_layer.size()) {
        ZeroMemory(&result, sizeof(result));
        return result;
    }

    uint32_t maskIdx = handle.uid;
    auto& mask = m_mask[maskIdx];
    auto& rootMask = m_mask[0];
    auto& layerMask = m_layerMask[maskIdx][layerIdx];
    auto& rootLayerMask = m_layerMask[0][layerIdx];
    auto& layer = m_layer[layerIdx];
    if (!layerMask.mask) {
        layerMask.mask = createMask(context, mask.dim);
    }

    if (layerMask.maskDirty) {
        NvFlowUint3 factors;
        NvFlowUint3 factorBits;

        factors.x = rootMask.dim.x / mask.dim.x;
        factors.y = rootMask.dim.y / mask.dim.y;
        factors.z = rootMask.dim.z / mask.dim.z;
        factorBits.x = log2ui(factors.x);
        factorBits.y = log2ui(factors.y);
        factorBits.z = log2ui(factors.z);

        auto scaleParams = (SparseMappingScaleShaderParams*)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (scaleParams) {
            scaleParams->factor = make_uint4(factors, 1);
            scaleParams->factorBits = make_uint4(factorBits, 0);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDispatchParams dispatchParams = {};
        dispatchParams.shader = m_sparseScaleCS;
        dispatchParams.gridDim = (mask.dim + 7) / 8;
        dispatchParams.rootConstantBuffer = m_constantBuffer;
        dispatchParams.readOnly[0] = NvFlowTexture3DGetResource(rootLayerMask.mask);
        dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(layerMask.mask);
        NvFlowContextDispatch(context, &dispatchParams);
        layerMask.maskDirty = 0;
    }

    result.mask = layerMask.mask;
    result.dim = mask.dim;
    result.enable = layer.enable;

    return result;
}

void SparseMappingImpl::unmapMaskLayerScaled(NvFlowContext* context,
                                             SparseMappingHandle handle,
                                             uint32_t layerIdx) {}

void SparseMappingImpl::unmapMaskScaled(NvFlowContext* context) {}

SparseMappingImpl::SparseMappingImpl(NvFlowContext* context,
                                     const SparseMappingDesc* desc) {
    m_desc = *desc;

    auto createShader = [context](const BYTE* cs, uint64_t cs_length,
                                  const wchar_t* label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_sparseClearCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseClearCS));

    m_sparseScaleCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseScaleCS));

    m_sparseShiftCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseShiftCS));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = 32;
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);
    m_mask.push_back({m_desc.maskDim});

    m_layerMask.allocateBackY();

    for (int layerIdx = 0; layerIdx < m_desc.initialNumLayers; ++layerIdx) {
        addLayer();

        SparseMappingHandle result = mapMask(context);
        unmapMask(context);
    }
}

SparseMappingImpl::~SparseMappingImpl() {
    for (uint32_t maskIdx = 0; maskIdx < m_layerMask.sizeY(); ++maskIdx) {
        for (uint32_t layerIdx = 0; layerIdx < m_layerMask.sizeX(); ++layerIdx) {
            auto& layerMask = m_layerMask[maskIdx][layerIdx];
            SafeRelease(layerMask.mask);
            SafeRelease(layerMask.accumFront);
            SafeRelease(layerMask.accumBack);
        }
    }

    SafeRelease(m_constantBuffer);
    SafeRelease(m_sparseClearCS);
    SafeRelease(m_sparseScaleCS);
    SafeRelease(m_sparseShiftCS);
}

NvFlowTexture3D* SparseMappingImpl::createMask(NvFlowContext* context,
                                               const NvFlowDim& dim) {
    NvFlowTexture3DDesc maskTexDesc = {};
    maskTexDesc.format = eNvFlowFormat_r32_uint;
    maskTexDesc.dim = dim;
    maskTexDesc.uploadAccess = 0;
    maskTexDesc.downloadAccess = 0;
    auto maskTex = NvFlowCreateTexture3D(context, &maskTexDesc);
    clearMask(context, maskTex);
    return maskTex;
}

void SparseMappingImpl::clearMask(NvFlowContext* context, NvFlowTexture3D* maskTex) {
    NvFlowTexture3DDesc texDesc;
    NvFlowTexture3DGetDesc(maskTex, &texDesc);
    NvFlowDim gridDim = (texDesc.dim + 7) / 8;
    NvFlowDispatchParams dispatchParams = {};
    dispatchParams.shader = m_sparseClearCS;
    dispatchParams.gridDim = gridDim;
    dispatchParams.rootConstantBuffer = nullptr;
    dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(maskTex);
    NvFlowContextDispatch(context, &dispatchParams);
}

void SparseMappingImpl::shiftMask(NvFlowContext* context, NvFlowTexture3D* destTex,
                                  NvFlowTexture3D* srcTex, const NvFlowInt3& offset) {
    NvFlowTexture3DDesc texDstDesc;
    NvFlowTexture3DGetDesc(destTex, &texDstDesc);

    auto shaderParams =
        (SparseMappingShiftShaderParams*)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (shaderParams) {
        shaderParams->maskOffset = make_int4(offset, 0);
    }
    NvFlowConstantBufferUnmap(context, m_constantBuffer);

    NvFlowDispatchParams dispatchParams = {};
    dispatchParams.shader = m_sparseShiftCS;
    dispatchParams.gridDim = (texDstDesc.dim + 7) >> 3;
    dispatchParams.rootConstantBuffer = m_constantBuffer;
    dispatchParams.readOnly[0] = NvFlowTexture3DGetResource(srcTex);
    dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(destTex);
    NvFlowContextDispatch(context, &dispatchParams);
}

void SparseMappingImpl::syncLayers(NvFlowContext* context) {
    while (m_layer.size() < m_numLayersTarget)
        addLayer(context);

    m_numLayers = m_numLayersTarget;

    for (uint32_t layerIdx = 0; layerIdx < m_layer.size(); ++layerIdx) {
        auto& layer = m_layer[layerIdx];
        auto& layerTarget = m_layerTarget[layerIdx];
        if (!layerTarget.enableTarget && layer.enable) {
            auto& rootLayerMask = m_layerMask[0][layerIdx];
            clearMask(context, rootLayerMask.mask);

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

SparseMappingHandle SparseMappingImpl::mapAccum(NvFlowContext* context) {
    ++m_mapAccumVersion;
    syncLayers(context);

    SparseMappingHandle result;
    result.handle = this;
    result.uid = m_mapAccumVersion;
    result.numLayers = m_layer.size();
    return result;
}

SparseMappingLayerHandle SparseMappingImpl::mapAccumLayer(SparseMappingHandle handle,
                                                          uint32_t layerIdx) {
    SparseMappingLayerHandle result;

    if (handle.uid == m_mapAccumVersion && layerIdx < m_layer.size()) {
        auto& layer = m_layer[layerIdx];

        auto& rootLayerMask = m_layerMask[0][layerIdx];
        result.mask = rootLayerMask.accumFront;
        result.dim = m_mask[0].dim;
        result.enable = layer.enable;
    } else
        ZeroMemory(&result, sizeof(result));

    return result;
}

void SparseMappingImpl::unmapAccumLayer(SparseMappingHandle handle, uint32_t layerIdx) {}

SparseMappingLayerHandle SparseMappingImpl::mapAccumBackLayer(SparseMappingHandle handle,
                                                              uint32_t layerIdx) {
    SparseMappingLayerHandle result;

    if (handle.uid == m_mapAccumVersion && layerIdx < m_layer.size()) {
        auto& layer = m_layer[layerIdx];

        auto& rootLayerMask = m_layerMask[0][layerIdx];
        result.mask = rootLayerMask.accumBack;
        result.dim = m_mask[0].dim;
        result.enable = layer.enable;
    } else
        ZeroMemory(&result, sizeof(result));

    return result;
}

void SparseMappingImpl::unmapAccumBackLayer(SparseMappingHandle handle, uint32_t layerIdx) {
}

void SparseMappingImpl::unmapAccum(NvFlowContext* context) {
    ++m_mapAccumVersion;
}

void SparseMappingImpl::swapAccum(NvFlowContext* context) {
    auto mapped = mapAccum(context);

    for (uint32_t layerIdx = 0; layerIdx < m_layer.size(); ++layerIdx) {
        auto& rootLayerMask = m_layerMask[0][layerIdx];
        swap(rootLayerMask.accumBack, rootLayerMask.accumBack);
    }

    unmapAccum(context);
}

void SparseMappingImpl::clearAccum(NvFlowContext* context) {
    SparseMappingHandle mapped;

    mapped = mapAccum(context);

    for (uint32_t layerIdx = 0; layerIdx < mapped.numLayers; ++layerIdx) {
        SparseMappingLayerHandle mappedLayer = mapAccumLayer(mapped, layerIdx);

        if (mappedLayer.enable) {
            clearMask(context, mappedLayer.mask);
        }

        unmapAccumLayer(mapped, layerIdx);
    }

    unmapAccum(context);
}

void SparseMappingImpl::shiftAccum(NvFlowContext* context, const NvFlowInt3& offset) {
    if (offset.x || offset.y || offset.z) {
        auto mapped = mapAccum(context);

        for (uint32_t layerIdx = 0; layerIdx < m_layer.size(); ++layerIdx) {
            auto mappedFrontLayer = mapAccumLayer(mapped, layerIdx);
            auto mappedBackLayer = mapAccumBackLayer(mapped, layerIdx);

            if (mappedFrontLayer.enable) {
                shiftMask(context, mappedBackLayer.mask, mappedFrontLayer.mask, offset);
            }

            unmapAccumBackLayer(mapped, layerIdx);
            unmapAccumLayer(mapped, layerIdx);
        }
    }

    unmapAccum(context);
    swapAccum(context);
}

SparseMappingHandle SparseMappingImpl::mapMask(NvFlowContext* context) {
    ++m_mapMaskVersion;
    syncLayers(context);
    SparseMappingHandle result;
    result.handle = this;
    result.uid = m_mapMaskVersion;
    result.numLayers = m_layer.size();
    return result;
}

SparseMappingLayerHandle SparseMappingImpl::mapMaskLayer(SparseMappingHandle handle,
                                                         uint32_t layerIdx) {
    SparseMappingLayerHandle result;

    if (handle.uid == m_mapMaskVersion && layerIdx < m_layer.size()) {
        auto& layer = m_layer[layerIdx];
        auto& rootLayerMask = m_layerMask[0][layerIdx];

        result.mask = rootLayerMask.mask;
        result.dim = m_mask[0].dim;
        result.enable = layer.enable;
    } else
        ZeroMemory(&result, sizeof(SparseMappingLayerHandle));

    return result;
}

void SparseMappingImpl::unmapMaskLayer(SparseMappingHandle handle, uint32_t layerIdx) {
    if (handle.uid == m_mapMaskVersion && layerIdx < m_layerTarget.size()) {
        auto& layer = m_layer[layerIdx];
        if (layer.enable) {
            for (uint32_t maskIdx = 1; maskIdx < m_layerMask.sizeY(); ++maskIdx) {
                auto& layerMask = m_layerMask[maskIdx][layerIdx];
                layerMask.maskDirty = 1;
            }
        }
    }
}

void SparseMappingImpl::unmapMask(NvFlowContext* context) {
    ++m_mapMaskVersion;
}

void SparseMappingImpl::addLayer() {
    ++m_numLayersTarget;
    while (m_layerTarget.size() < m_numLayersTarget) {
        uint32_t layerIdx = m_layerTarget.allocateBack();
        auto& layerTarget = m_layerTarget[layerIdx];
        layerTarget = SparseMappingLayerTarget{};
    }
}

void SparseMappingImpl::addLayer(NvFlowContext* context) {
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

    rootLayerMask.mask = createMask(context, Mask.dim);
    rootLayerMask.accumFront = createMask(context, Mask.dim);
    rootLayerMask.accumBack = createMask(context, Mask.dim);
}

void SparseMappingImpl::enableLayer(uint32_t layerIdx) {
    if (layerIdx < m_layerTarget.size()) m_layerTarget[layerIdx].enableTarget = 1;
}

void SparseMappingImpl::disableLayer(uint32_t layerIdx) {
    if (layerIdx < m_layerTarget.size()) m_layerTarget[layerIdx].enableTarget = 0;
}

SparseMapping* createSparseMapping(NvFlowContext* context, const SparseMappingDesc* desc) {
    return new SparseMappingImpl(context, desc);
}

SparseMappingLayerHandle SparseMappingHandle::mapAccumLayer(uint32_t layerIdx) {
    return handle->mapAccumLayer(*this, layerIdx);
}

SparseMappingLayerHandle SparseMappingHandle::mapMaskLayer(uint32_t layerIdx) {
    return handle->mapMaskLayer(*this, layerIdx);
}

void SparseMappingHandle::unmapAccumLayer(uint32_t layerIdx) {
    handle->unmapAccumLayer(*this, layerIdx);
}

void SparseMappingHandle::unmapMaskLayer(uint32_t layerIdx) {
    handle->unmapMaskLayer(*this, layerIdx);
}

}  // namespace NvFlow