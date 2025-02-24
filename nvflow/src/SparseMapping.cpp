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

SparseMappingLayerHandle* SparseMappingImpl::mapMAskLayerScaled(
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
}

SparseMappingImpl::~SparseMappingImpl() {}

uint64_t SparseMappingImpl::getGPUBytesUsed() {
    return 0;
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

void SparseMappingImpl::clearAccum(NvFlowContext* ctx) {}

void SparseMappingImpl::shiftAccum(NvFlowContext* ctx) {}

SparseMappingHandle* SparseMappingImpl::mapMask(SparseMappingHandle* result,
                                                NvFlowContext* ctx) {
    return nullptr;
}

SparseMappingLayerHandle* SparseMappingImpl::mapMaskLayer(SparseMappingLayerHandle* result,
                                                          SparseMappingHandle mapping,
                                                          uint32_t idx) {
    return nullptr;
}

void SparseMappingImpl::unmapMaskLayer(SparseMappingHandle mapping, uint32_t idx) {}

void SparseMappingImpl::unmapMask(NvFlowContext* ctx) {}

void SparseMappingImpl::addLayer() {}

void SparseMappingImpl::enableLayer(uint32_t idx) {}

void SparseMappingImpl::disableLayer(uint32_t idx) {}

}  // namespace NvFlow