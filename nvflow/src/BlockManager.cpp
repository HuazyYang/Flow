#include "BlockManager.h"
#include "ClientHelper.h"

namespace NvFlow {

struct BlockManagerImpl : Object, BlockManager {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    SparseMappingHandle map(NvFlowContext *context) override;
    void unmap(NvFlowContext *context) override;
    NvFlowDim getDim() override;
    SparseMapping *getSparseMapping() override;
    SparseFadeField *getFadeField() override;
    NvFlowResult commit(SparseTextureFront *velocity, SparseTextureFront *density,
                        NvFlowContext *context, SparseMappable *const *fields,
                        uint32_t numFields) override;
    bool updateLocation(NvFlowFloat3 *newGridLocation,
                        NvFlowFloat3 *oldGridLocation) override;
    void update(NvFlowContext *context, SparseTextureFront *velocity,
                SparseTextureFront *density, SparseTextureFront *coarseDensity,
                SparseMappable *const *fields, uint32_t numFields,
                const BlockManagerParams *params) override;

    // Details
    BlockManagerImpl(NvFlowContext *context, const BlockManagerDesc *desc);

    ~BlockManagerImpl();

    static NvFlowResource *getFadeFieldFunc(void *userdata, bool isVelocityField,
                                            uint32_t layerIdx);

    NvFlowResult updateMappings(NvFlowContext *context, SparseTextureFront *velocity,
                                SparseTextureFront *density, SparseMappable *const *fields,
                                uint32_t numFields, uint32_t fieldUpdateRate);

    void syncFieldFadeLayers(NvFlowContext *context, uint32_t numFieldFadeLayers);

    void syncLayers(NvFlowContext *context, uint32_t numLayers);

    void updateFieldFade(NvFlowContext *context, SparseTextureFront *velocity,
                         SparseTextureFront *density);

    struct PerLayer {
        NvFlowTexture3D *m_velocitySummaryTex;
        NvFlowTexture3D *m_densitySummaryTex;
    };

    struct FieldFadePerLayer {
        NvFlowTexture3D *m_fieldFadeVelocity;
        NvFlowTexture3D *m_fieldFadeDensity;
    };

    BlockManagerDesc m_desc;
    NvFlowUint3 m_velDownsample;
    SparseMapping *m_fieldMapping;
    NvFlowComputeShader *m_velocitySummaryCS;
    NvFlowComputeShader *m_densitySummaryCS;
    NvFlowComputeShader *m_densitySummaryCoarseCS;
    NvFlowComputeShader *m_blockManager1CS;
    NvFlowComputeShader *m_blockManager2CS;
    NvFlowComputeShader *m_blockManager2CS_big;
    NvFlowComputeShader *m_clearTexture3dCS_r;
    NvFlowComputeShader *m_clearTexture3dCS_rgba;
    NvFlowComputeShader *m_sparseFadeCS;
    NvFlowConstantBuffer *m_constantBuffer;
    SparseFadeField m_fadeField;
    uint32_t m_state;
    bool m_gridLocationDirty;
    NvFlowFloat3 m_gridHalfSize;
    NvFlowFloat3 m_gridLocation;
    NvFlowFloat3 m_oldGridLocation;
    NvFlowInt3 m_mapIdxReadOffset;
    uint64_t m_mappingVersion;
    VectorCached<BlockManagerImpl::PerLayer, 16> m_layers;
    VectorCached<BlockManagerImpl::FieldFadePerLayer, 16> m_fieldFadeLayers;
};

struct BlockManagerShaderParams {
#include <blockManagerShaderParams.h>
};

struct SparseFadeShaderParams {
    NvFlowUint4 fadeFieldDim;
    NvFlowFloat4 fadeFieldDimInv;
    NvFlowFloat4 blockTableVelocityDim;
    NvFlowFloat4 blockTableDensityDim;
};

struct VelocitySummaryShaderParams {
    NvFlowShaderLinearParams velocityParams;
};

struct DensitySummaryShaderParams {
    NvFlowShaderLinearParams velocityParams;
};

#include "velocitySummaryCS.hlsl.h"
#include "densitySummaryCS.hlsl.h"
#include "densitySummaryCoarseCS.hlsl.h"
#include "blockManager1CS.hlsl.h"
#include "blockManager2CS.hlsl.h"
#include "blockManager2CS_big.hlsl.h"
#include "clearTexture3dCS_r.hlsl.h"
#include "clearTexture3dCS_rgba.hlsl.h"
#include "sparseFadeCS.hlsl.h"

BlockManagerImpl::BlockManagerImpl(NvFlowContext *context, const BlockManagerDesc *desc)
    : m_fieldMapping(0),
      m_velocitySummaryCS(0),
      m_densitySummaryCS(0),
      m_densitySummaryCoarseCS(0),
      m_blockManager1CS(0),
      m_blockManager2CS(0),
      m_blockManager2CS_big(0),
      m_clearTexture3dCS_r(0),
      m_clearTexture3dCS_rgba(0),
      m_sparseFadeCS(0),
      m_constantBuffer(0),
      m_fadeField{},
      m_state(1),
      m_gridLocationDirty(0),
      m_gridHalfSize{1.f, 1.f, 1.f},
      m_gridLocation{0.f, 0.f, 0.f},
      m_oldGridLocation{0.f, 0.f, 0.f},
      m_mapIdxReadOffset{0, 0, 0},
      m_mappingVersion{1} {
    m_desc = *desc;
    m_fadeField.userData = this;
    m_fadeField.getFadeField = getFadeFieldFunc;

    m_velDownsample = make_uint3(m_desc.densityVirtualDim / m_desc.velocityVirtualDim);

    SparseMappingDesc mappingDesc = {};
    mappingDesc.maskDim = desc->densityVirtualDim / 16;
    mappingDesc.initialNumLayers = 1;
    m_fieldMapping = createSparseMapping(context, &mappingDesc);

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_velocitySummaryCS = createShader(NVFLOW_CREATE_SHADER_ARGS(velocitySummaryCS));
    m_densitySummaryCS = createShader(NVFLOW_CREATE_SHADER_ARGS(densitySummaryCS));
    m_densitySummaryCoarseCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(densitySummaryCoarseCS));
    m_blockManager1CS = createShader(NVFLOW_CREATE_SHADER_ARGS(blockManager1CS));
    m_blockManager2CS = createShader(NVFLOW_CREATE_SHADER_ARGS(blockManager2CS));
    m_blockManager2CS_big = createShader(NVFLOW_CREATE_SHADER_ARGS(blockManager2CS_big));
    m_clearTexture3dCS_r = createShader(NVFLOW_CREATE_SHADER_ARGS(clearTexture3dCS_r));
    m_clearTexture3dCS_rgba =
        createShader(NVFLOW_CREATE_SHADER_ARGS(clearTexture3dCS_rgba));
    m_sparseFadeCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseFadeCS));

    uint32_t maxSize =
        max<uint32_t>(sizeof(BlockManagerShaderParams), sizeof(NvFlowShaderLinearParams));
    maxSize = max<uint32_t>(maxSize, sizeof(SparseFadeShaderParams));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = maxSize;
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);

    m_fieldMapping->clearAccum(context);
}

BlockManagerImpl::~BlockManagerImpl() {
    for (auto &perLayer : m_layers) {
        SafeRelease(perLayer.m_velocitySummaryTex);
        SafeRelease(perLayer.m_densitySummaryTex);
    }

    for (auto &fieldFadePerLayer : m_fieldFadeLayers) {
        SafeRelease(fieldFadePerLayer.m_fieldFadeVelocity);
        SafeRelease(fieldFadePerLayer.m_fieldFadeDensity);
    }

    SafeRelease(m_fieldMapping);

    SafeRelease(m_velocitySummaryCS);
    SafeRelease(m_densitySummaryCS);
    SafeRelease(m_densitySummaryCoarseCS);
    SafeRelease(m_blockManager1CS);
    SafeRelease(m_blockManager2CS);
    SafeRelease(m_blockManager2CS_big);
    SafeRelease(m_clearTexture3dCS_r);
    SafeRelease(m_clearTexture3dCS_rgba);
    SafeRelease(m_sparseFadeCS);
    SafeRelease(m_constantBuffer);
}

NvFlowResource *BlockManagerImpl::getFadeFieldFunc(void *userdata, bool isVelocityField,
                                                   uint32_t layerIdx) {
    auto blockManager = (BlockManagerImpl *)(userdata);

    if (layerIdx >= blockManager->m_fieldFadeLayers.size())
        return nullptr;

    auto &fieldFadeLayer = blockManager->m_fieldFadeLayers[layerIdx];
    auto tex = isVelocityField ? fieldFadeLayer.m_fieldFadeVelocity
                               : fieldFadeLayer.m_fieldFadeDensity;
    return NvFlowTexture3DGetResource(tex);
}

NvFlowResult BlockManagerImpl::updateMappings(
    NvFlowContext *context, SparseTextureFront *velocity, SparseTextureFront *density,
    SparseMappable *const *fields, uint32_t numFields, uint32_t fieldUpdateRate) {
    bool allcomplete = 1;
    for (uint32_t i = 0; i < numFields; ++i) {
        fields[i]->updateMapping(context, fieldUpdateRate);
        allcomplete &= fields[i]->canCommitMapping(context, m_mappingVersion);
    }

    if (!allcomplete)
        return eNvFlowFail;

    for (uint32_t i = 0; i < numFields; ++i)
        fields[i]->commitMapping(context, m_mappingVersion);

    updateFieldFade(context, velocity, density);

    m_fieldMapping->shiftAccum(context, m_mapIdxReadOffset);

    m_gridLocationDirty = 1;
    m_state = 1;
    return eNvFlowSuccess;
}

void BlockManagerImpl::syncFieldFadeLayers(NvFlowContext *context,
                                           uint32_t numFieldFadeLayers) {
    while (m_fieldFadeLayers.size() < numFieldFadeLayers) {
        uint32_t fieldFadeLayerIdx = m_fieldFadeLayers.allocateBack();
        auto &fieldFadePerLayer = m_fieldFadeLayers[fieldFadeLayerIdx];

        NvFlowTexture3DDesc texDesc = {};
        texDesc.format = eNvFlowFormat_r8_unorm;
        texDesc.dim = m_fieldMapping->getMaskDim();
        texDesc.uploadAccess = 0;
        texDesc.downloadAccess = 0;
        fieldFadePerLayer.m_fieldFadeVelocity = NvFlowCreateTexture3D(context, &texDesc);
        fieldFadePerLayer.m_fieldFadeDensity = NvFlowCreateTexture3D(context, &texDesc);

        NvFlowDim gridDim = (texDesc.dim + 7) >> 3;
        NvFlowDispatchParams dparams = {};
        dparams.rootConstantBuffer = 0;
        dparams.gridDim = gridDim;
        dparams.shader = m_clearTexture3dCS_r;
        dparams.readWrite[0] =
            NvFlowTexture3DGetResourceRW(fieldFadePerLayer.m_fieldFadeVelocity);
        NvFlowContextDispatch(context, &dparams);
        dparams.readWrite[0] =
            NvFlowTexture3DGetResourceRW(fieldFadePerLayer.m_fieldFadeDensity);
        NvFlowContextDispatch(context, &dparams);
    }
}

void BlockManagerImpl::syncLayers(NvFlowContext *context, uint32_t numLayers) {
    while (m_layers.size() < numLayers) {
        uint32_t layerIdx = m_layers.allocateBack();
        auto &perLayer = m_layers[layerIdx];

        NvFlowTexture3DDesc texDesc = {};
        texDesc.format = eNvFlowFormat_r32_float;
        texDesc.dim = m_fieldMapping->getMaskDim();
        texDesc.uploadAccess = 0;
        texDesc.downloadAccess = 0;
        perLayer.m_velocitySummaryTex = NvFlowCreateTexture3D(context, &texDesc);
        texDesc.format = eNvFlowFormat_r32g32b32a32_float;
        perLayer.m_densitySummaryTex = NvFlowCreateTexture3D(context, &texDesc);

        NvFlowDim gridDim = (texDesc.dim + 7) >> 3;
        NvFlowDispatchParams dparams = {};
        dparams.rootConstantBuffer = 0;
        dparams.gridDim = gridDim;
        dparams.shader = m_clearTexture3dCS_r;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(perLayer.m_velocitySummaryTex);
        NvFlowContextDispatch(context, &dparams);
        dparams.shader = m_clearTexture3dCS_rgba;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(perLayer.m_densitySummaryTex);
        NvFlowContextDispatch(context, &dparams);
    }
}

void BlockManagerImpl::updateFieldFade(NvFlowContext *context, SparseTextureFront *velocity,
                                       SparseTextureFront *density) {
    auto velocityTex = velocity->acquireTexture(context);
    auto densityTex = density->acquireTexture(context);
    auto velocityWriteHandle = velocityTex.writeLinearHandle(context);
    auto densityWriteHandle = densityTex.writeLinearHandle(context);
    auto velocityLayeredView = velocityWriteHandle.layeredView();
    auto densityLayeredView = densityWriteHandle.layeredView();

    uint32_t numLayers = densityWriteHandle.numLayers;
    syncFieldFadeLayers(context, densityWriteHandle.numLayers);

    auto fieldFadeDim = m_fieldMapping->getMaskDim();
    for (uint32_t layerIdx = 0; layerIdx < m_fieldFadeLayers.size(); ++layerIdx) {
        auto &fieldFadeLayer = m_fieldFadeLayers[layerIdx];

        auto velocityLayerView = velocityWriteHandle.layerView(layerIdx);
        auto densityLayerView = densityWriteHandle.layerView(layerIdx);
        auto mapped =
            (SparseFadeShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            mapped->fadeFieldDim = make_uint4(fieldFadeDim, 1);
            mapped->fadeFieldDimInv = 1.f / make_float4(fieldFadeDim, 1);
            mapped->blockTableVelocityDim = make_float4(velocityLayeredView.params.gridDim);
            mapped->blockTableDensityDim = make_float4(densityLayeredView.params.gridDim);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_sparseFadeCS;
        dparams.gridDim = (fieldFadeDim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = velocityLayerView.mapping.blockTable;
        dparams.readOnly[1] = densityLayerView.mapping.blockTable;
        dparams.readWrite[0] =
            NvFlowTexture3DGetResourceRW(fieldFadeLayer.m_fieldFadeVelocity);
        dparams.readWrite[1] =
            NvFlowTexture3DGetResourceRW(fieldFadeLayer.m_fieldFadeDensity);
        NvFlowContextDispatch(context, &dparams);
    }
    velocityTex.releaseTexture();
    densityTex.releaseTexture();
}

uint64_t BlockManagerImpl::getGPUBytesUsed() {
    uint64_t totalBytes = 0;
    if (m_fieldMapping)
        totalBytes += m_fieldMapping->getGPUBytesUsed();

    for (auto &perLayer : m_layers) {
        if (perLayer.m_velocitySummaryTex)
            totalBytes += perLayer.m_velocitySummaryTex->getGPUBytesUsed();
        if (perLayer.m_densitySummaryTex)
            totalBytes += perLayer.m_densitySummaryTex->getGPUBytesUsed();
    }

    return totalBytes;
}

SparseMappingHandle BlockManagerImpl::map(NvFlowContext *ctx) {
    return m_fieldMapping->mapAccum(ctx);
}

void BlockManagerImpl::unmap(NvFlowContext *ctx) {
    m_fieldMapping->unmapAccum(ctx);
}

NvFlowDim BlockManagerImpl::getDim() {
    return m_fieldMapping->getMaskDim();
}

SparseMapping *BlockManagerImpl::getSparseMapping() {
    return m_fieldMapping;
}

SparseFadeField *BlockManagerImpl::getFadeField() {
    return &m_fadeField;
}

NvFlowResult BlockManagerImpl::commit(SparseTextureFront *velocity,
                                      SparseTextureFront *density, NvFlowContext *context,
                                      SparseMappable *const *fields, uint32_t numFields) {
    if (!m_desc.lowLatencyMapping || m_state)
        return eNvFlowFail;
    else
        return updateMappings(context, velocity, density, fields, numFields, 8);
}

bool BlockManagerImpl::updateLocation(NvFlowFloat3 *newGridLocation,
                                      NvFlowFloat3 *oldGridLocation) {
    if (!m_gridLocationDirty)
        return false;
    m_gridLocationDirty = 1;
    if (newGridLocation)
        *newGridLocation = m_gridLocation;
    if (oldGridLocation)
        *oldGridLocation = m_oldGridLocation;
    return true;
}

void BlockManagerImpl::update(NvFlowContext *context, SparseTextureFront *velocity,
                              SparseTextureFront *density,
                              SparseTextureFront *coarseDensity,
                              SparseMappable *const *fields, uint32_t numFields,
                              const BlockManagerParams *params) {
    if (m_state) {
        if (m_state == 1) {
            NvFlowDim maskDim = getDim();
            NvFlowDim snapGridDim = maskDim >> 2;
            NvFlowFloat3 snapCellSize =
                (2.f * params->gridHalfSize) / (make_float3(maskDim >> 2) + 0.f);
            NvFlowFloat3 snapTargetLocation;
            snapTargetLocation.x =
                (float)(int)(params->gridTargetLocation.x / snapCellSize.x) *
                snapCellSize.x;
            snapTargetLocation.y =
                (float)(int)(params->gridTargetLocation.y / snapCellSize.y) *
                snapCellSize.y;
            snapTargetLocation.z =
                (float)(int)(params->gridTargetLocation.z / snapCellSize.z) *
                snapCellSize.z;
            m_oldGridLocation = m_gridLocation;
            m_gridLocation = snapTargetLocation;
            m_gridHalfSize = params->gridHalfSize;

            NvFlowFloat3 locationOffset = m_gridLocation - m_oldGridLocation;

            m_mapIdxReadOffset =
                make_int3(make_float3(maskDim) * locationOffset / (2.f * m_gridHalfSize));

            SparseReadLinearHandle velocityHandle =
                velocity->front.readLinearHandle(context);
            SparseReadLinearLayeredView velocityLayeredView = velocityHandle.layeredView();
            syncLayers(context, velocityHandle.numLayers);

            for (uint32_t layerIdx = 0; layerIdx < velocityHandle.numLayers; ++layerIdx) {
                auto &perLayer = m_layers[layerIdx];
                auto velocityLayerView = velocityHandle.layerView(layerIdx);
                auto mapped = NvFlowConstantBufferMap(context, m_constantBuffer);
                if (mapped) {
                    CopyMemory(mapped, &velocityLayeredView.params,
                               sizeof(VelocitySummaryShaderParams));
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                params->reportVelocityLayerNumBlocks(
                    params->userdata, velocityLayerView.mapping.numBlocks, layerIdx);

                NvFlowDim gridDim;

                gridDim.x = (((velocityLayeredView.params.blockDim.x *
                               velocityLayerView.mapping.numBlocks) >>
                              1) +
                             7) >>
                            3,
                gridDim.y = (velocityLayeredView.params.blockDim.y + 7) >> 3;
                gridDim.z = (velocityLayeredView.params.blockDim.z + 7) >> 3;

                NvFlowDispatchParams dparams = {};
                dparams.shader = m_velocitySummaryCS;
                dparams.gridDim = gridDim;
                dparams.rootConstantBuffer = m_constantBuffer;
                dparams.readOnly[0] = velocityLayerView.mapping.blockList;
                dparams.readOnly[1] = velocityLayerView.data;
                dparams.readOnly[2] = velocityLayerView.mapping.blockTable;
                dparams.readWrite[0] =
                    NvFlowTexture3DGetResourceRW(perLayer.m_velocitySummaryTex);
                NvFlowContextDispatch(context, &dparams);
            }

            bool useCoarse = m_desc.velocityVirtualDim != m_desc.densityVirtualDim;
            SparseReadLinearHandle densityHandle;
            if (useCoarse)
                densityHandle = coarseDensity->front.readLinearHandle(context);
            else
                densityHandle = density->front.readLinearHandle(context);
            auto densityLayeredView = densityHandle.layeredView();
            syncLayers(context, densityHandle.numLayers);

            for (uint32_t layerIdx = 0; layerIdx < densityHandle.numLayers; ++layerIdx) {
                auto &perLayer = m_layers[layerIdx];
                auto densityLayerView = densityHandle.layerView(layerIdx);
                auto mapped = NvFlowConstantBufferMap(context, m_constantBuffer);
                if (mapped) {
                    CopyMemory(mapped, &densityLayeredView.params,
                               sizeof(DensitySummaryShaderParams));
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                params->reportDensityLayerNumBlocks(
                    params->userdata, densityLayerView.mapping.numBlocks, layerIdx);

                NvFlowDim gridDim;

                gridDim.x = (((densityLayeredView.params.blockDim.x *
                               densityLayerView.mapping.numBlocks) >>
                              1) +
                             7) >>
                            3,
                gridDim.y = (densityLayeredView.params.blockDim.y + 7) >> 3;
                gridDim.z = (densityLayeredView.params.blockDim.z + 7) >> 3;

                NvFlowDispatchParams dparams = {};
                dparams.shader = useCoarse ? m_densitySummaryCoarseCS : m_densitySummaryCS;
                dparams.gridDim = gridDim;
                dparams.rootConstantBuffer = m_constantBuffer;
                dparams.readOnly[0] = densityLayerView.mapping.blockList;
                dparams.readOnly[1] = densityLayerView.data;
                dparams.readOnly[2] = densityLayerView.mapping.blockTable;
                dparams.readWrite[0] =
                    NvFlowTexture3DGetResourceRW(perLayer.m_densitySummaryTex);
                NvFlowContextDispatch(context, &dparams);
            }

            auto dim = m_fieldMapping->getMaskDim();
            NvFlowFloat3 velocityScale3 = make_float3(dim) *
                                          float(params->bigEffectPredictTime) /
                                          (2.f * params->gridHalfSize);
            float velocityScale =
                max3(velocityScale3.x, velocityScale3.y, velocityScale3.z);

            SparseMappingHandle userMappingLayered = m_fieldMapping->mapAccum(context);
            SparseMappingHandle fieldMappingLayered = m_fieldMapping->mapMask(context);
            syncLayers(context, fieldMappingLayered.numLayers);

            for (uint32_t j = 0; j < fieldMappingLayered.numLayers; ++j) {
                BlockManagerPerLayerParams layerParams;
                params->getPerLayer(&layerParams, params->userdata, j);

                auto &perLayer = m_layers[j];
                auto userMappingHandle = userMappingLayered.mapAccumLayer(j);
                auto fieldMappingHandle = fieldMappingLayered.mapMaskLayer(j);

                if (userMappingHandle.enable && fieldMappingHandle.enable) {
                    auto userMapping = userMappingHandle.mask;
                    auto fieldMapping = fieldMappingHandle.mask;

                    auto sp = (BlockManagerShaderParams *)NvFlowConstantBufferMap(
                        context, m_constantBuffer);
                    if (sp) {
                        sp->velFactor = make_uint4(m_velDownsample, 1);
                        sp->velFactorBits.x = log2ui(m_velDownsample.x);
                        sp->velFactorBits.y = log2ui(m_velDownsample.y);
                        sp->velFactorBits.z = log2ui(m_velDownsample.z);
                        sp->velFactorBits.w = 0;
                        sp->denFactor = make_uint4(1);
                        sp->denFactorBits = make_uint4(0);
                        sp->velocityWeight = layerParams.velocityWeight;
                        sp->smokeWeight = layerParams.smokeWeight;
                        sp->tempWeight = layerParams.tempWeight;
                        sp->fuelWeight = layerParams.fuelWeight;
                        sp->velocityThreshold = layerParams.velocityThreshold;
                        sp->smokeThreshold = layerParams.smokeThreshold;
                        sp->tempThreshold = layerParams.tempThreshold;
                        sp->fuelThreshold = layerParams.fuelThreshold;
                        sp->importanceThreshold = 0.f;
                        sp->velocityScale = velocityScale;
                        sp->pad1 = 0;
                        sp->pad2 = 0;
                        sp->mapIdxReadOffset = make_int4(m_mapIdxReadOffset, 0);

                        NvFlowConstantBufferUnmap(context, m_constantBuffer);
                    }

                    NvFlowDispatchParams dparams = {};
                    dparams.gridDim = (dim + 7) >> 3;
                    dparams.shader = m_blockManager1CS;
                    dparams.rootConstantBuffer = m_constantBuffer;
                    dparams.readOnly[0] = NvFlowTexture3DGetResource(fieldMapping);
                    dparams.readOnly[1] =
                        NvFlowTexture3DGetResource(perLayer.m_velocitySummaryTex);
                    dparams.readOnly[2] =
                        NvFlowTexture3DGetResource(perLayer.m_densitySummaryTex);
                    dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(userMapping);
                    NvFlowContextDispatch(context, &dparams);

                    dparams.shader =
                        params->bigEffectMode ? m_blockManager2CS_big : m_blockManager2CS;
                    dparams.readOnly[0] = NvFlowTexture3DGetResource(userMapping);
                    dparams.readOnly[1] =
                        NvFlowTexture3DGetResource(perLayer.m_velocitySummaryTex);
                    dparams.readOnly[2] = 0;
                    dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(fieldMapping);
                    NvFlowContextDispatch(context, &dparams);
                }

                userMappingLayered.unmapAccumLayer(j);
                fieldMappingLayered.unmapMaskLayer(j);
            }

            m_fieldMapping->unmapAccum(context);
            m_fieldMapping->unmapMask(context);
            m_fieldMapping->clearAccum(context);

            ++m_mappingVersion;

            for (uint32_t i = 0; i < numFields; ++i)
                fields[i]->pushMapping(context, m_mappingVersion, m_fieldMapping);

            params->reportSummaryUpdate(params->userdata);
            m_state = 0;
        }
    } else {
        updateMappings(context, velocity, density, fields, numFields, 1);
    }

    if (m_desc.lowLatencyMapping) {
        if (!m_state)
            updateMappings(context, velocity, density, fields, numFields, 8);
    }
}

BlockManager *createBlockManager(NvFlowContext *context, const BlockManagerDesc *desc) {
    return new BlockManagerImpl(context, desc);
}

}  // namespace NvFlow