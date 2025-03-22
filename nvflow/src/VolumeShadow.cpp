#include "VolumeShadow.h"
#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ClientHelper.h"
#include "GridExport.h"
#include "GridImport.h"
#include "VolumeRenderUtils.h"
#include "RenderMaterialPool.h"

namespace NvFlow {

struct VolumeShadow : Object, NvFlowVolumeShadow {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void update(NvFlowContext *context, NvFlowGridExport *gridExport,
                const NvFlowVolumeShadowParams *params) override;
    NvFlowGridExport *getGridExport(NvFlowContext *context) override;
    void debugRender(NvFlowContext *context,
                     const NvFlowVolumeShadowDebugRenderParams *params) override;
    void getStats(NvFlowVolumeShadowStats *stats) override;

    // Details
    void allocatePool(NvFlowContext *context, float residentScale);

    void releasePool();

    VolumeShadow(NvFlowContext *context, const NvFlowVolumeShadowDesc *desc);
    ~VolumeShadow();

    NvFlowVolumeShadowDesc m_desc;
    NvFlowDim m_gridDim;
    NvFlowDim m_poolGridDim;
    NvFlowDim m_poolDim;
    unsigned int m_maxBlocks;
    unsigned int m_residentBlocks;
    float m_currentResidentScale;
    NvFlowFloat4x4 m_shadowProjection;
    NvFlowFloat4x4 m_shadowView;
    bool m_atomicReadbackActive;
    unsigned int m_shadowColumnsActive;
    unsigned int m_shadowBlocksActive;
    NvFlowGraphicsShader *m_volumeShadowDebug;
    NvFlowComputeShader *m_volumeShadowClearAllocMaskCS;
    NvFlowComputeShader *m_volumeShadowGenAllocMaskCS;
    NvFlowComputeShader *m_volumeShadowGenListCS;
    NvFlowComputeShader *m_volumeShadowStaticCS;
    NvFlowComputeShader *m_volumeShadowApplyCS;
    NvFlowComputeShader *m_volumeShadowApplyCS_SST;
    NvFlowComputeShader *m_volumeShadowApplyCS_VTR;
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowVertexBuffer *m_vertexBuffer;
    NvFlowIndexBuffer *m_indexBuffer;
    NvFlowTexture3D *m_shadow3DTex;
    NvFlowTexture3D *m_allocMaskTex;
    NvFlowTexture3D *m_shadowBlockTable;
    NvFlowBuffer *m_shadowBlockList;
    NvFlowBuffer *m_atomicBuf;
    NvFlowConstantBuffer *m_atomicConst;
    NvFlowGridImport *m_gridImport;
};

struct VolumeShadowAllocShaderParams {
    NvFlowFloat4x4 vidxNormToShadow;
    NvFlowFloat4x4 shadowToVidxNorm;
    NvFlowFloat4 linearDepthTransform;
    NvFlowUint4 shadowVolumeDim;
    NvFlowFloat4 shadowVolumeDimInv;
    NvFlowUint4 shadowBlockDimBits;
    NvFlowUint4 numBlocks;
    NvFlowFloat4 gridDimInv;
    NvFlowShaderLinearParams exportParams;
};

struct VolumeShadowApplyShaderParams {
#include "volumeShadowShaderParams.h"
};

static constexpr NvFlowUint4 shadowBlockDim = {16, 16, 16, 1};
static constexpr NvFlowUint4 shadowBlockDimBits = {4, 4, 4, 0};
static constexpr NvFlowFloat4 shadowBlockDimInv = {0.0625f, 0.0625f, 0.0625f, 1.f};
static constexpr NvFlowFloat4 shadowCellIdxInflate = {1.0666667, 1.0666667, 1.0666667, 1.0};
static constexpr NvFlowFloat4 shadowCellIdxInflateInv = {0.9375, 0.9375, 0.9375, 1.0};
static constexpr NvFlowFloat4 shadowCellIdxOffset = {0.5, 0.5, 0.5, 1.0};

uint64_t VolumeShadow::getGPUBytesUsed() {
    return 0;
}

void VolumeShadow::update(NvFlowContext *context, NvFlowGridExport *gridExport,
                          const NvFlowVolumeShadowParams *params) {
    auto renderMode = params->renderMode;
    m_shadowProjection = params->projectionMatrix;
    m_shadowView = params->viewMatrix;

    NvFlowContextProfileGroupBegin(context, L"VolumeShadowUpdate");
    auto channel = eNvFlowGridTextureChannelDensity;
    auto exportHandle = NvFlowGridExportGetHandle(gridExport, context, channel);

    NvFlowGridImportParams importParams;
    importParams.gridExport = gridExport;
    importParams.channel = channel;
    importParams.importMode = eNvFlowGridImportModeLinear;
    auto importHandle = NvFlowGridImportGetHandle(m_gridImport, context, &importParams);

    NvFlowGridExportLayeredView exportLayeredView = {};
    NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

    NvFlowGridImportLayeredView importLayeredView = {};
    NvFlowGridImportGetLayeredView(importHandle, &importLayeredView);

    NvFlowGridExportLayerView exportLayerView = {};
    NvFlowGridImportLayerView importLayerView = {};

    for (uint32_t j = 0; j < importHandle.numLayerViews; ++j) {
        NvFlowGridImportGetLayerView(importHandle, j, &importLayerView);
        NvFlowGridExportGetLayerView(exportHandle, j, &exportLayerView);

        auto matHandle =
            params->materialPool->getRenderMaterialHandle(exportLayerView.mapping.material);
        auto material = params->materialPool->getMaterialParams(matHandle);
        auto colorMap = params->materialPool->getColorMap(matHandle);

        auto modelMat = exportLayeredView.mapping.modelMatrix;
        auto viewMat = params->viewMatrix;
        auto projMat = params->projectionMatrix;
        auto modelView = modelMat * viewMat;
        auto modeViewInv = inverse(modelView);
        auto modelViewProj = modelView * projMat;
        auto modelViewProjT = transpose(modelViewProj);
        auto projInv = inverse(projMat);
        auto modelViewProjInv = inverse(modelViewProj);
        auto modelViewProjInvT = transpose(modelViewProjInv);

        NvFlowFloat4 linearDepthTransform;
        VolumeRenderUtils::compute_linearDepthTransform(&linearDepthTransform, projMat);

        if (m_desc.maxResidentScale > m_desc.minResidentScale &&
            m_shadowBlocksActive > m_residentBlocks) {
            releasePool();
            float residentScale = 2.f * m_currentResidentScale;
            if (residentScale > m_desc.maxResidentScale)
                residentScale = m_desc.maxResidentScale;
            allocatePool(context, residentScale);
        }

        NvFlowFloat4 vGridDimInv;
        (NvFlowFloat3 &)vGridDimInv =
            1.f / make_float3(
                      (const NvFlowUint3 &)exportLayeredView.mapping.shaderParams.gridDim);
        vGridDimInv.w = 1.f;

        auto mapped = (VolumeShadowAllocShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        mapped->vidxNormToShadow = modelViewProjT;
        mapped->shadowToVidxNorm = modelViewProjInvT;
        mapped->linearDepthTransform = linearDepthTransform;
        mapped->shadowVolumeDim =
            make_uint4(m_desc.mapWidth, m_desc.mapHeight, m_desc.mapDepth, 1);
        mapped->shadowVolumeDimInv =
            1.f / make_float4(m_desc.mapWidth, m_desc.mapHeight, m_desc.mapDepth, 1);
        mapped->shadowBlockDimBits = shadowBlockDimBits;
        mapped->numBlocks = make_uint4(exportLayerView.mapping.numBlocks, 0, 0, 0);
        mapped->gridDimInv = vGridDimInv;
        mapped->exportParams = exportLayeredView.mapping.shaderParams;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_volumeShadowClearAllocMaskCS;
        dparams.gridDim = (m_gridDim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_allocMaskTex);
        NvFlowContextDispatch(context, &dparams);

        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_volumeShadowGenAllocMaskCS;
        dparams.gridDim.x = (exportLayerView.mapping.numBlocks + 127) / 128;
        dparams.gridDim.y = 1;
        dparams.gridDim.z = 1;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = exportLayerView.mapping.blockList;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_allocMaskTex);
        NvFlowContextDispatch(context, &dparams);

        auto atomicData = (NvFlowUint *)NvFlowBufferMap(context, m_atomicBuf);
        if (atomicData) {
            NvFlowBufferDesc desc;
            NvFlowBufferGetDesc(m_atomicBuf, &desc);
            memset(atomicData, 0, sizeof(NvFlowUint) * desc.dim);
            NvFlowBufferUnmap(context, m_atomicBuf);
        }

        struct ShaderParams {
            NvFlowUint4 gridDim;
            NvFlowUint4 poolGridDim;
        };

        auto mapped2 = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        mapped2->gridDim = make_uint4(m_gridDim, 1);
        mapped2->poolGridDim = make_uint4(m_poolGridDim, 1);
        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_volumeShadowGenListCS;
        dparams.gridDim.x = (m_gridDim.x + 7) >> 3;
        dparams.gridDim.y = (m_gridDim.y + 7) >> 3;
        dparams.gridDim.z = 1;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_allocMaskTex);
        dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_shadowBlockList);
        dparams.readWrite[1] = NvFlowBufferGetResourceRW(m_atomicBuf);
        dparams.readWrite[2] = NvFlowTexture3DGetResourceRW(m_shadowBlockTable);
        NvFlowContextDispatch(context, &dparams);

        if (m_atomicReadbackActive) {
            auto atomicData = (NvFlowUint *)NvFlowBufferMapDownload(context, m_atomicBuf);
            if (atomicData) {
                m_shadowColumnsActive = atomicData[0];
                m_shadowBlocksActive = atomicData[1];
                NvFlowBufferUnmapDownload(context, m_atomicBuf);
                m_atomicReadbackActive = 0;
            }
        } else {
            NvFlowBufferDownload(context, m_atomicBuf);
            m_atomicReadbackActive = 1;
        }

        NvFlowContextCopyConstantBuffer(context, m_atomicConst, m_atomicBuf);

        auto mapped3 = (VolumeShadowApplyShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        mapped3->vidxNormToShadow = modelViewProjT;
        mapped3->shadowToVidxNorm = modelViewProjInvT;
        mapped3->linearDepthTransform = linearDepthTransform;
        mapped3->shadowVdim =
            make_uint4(m_desc.mapWidth, m_desc.mapHeight, m_desc.mapDepth, 1);
        mapped3->shadowVdimInv =
            1.f / make_float4(m_desc.mapWidth, m_desc.mapHeight, m_desc.mapDepth, 1);
        mapped3->shadowRdim = make_uint4(m_poolDim, 1);
        mapped3->shadowRdimInv = 1.f / make_float4(m_poolDim, 1);
        mapped3->shadowBlockDim = shadowBlockDim;
        mapped3->shadowBlockDimBits = shadowBlockDimBits;
        mapped3->shadowBlockDimInv = shadowBlockDimInv;
        mapped3->shadowCellIdxInflate = shadowCellIdxInflate;
        mapped3->shadowCellIdxInflateInv = shadowCellIdxInflateInv;
        mapped3->shadowCellIdxOffset = shadowCellIdxOffset;

        mapped3->alphaScale = material->alphaScale * params->intensityScale;
        mapped3->minIntensity = params->minIntensity;
        mapped3->shadowBlendBias = params->shadowBlendBias;
        mapped3->pad3 = 0.f;
        mapped3->renderMode = make_uint4(renderMode);
        mapped3->alphaBias_layer0 = material->alphaBias;
        mapped3->intensityBias_layer0 = material->intensityBias;
        mapped3->pad4 = 0.f;
        mapped3->pad5 = 0.f;
        mapped3->colorMapCompMask_layer0 = material->colorMapCompMask;
        mapped3->alphaCompMask_layer0 = material->alphaCompMask;
        mapped3->intensityCompMask_layer0 = material->intensityCompMask;
        mapped3->shadowBlendCompMask = params->shadowBlendCompMask;

        mapped3->colorMapRange_layer0 = make_float4(
            material->colorMapMinX, 1.f / (material->colorMapMaxX - material->colorMapMinX),
            material->colorMapMinX, material->colorMapMaxX);

        mapped3->exportParams = exportLayeredView.mapping.shaderParams;
        mapped3->importParams = importLayeredView.mapping.shaderParams;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_volumeShadowStaticCS;
        dparams.gridDim.x = (shadowBlockDim.x * m_residentBlocks + 7) >> 3;
        dparams.gridDim.y = (shadowBlockDim.y + 7) >> 3;
        dparams.gridDim.z = 1;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.secondConstantBuffer = m_atomicConst;
        dparams.readOnly[0] = exportLayerView.mapping.blockList;
        dparams.readOnly[1] = exportLayerView.mapping.blockTable;
        dparams.readOnly[2] = exportLayerView.data;
        dparams.readOnly[3] = colorMap;
        dparams.readOnly[4] = NvFlowBufferGetResource(m_shadowBlockList);
        dparams.readOnly[5] = NvFlowTexture3DGetResource(m_shadowBlockTable);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_shadow3DTex);
        NvFlowContextDispatch(context, &dparams);

        bool enableVTR = importLayeredView.mapping.shaderParams.isVTR.x;
        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_volumeShadowApplyCS;
        if (enableVTR) {
            dparams.shader = m_volumeShadowApplyCS_VTR;
            dparams.gridDim.x = (importLayeredView.mapping.shaderParams.blockDim.x *
                                     importLayerView.mapping.numBlocks +
                                 7) >>
                                3;
            dparams.gridDim.y =
                (importLayeredView.mapping.shaderParams.blockDim.y + 7) >> 8;
            dparams.gridDim.z =
                (importLayeredView.mapping.shaderParams.blockDim.z + 7) >> 3;
        } else {
            dparams.shader = m_volumeShadowApplyCS_SST;
            dparams.gridDim.x =
                (importLayeredView.mapping.shaderParams.linearBlockDim.z *
                     importLayeredView.mapping.shaderParams.linearBlockDim.y *
                     importLayeredView.mapping.shaderParams.linearBlockDim.x +
                 127) /
                128;
            dparams.gridDim.y = importLayerView.mapping.numBlocks;
            dparams.gridDim.z = 1;
        }
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = exportLayerView.mapping.blockList;
        dparams.readOnly[1] = exportLayerView.mapping.blockTable;
        dparams.readOnly[2] = exportLayerView.data;
        dparams.readOnly[3] = importLayerView.mapping.blockList;
        dparams.readOnly[4] = importLayerView.mapping.blockTable;
        dparams.readOnly[5] = NvFlowTexture3DGetResource(m_shadow3DTex);
        dparams.readOnly[6] = NvFlowTexture3DGetResource(m_shadowBlockTable);
        dparams.readWrite[0] = importLayerView.dataRW;
        NvFlowContextDispatch(context, &dparams);
    }

    NvFlowGridImportGetGridExport(m_gridImport, context);
    NvFlowContextProfileGroupEnd(context);
}

NvFlowGridExport *VolumeShadow::getGridExport(NvFlowContext *context) {
    return NvFlowGridImportGetGridExport(m_gridImport, context);
}

void VolumeShadow::debugRender(NvFlowContext *context,
                               const NvFlowVolumeShadowDebugRenderParams *params) {
    auto rtv = params->renderTargetView;
    uint32_t totalBlocks = m_maxBlocks;
    NvFlowFloat4x4 shadowProj = m_shadowProjection;
    NvFlowFloat4x4 shadowView = m_shadowView;
    NvFlowFloat4x4 proj = params->projectionMatrix;
    NvFlowFloat4x4 view = params->viewMatrix;

    NvFlowFloat4 linearDepthTransform;
    VolumeRenderUtils::compute_linearDepthTransform(&linearDepthTransform, shadowProj);

    NvFlowFloat4x4 viewProj = view * proj;
    NvFlowFloat4x4 shadowViewProj = shadowView * shadowProj;
    NvFlowFloat4x4 shadowViewProjInv = inverse(shadowViewProj);
    NvFlowFloat4x4 shadowProjToMainProj = shadowViewProjInv * viewProj;
    NvFlowFloat4x4 shadowProjToMainRrojT = transpose(shadowProjToMainProj);
    NvFlowFloat4x4 modelViewProjT = shadowProjToMainRrojT;

    NvFlowUint4 vGridDim = make_uint4(m_gridDim, 1);
    NvFlowFloat4 vGridDimInv = 1.f / make_float4(vGridDim);

    struct ShaderParams {
        NvFlowFloat4x4 modelViewProj;
        NvFlowFloat4 vGridDimInv;
        NvFlowUint4 vGridDim;
        NvFlowFloat4 linearDepthTransform;
    };

    auto mapped = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    mapped->modelViewProj = modelViewProjT;
    mapped->vGridDimInv = vGridDimInv;
    mapped->vGridDim = vGridDim;
    mapped->linearDepthTransform = linearDepthTransform;
    NvFlowConstantBufferUnmap(context, m_constantBuffer);

    auto RenderTarget = NvFlowRenderTargetViewGetRenderTarget(rtv);
    NvFlowRenderTargetDesc rtv_desc;
    NvFlowRenderTargetGetDesc(RenderTarget, &rtv_desc);
    NvFlowGraphicsShaderSetFormats(context, m_volumeShadowDebug, rtv_desc.rt_format,
                                   eNvFlowFormat_d32_float);

    NvFlowDrawParams drawParams = {};
    drawParams.shader = m_volumeShadowDebug;
    drawParams.rootConstantBuffer = m_constantBuffer;
    drawParams.vs_readOnly[0] = NvFlowTexture3DGetResource(m_shadowBlockTable);
    NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
    NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0x90u);
    NvFlowContextDrawIndexedInstanced(context, 0x18u, totalBlocks + 1, &drawParams);
}

void VolumeShadow::getStats(NvFlowVolumeShadowStats *stats) {
    if (stats) {
        stats->shadowColumnsActive = m_shadowColumnsActive;
        stats->shadowBlocksActive = m_shadowBlocksActive;
        stats->shadowCellsActive = (shadowBlockDim.z - 1) * (shadowBlockDim.y - 1) *
                                   (shadowBlockDim.x - 1) * m_shadowBlocksActive;
    }
}

void VolumeShadow::allocatePool(NvFlowContext *context, float residentScale) {
    m_residentBlocks = int(residentScale * m_maxBlocks);
    float rgridDimf = pow(float(m_residentBlocks), 1.f / 3.f);
    m_poolGridDim.z = ceil(rgridDimf);
    rgridDimf = sqrt(float(m_residentBlocks) / m_poolGridDim.z);
    m_poolGridDim.y = ceil(rgridDimf);
    uint32_t yzSlice = m_poolGridDim.y * m_poolGridDim.z;
    m_poolGridDim.x = (m_residentBlocks + yzSlice - 1) / yzSlice;
    m_residentBlocks = m_poolGridDim.z * m_poolGridDim.y * m_poolGridDim.x;

    m_poolDim.x = m_poolGridDim.x * shadowBlockDim.x;
    m_poolDim.y = m_poolGridDim.y * shadowBlockDim.y;
    m_poolDim.z = m_poolGridDim.z * shadowBlockDim.z;

    NvFlowTexture3DDesc texDesc = {};
    texDesc.format = eNvFlowFormat_r16_float;
    texDesc.dim = m_poolDim;
    texDesc.uploadAccess = 0;
    texDesc.downloadAccess = 0;
    m_shadow3DTex = NvFlowCreateTexture3D(context, &texDesc);
    m_currentResidentScale = residentScale;
}

void VolumeShadow::releasePool() {
    SafeRelease(m_shadow3DTex);
}

#include "volumeShadowDebugVS.hlsl.h"
#include "volumeShadowDebugPS.hlsl.h"
#include "volumeShadowClearAllocMaskCS.hlsl.h"
#include "volumeShadowGenAllocMaskCS.hlsl.h"
#include "volumeShadowGenListCS.hlsl.h"
#include "volumeShadowStaticCS.hlsl.h"
#include "volumeShadowApplyCS.hlsl.h"
#include "volumeShadowApplyCS_SST.hlsl.h"
#include "volumeShadowApplyCS_VTR.hlsl.h"

VolumeShadow::VolumeShadow(NvFlowContext *context, const NvFlowVolumeShadowDesc *desc)
    : m_desc{},
      m_gridDim{},
      m_poolGridDim{},
      m_poolDim{},
      m_maxBlocks{0},
      m_residentBlocks{0},
      m_currentResidentScale{0.f},
      m_shadowProjection{},
      m_shadowView{},
      m_atomicReadbackActive{0},
      m_shadowColumnsActive{0},
      m_shadowBlocksActive{0},
      m_volumeShadowDebug{0},
      m_volumeShadowClearAllocMaskCS{0},
      m_volumeShadowGenAllocMaskCS{0},
      m_volumeShadowGenListCS{0},
      m_volumeShadowStaticCS{0},
      m_volumeShadowApplyCS{0},
      m_volumeShadowApplyCS_SST{0},
      m_volumeShadowApplyCS_VTR{0},
      m_constantBuffer{0},
      m_vertexBuffer{0},
      m_indexBuffer{0},
      m_shadow3DTex{0},
      m_allocMaskTex{0},
      m_shadowBlockTable{0},
      m_shadowBlockList{0},
      m_atomicBuf{0},
      m_atomicConst{0},
      m_gridImport{0} {
    m_desc = *desc;

    m_gridDim.x = (m_desc.mapWidth + shadowBlockDim.x - 1) >> shadowBlockDimBits.x;
    m_gridDim.y = (m_desc.mapHeight + shadowBlockDim.y - 1) >> shadowBlockDimBits.y;
    m_gridDim.z = (m_desc.mapDepth + shadowBlockDim.z - 1) >> shadowBlockDimBits.z;
    m_maxBlocks = m_gridDim.z * m_gridDim.y * m_gridDim.x;

    m_desc.mapWidth = m_gridDim.x << shadowBlockDimBits.x;
    m_desc.mapHeight = m_gridDim.x << shadowBlockDimBits.y;
    m_desc.mapDepth = m_gridDim.x << shadowBlockDimBits.z;

    NvFlowInputElementDesc elementDescs[1];
    elementDescs[0].semanticName = "POSITION";
    elementDescs[0].format = eNvFlowFormat_r32g32b32a32_float;

    NvFlowGraphicsShaderDesc shaderDesc = {};
    shaderDesc.numInputElements = 1;
    shaderDesc.inputElementDescs = elementDescs;
    shaderDesc.blendState.enable = 0;
    shaderDesc.depthState.depthEnable = 0;
    shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
    shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
    shaderDesc.uavTarget = 0;
    shaderDesc.depthClipEnable = 1;
    shaderDesc.numRenderTargets = 1;
    shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r8g8b8a8_unorm;
    shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
    shaderDesc.vs = g_volumeShadowDebugVS;
    shaderDesc.vs_length = sizeof(g_volumeShadowDebugVS);
    shaderDesc.ps = g_volumeShadowDebugPS;
    shaderDesc.ps_length = sizeof(g_volumeShadowDebugPS);
    shaderDesc.label = L"volumeShadowDebug";
    shaderDesc.lineList = 1;
    m_volumeShadowDebug = NvFlowCreateGraphicsShader(context, &shaderDesc);

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_volumeShadowClearAllocMaskCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowClearAllocMaskCS));
    m_volumeShadowGenAllocMaskCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowGenAllocMaskCS));
    m_volumeShadowGenListCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowGenListCS));
    m_volumeShadowStaticCS = createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowStaticCS));
    m_volumeShadowApplyCS = createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowApplyCS));
    m_volumeShadowApplyCS_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowApplyCS_SST));
    m_volumeShadowApplyCS_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(volumeShadowApplyCS_VTR));

    const uint32_t cbSizeA = sizeof(VolumeShadowApplyShaderParams);
    const uint32_t cbSizeB = sizeof(VolumeShadowAllocShaderParams);
    const uint32_t cbSize = max(cbSizeA, cbSizeB);
    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = cbSize;
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);

    NvFlowFloat4 pts[8] = {{-1.f, -1.f, -1.f, 1.f}, {1.f, -1.f, -1.f, 1.f},
                           {1.f, 1.f, -1.f, 1.f},   {-1.f, 1.f, -1.f, 1.f},
                           {-1.f, 1.f, 1.f, 1.f},   {1.f, 1.f, 1.f, 1.f},
                           {1.f, -1.f, 1.f, 1.f},   {-1.f, -1.f, 1.f, 1.f}};
    NvFlowVertexBufferDesc vbufDesc = {};
    vbufDesc.data = pts;
    vbufDesc.sizeInBytes = sizeof(pts);
    m_vertexBuffer = NvFlowCreateVertexBuffer(context, &vbufDesc);

    uint32_t indices[] = {
        0, 1, 3, 3, 1, 2, 5, 6, 4, 4, 6, 7, 1, 0, 6, 6, 0, 7, 2, 5, 3, 3,
        5, 4, 3, 4, 0, 0, 4, 7, 1, 6, 2, 2, 6, 5, 0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4, 0, 7, 1, 6, 2, 5, 3, 4, 0, 3, 1, 1, 3, 2,
    };
    NvFlowIndexBufferDesc ibufDesc = {};
    ibufDesc.format = eNvFlowFormat_r32_uint;
    ibufDesc.data = indices;
    ibufDesc.sizeInBytes = sizeof(indices);
    m_indexBuffer = NvFlowCreateIndexBuffer(context, &ibufDesc);

    allocatePool(context, m_desc.minResidentScale);

    NvFlowTexture3DDesc texDesc = {};
    texDesc.format = eNvFlowFormat_r16_float;
    texDesc.dim = m_gridDim;
    texDesc.uploadAccess = 0;
    texDesc.downloadAccess = 0;
    m_allocMaskTex = NvFlowCreateTexture3D(context, &texDesc);

    texDesc.format = eNvFlowFormat_r32_uint;
    m_shadowBlockTable = NvFlowCreateTexture3D(context, &texDesc);

    NvFlowBufferDesc bufDesc = {};
    bufDesc.dim = m_residentBlocks;
    bufDesc.format = eNvFlowFormat_r32g32_uint;
    bufDesc.uploadAccess = 0;
    bufDesc.downloadAccess = 0;
    m_shadowBlockList = NvFlowCreateBuffer(context, &bufDesc);

    NvFlowBufferDesc atomicBufDesc = {};
    atomicBufDesc.format = eNvFlowFormat_r32_uint;
    atomicBufDesc.dim = 64;
    atomicBufDesc.uploadAccess = 1;
    atomicBufDesc.downloadAccess = 1;
    m_atomicBuf = NvFlowCreateBuffer(context, &atomicBufDesc);

    NvFlowConstantBufferDesc atomicConstDesc = {};
    atomicConstDesc.sizeInBytes = 4 * atomicBufDesc.dim;
    atomicConstDesc.uploadAccess = 0;
    m_atomicConst = NvFlowCreateConstantBuffer(context, &atomicConstDesc);

    NvFlowGridImportDesc importDesc = {};
    importDesc.gridExport = m_desc.gridExport;
    m_gridImport = NvFlowCreateGridImport(context, &importDesc);
}

VolumeShadow::~VolumeShadow() {
    SafeRelease(m_volumeShadowDebug);
    SafeRelease(m_volumeShadowClearAllocMaskCS);
    SafeRelease(m_volumeShadowGenAllocMaskCS);
    SafeRelease(m_volumeShadowGenListCS);
    SafeRelease(m_volumeShadowStaticCS);
    SafeRelease(m_volumeShadowApplyCS);
    SafeRelease(m_volumeShadowApplyCS_SST);
    SafeRelease(m_volumeShadowApplyCS_VTR);
    SafeRelease(m_constantBuffer);
    SafeRelease(m_vertexBuffer);
    SafeRelease(m_indexBuffer);
    releasePool();
    SafeRelease(m_allocMaskTex);
    SafeRelease(m_shadowBlockTable);
    SafeRelease(m_shadowBlockList);
    SafeRelease(m_atomicBuf);
    SafeRelease(m_atomicConst);
    SafeRelease(m_gridImport);
}

NvFlowVolumeShadow *FlowCreateVolumeShadow(NvFlowContext *context,
                                           const NvFlowVolumeShadowDesc *desc) {
    return new VolumeShadow(context, desc);
}

}  // namespace NvFlow
