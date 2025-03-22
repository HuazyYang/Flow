#include "CrossSection.h"
#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ClientHelper.h"
#include "RenderMaterialPool.h"

namespace NvFlow {

struct CrossSection : Object, NvFlowCrossSection {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void render(NvFlowContext *context, const NvFlowCrossSectionParams *params) override;

    // Details
    CrossSection(NvFlowContext *context, const NvFlowCrossSectionDesc *desc);
    ~CrossSection();

    NvFlowGraphicsShader *m_crossSectionShader;
    NvFlowGraphicsShader *m_crossSectionVectorShader;
    NvFlowVertexBuffer *m_vertexBuffer;
    NvFlowIndexBuffer *m_indexBuffer;
    NvFlowConstantBuffer *m_constantBuffer;
};

struct CrossSectionShaderParams {
#include "crossSectionShaderParams.h"
};

uint64_t CrossSection::getGPUBytesUsed() {
    return 0;
}

void CrossSection::render(NvFlowContext *context, const NvFlowCrossSectionParams *params) {
    auto rtv = params->renderTargetView;
    auto dsv = params->depthStencilView;
    auto rt = NvFlowRenderTargetViewGetRenderTarget(rtv);
    auto ds = NvFlowDepthStencilViewGetDepthStencil(dsv);
    NvFlowRenderTargetDesc rt_desc;
    NvFlowDepthStencilDesc ds_desc;
    NvFlowRenderTargetGetDesc(rt, &rt_desc);
    NvFlowDepthStencilGetDesc(ds, &ds_desc);
    NvFlowViewport viewport = {};
    if (rt) {
        viewport = rt_desc.viewport;
        if (!params->fullscreen) {
            viewport.topLeftX = viewport.width / 2.f;
            viewport.width = (viewport.width + 1.f) / 2.f;
            viewport.height = (viewport.height + 1.f) / 2.f;
        }
    }
    NvFlowFloat4 pixelSize =
        make_float4(2.f / viewport.width, 2.f / viewport.height, 1.f, 1.f);

    NvFlowGridExportHandle exportHandle;
    if (!params->gridExportDebugVis ||
        params->renderChannel == eNvFlowGridTextureChannelDensity) {
        exportHandle =
            NvFlowGridExportGetHandle(params->gridExport, context, params->renderChannel);
    } else {
        exportHandle = NvFlowGridExportGetHandle(params->gridExportDebugVis, context,
                                                 params->renderChannel);
    }

    if (exportHandle.numLayerViews) {
        NvFlowGridExportLayeredView layeredView = {};
        NvFlowGridExportGetLayeredView(exportHandle, &layeredView);
        NvFlowGridExportLayerView layerView = {};
        NvFlowGridExportGetLayerView(exportHandle, 0, &layerView);
        auto matHandle =
            params->materialPool->getRenderMaterialHandle(layerView.mapping.material);
        auto material = params->materialPool->getMaterialParams(matHandle);
        auto colorMap = params->materialPool->getColorMap(matHandle);

        NvFlowFloat3 crossSectionScale = make_float3(1.f / params->crossSectionScale);
        float aspectRatio = rt_desc.viewport.width / rt_desc.viewport.height;
        auto mapped =
            (CrossSectionShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            mapped->valueParams = layeredView.mapping.shaderParams;
            NvFlowFloat4 posScale;
            posScale.x = 1.f;
            posScale.y = aspectRatio;
            posScale.z = 0.f;
            posScale.w = 0.f;
            mapped->posScale = posScale;
            mapped->uvScale = make_float4(0.5f);
            mapped->crossSectionPosition = make_float4(params->crossSectionPosition, 0.f);
            mapped->crossSectionScale = make_float4(crossSectionScale, 0.f);
            mapped->crossSectionAxis = params->crossSectionAxis;
            mapped->renderMode = params->renderMode;
            mapped->alphaBias_layer0 = material->alphaBias;
            mapped->intensityBias_layer0 = material->intensityBias;
            mapped->colorMapCompMask_layer0 = material->colorMapCompMask;
            mapped->colorMapRange_layer0 =
                make_float4(material->colorMapMinX,
                            1.f / (material->colorMapMaxX - material->colorMapMinX),
                            material->colorMapMinX, material->colorMapMaxX);
            mapped->alphaCompMask_layer0 = material->alphaCompMask;
            mapped->intensityCompMask_layer0 = material->intensityCompMask;
            mapped->intensityScale = params->intensityScale;
            mapped->pointFilter = params->pointFilter;
            mapped->velocityScale = params->velocityScale;
            mapped->vectorLengthScale = 0.002f * params->vectorLengthScale;
            mapped->lineColor = params->lineColor;
            mapped->backgroundColor = params->backgroundColor;
            mapped->pixelSize = pixelSize;
            mapped->cellColor = params->cellColor;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowContextSetRenderTarget(context, rt, nullptr);
        if (!params->fullscreen) NvFlowContextSetViewport(context, &viewport);

        NvFlowDrawParams drawParams = {};
        drawParams.shader = m_crossSectionShader;
        NvFlowGraphicsShaderSetFormats(context, drawParams.shader, rt_desc.rt_format,
                                       ds_desc.ds_format);
        drawParams.rootConstantBuffer = m_constantBuffer;
        drawParams.ps_readOnly[0] = layerView.data;
        drawParams.ps_readOnly[1] = layerView.mapping.blockTable;
        drawParams.ps_readOnly[2] = colorMap;
        NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);
        NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
        NvFlowContextSetRenderTarget(context, rt, ds);
    }

    if (params->velocityVectors) {
        if (params->gridExportDebugVis) {
            auto velocityHandle = NvFlowGridExportGetHandle(
                params->gridExportDebugVis, context, eNvFlowGridTextureChannelVelocity);
            if (velocityHandle.numLayerViews) {
                NvFlowGridExportLayeredView layeredView = {};
                NvFlowGridExportGetLayeredView(velocityHandle, &layeredView);
                NvFlowGridExportLayerView layerView = {};
                NvFlowGridExportGetLayerView(velocityHandle, 0, &layerView);
                auto matHandle = params->materialPool->getRenderMaterialHandle(
                    layerView.mapping.material);
                auto material = params->materialPool->getMaterialParams(matHandle);
                auto colorMap = params->materialPool->getColorMap(matHandle);

                NvFlowFloat3 crossSectionScale =
                    make_float3(1.f / params->crossSectionScale);
                float aspectRatio = rt_desc.viewport.width / rt_desc.viewport.height;
                auto mapped = (CrossSectionShaderParams *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                if (mapped) {
                    mapped->valueParams = layeredView.mapping.shaderParams;
                    NvFlowFloat4 posScale;
                    posScale.x = 1.f;
                    posScale.y = aspectRatio;
                    posScale.z = 0.f;
                    posScale.w = 0.f;
                    mapped->posScale = posScale;
                    mapped->uvScale = make_float4(0.5f);
                    mapped->crossSectionPosition =
                        make_float4(params->crossSectionPosition, 0.f);
                    NvFlowFloat3 crossSectionScale =
                        make_float3(1.f / params->crossSectionScale);
                    mapped->crossSectionScale = make_float4(crossSectionScale, 0.f);
                    mapped->crossSectionAxis = params->crossSectionAxis;
                    mapped->renderMode = params->renderMode;
                    mapped->alphaBias_layer0 = material->alphaBias;
                    mapped->intensityBias_layer0 = material->intensityBias;
                    mapped->colorMapCompMask_layer0 = material->colorMapCompMask;
                    mapped->colorMapRange_layer0 =
                        make_float4(material->colorMapMinX,
                                    1.f / (material->colorMapMaxX - material->colorMapMinX),
                                    material->colorMapMinX, material->colorMapMaxX);
                    mapped->alphaCompMask_layer0 = material->alphaCompMask;
                    mapped->intensityCompMask_layer0 = material->intensityCompMask;
                    mapped->intensityScale = params->intensityScale;
                    mapped->pointFilter = params->pointFilter;
                    mapped->velocityScale = params->velocityScale;
                    mapped->vectorLengthScale = 0.002f * params->vectorLengthScale;
                    mapped->lineColor = params->lineColor;
                    mapped->backgroundColor = params->backgroundColor;
                    mapped->pixelSize = pixelSize;
                    mapped->cellColor = params->cellColor;
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                NvFlowContextSetRenderTarget(context, rt, nullptr);
                if (!params->fullscreen) NvFlowContextSetViewport(context, &viewport);

                NvFlowDrawParams drawParams = {};
                drawParams.shader = m_crossSectionVectorShader;
                NvFlowGraphicsShaderSetFormats(context, drawParams.shader,
                                               rt_desc.rt_format, ds_desc.ds_format);
                drawParams.rootConstantBuffer = m_constantBuffer;
                drawParams.vs_readOnly[0] = layerView.data;
                drawParams.vs_readOnly[1] = layerView.mapping.blockTable;
                NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4),
                                             0);
                NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);

                uint32_t maxLinesX = layeredView.mapping.shaderParams.vdim.x *
                                     layeredView.mapping.shaderParams.vdim.y;
                uint32_t maxLinesY = layeredView.mapping.shaderParams.vdim.y *
                                     layeredView.mapping.shaderParams.vdim.z;
                uint32_t maxLinesZ = layeredView.mapping.shaderParams.vdim.z *
                                     layeredView.mapping.shaderParams.vdim.x;

                uint32_t numInstances = max3(maxLinesX, maxLinesY, maxLinesZ);
                NvFlowContextDrawIndexedInstanced(context, 2, numInstances, &drawParams);

                if (params->outlineCells) {
                    NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0x18u);
                    NvFlowContextDrawIndexedInstanced(context, 8, numInstances,
                                                      &drawParams);
                }

                NvFlowContextSetRenderTarget(context, rt, ds);
            }
        }
    }
}

#include "crossSectionVS.hlsl.h"
#include "crossSectionPS.hlsl.h"
#include "crossSectionVectorVS.hlsl.h"
#include "crossSectionVectorPS.hlsl.h"

CrossSection::CrossSection(NvFlowContext *context, const NvFlowCrossSectionDesc *desc)
    : m_crossSectionShader{0},
      m_crossSectionVectorShader{0},
      m_vertexBuffer{0},
      m_indexBuffer{0},
      m_constantBuffer{0} {
    NvFlowInputElementDesc elementDescs[1];
    elementDescs[0].semanticName = "POSITION";
    elementDescs[0].format = eNvFlowFormat_r32g32b32a32_float;
    NvFlowGraphicsShaderDesc shaderDesc = {};
    shaderDesc.vs = g_crossSectionVS;
    shaderDesc.vs_length = sizeof(g_crossSectionVS);
    shaderDesc.ps = g_crossSectionPS;
    shaderDesc.ps_length = sizeof(g_crossSectionPS);
    shaderDesc.label = L"crossSectionShader";
    shaderDesc.numInputElements = 1;
    shaderDesc.inputElementDescs = elementDescs;
    shaderDesc.blendState.enable = 0;
    shaderDesc.blendState.srcBlendColor = eNvFlowBlend_One;
    shaderDesc.blendState.dstBlendColor = eNvFlowBlend_SrcAlpha;
    shaderDesc.blendState.blendOpColor = eNvFlowBlendOp_Add;
    shaderDesc.blendState.srcBlendAlpha = eNvFlowBlend_One;
    shaderDesc.blendState.dstBlendAlpha = eNvFlowBlend_SrcAlpha;
    shaderDesc.blendState.blendOpAlpha = eNvFlowBlendOp_Add;
    shaderDesc.depthState.depthEnable = 0;
    shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
    shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
    shaderDesc.numRenderTargets = 1;
    shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r16g16b16a16_float;
    shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
    shaderDesc.uavTarget = 0;
    shaderDesc.depthClipEnable = 1;
    m_crossSectionShader = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.vs = g_crossSectionVectorVS;
    shaderDesc.vs_length = sizeof(g_crossSectionVectorVS);
    shaderDesc.ps = g_crossSectionVectorPS;
    shaderDesc.ps_length = sizeof(g_crossSectionVectorPS);
    shaderDesc.label = L"crossSectionVectorShader";
    shaderDesc.lineList = 1;
    m_crossSectionVectorShader = NvFlowCreateGraphicsShader(context, &shaderDesc);

    float pts[] = {
        -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, 1.f,  -1.f, 1.f,  1.f, 1.f,
        1.f,  -1.f, 1.f,  -1.f, 1.f, -1.f, -1.f, -1.f, -1.f, 1.f, -1.f,
        1.f,  -1.f, 1.f,  1.f,  1.f, 1.f,  -1.f, 1.f,  -1.f, 1.f,
    };
    NvFlowUint indices[] = {
        0, 3, 1, 1, 3, 2, 4, 5, 5, 6, 6, 7, 7, 4,
    };

    NvFlowVertexBufferDesc vbufDesc = {};
    vbufDesc.data = pts;
    vbufDesc.sizeInBytes = sizeof(pts);
    m_vertexBuffer = NvFlowCreateVertexBuffer(context, &vbufDesc);

    NvFlowIndexBufferDesc ibufDesc = {};
    ibufDesc.data = indices;
    ibufDesc.sizeInBytes = sizeof(indices);
    ibufDesc.format = eNvFlowFormat_r32_uint;
    m_indexBuffer = NvFlowCreateIndexBuffer(context, &ibufDesc);

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(CrossSectionShaderParams);
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);
}

CrossSection::~CrossSection() {
    SafeRelease(m_crossSectionShader);
    SafeRelease(m_crossSectionVectorShader);
    SafeRelease(m_vertexBuffer);
    SafeRelease(m_indexBuffer);
    SafeRelease(m_constantBuffer);
}

NvFlowCrossSection *FlowCreateCrossSection(NvFlowContext *context,
                                           const NvFlowCrossSectionDesc *desc) {
    return new CrossSection(context, desc);
}

}  // namespace NvFlow