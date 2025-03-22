#include "VolumeRender.h"

namespace NvFlow {

#pragma region "VolumeRender Constructor"

#include "compositeVS.hlsl.h"
#include "compositePS.hlsl.h"
#include "compositePS_LMS.hlsl.h"
#include "compositeSmoothPS.hlsl.h"
#include "compositeSmoothPS_LMS.hlsl.h"
#include "compositeDepthEstimatePS.hlsl.h"
#include "compositeDepthEstimatePS_LMS.hlsl.h"
#include "compositeDepthDebugPS.hlsl.h"
#include "compositeDepthDebugPS_LMS.hlsl.h"
#include "volumeRenderVS.hlsl.h"
#include "volumeRenderPS.hlsl.h"
#include "volumeRenderPS_colormap.hlsl.h"
#include "volumeRenderPS_debug.hlsl.h"
#include "volumeRenderPS_raw.hlsl.h"
#include "volumeRenderVS2.hlsl.h"
#include "volumeRenderPS2.hlsl.h"
#include "volumeRenderPS2_colormap.hlsl.h"
#include "volumeRenderPS2_debug.hlsl.h"
#include "volumeRenderPS2_raw.hlsl.h"
#include "volumeRenderVS3.hlsl.h"
#include "volumeRenderPS3.hlsl.h"
#include "volumeRenderPS3_colormap.hlsl.h"
#include "volumeRenderPS3_debug.hlsl.h"
#include "volumeRenderPS3_raw.hlsl.h"
#include "volumeRenderVS4.hlsl.h"
#include "volumeRenderPS4.hlsl.h"
#include "volumeRenderPS4_colormap.hlsl.h"
#include "volumeRenderPS4_debug.hlsl.h"
#include "volumeRenderPS4_raw.hlsl.h"
#include "volumeRenderBoxVS.hlsl.h"
#include "volumeRenderBoxPS.hlsl.h"
#include "depthDownsampleVS.hlsl.h"
#include "depthDownsamplePS.hlsl.h"
#include "depthDownsamplePS_depthMask.hlsl.h"
#include "depthDownsamplePS_LMS.hlsl.h"
#include "depthDownsamplePS_LMS_depthMask.hlsl.h"
#include "volumeRenderDebugVS.hlsl.h"
#include "volumeRenderDebugPS.hlsl.h"
#include "volumeRenderDebugEmitBoundsVS.hlsl.h"
#include "volumeRenderDebugEmitBoundsPS.hlsl.h"
#include "volumeRenderDebugShapesSimpleVS.hlsl.h"
#include "volumeRenderDebugShapesSimplePS.hlsl.h"
#include "volumeRenderSortCS.hlsl.h"
#include "multiResColorUpsampleVS.hlsl.h"
#include "multiResColorUpsamplePS.hlsl.h"
#include "multiResDepthDownsampleVS.hlsl.h"
#include "multiResDepthDownsamplePS.hlsl.h"
#include "multiResDepthUpsampleVS.hlsl.h"
#include "multiResDepthUpsamplePS.hlsl.h"
#include "volumeRenderDepthVS.hlsl.h"
#include "volumeRenderDepthPS.hlsl.h"
#include "rayMarchMaskVS.hlsl.h"
#include "rayMarchMaskPS.hlsl.h"
#include "volumeRenderDepthEstimateVS.hlsl.h"
#include "volumeRenderDepthEstimatePS.hlsl.h"
#include "volumeRenderLightingCS.hlsl.h"
#include "volumeRenderLightingCS_SST.hlsl.h"
#include "volumeRenderLightingCS_VTR.hlsl.h"

VolumeRender::VolumeRender(NvFlowContext *context, const NvFlowVolumeRenderDesc *desc)
    : m_constantBuffer{0},
      m_vertexBuffer{0},
      m_indexBuffer{0},
      m_compositeVertexBufferRect{0},
      m_compositeIndexBufferRect{0},
      m_compositeVertexBufferMultiRes{0},
      m_compositeIndexBufferMultiRes{0},
      m_debugEmitBoundsBuffer{},
      m_debugSphereBuffer{},
      m_debugCapsuleBuffer{},
      m_debugBoxBuffer{},
      m_debugSimpleShapeMeshes{},
      m_compositeShader{0},
      m_compositeShader_LMS{0},
      m_compositeSmoothShader{0},
      m_compositeSmoothShader_LMS{0},
      m_compositeDepthEstimate{0, 0},
      m_compositeDepthEstimate_LMS{0, 0},
      m_compositeDepthDebug{0},
      m_compositeDepthDebug_LMS{0},
      m_depthDownsampleShader{0, 0},
      m_volumeRender{},
      m_volumeRenderDepthEstimate{0},
      m_volumeRenderBox{0},
      m_volumeRenderDebug{0},
      m_volumeRenderDebugEmitBounds{0},
      m_volumeRenderDebugShapesSimple{0},
      m_sortShader{0},
      m_multiResColorUpsampleShader{0},
      m_multiResDepthDownsampleShader{0},
      m_multiResDepthUpsampleShader{0},
      m_volumeRenderDepth{0, 0},
      m_rayMarchMask{0},
      m_depthMask{0},
      m_depthEstimate{0},
      m_depthEstimateViewport{},
      m_depthEstimateScreenPercentX{1.f},
      m_depthEstimateScreenPercentY{1.f},
      m_sort{0},
      m_sortCPU{0},
      m_blockListUpload{0},
      m_gridImport{0},
      m_lightingShader{0},
      m_lightingShader_SST{0},
      m_lightingShader_VTR{0} {
    m_desc = *desc;

    m_offscreenBuffers.allocateBack();

    uint32_t requestedCapacity = 1;
    if (m_desc.gridExport) {
        for (uint32_t channel = eNvFlowGridTextureChannelVelocity;
             channel < eNvFlowGridTextureChannelCount; ++channel) {
            NvFlowGridExportHandle handle = NvFlowGridExportGetHandle(
                m_desc.gridExport, context, (NvFlowGridTextureChannel)channel);
            NvFlowGridExportLayeredView layeredView = {};
            NvFlowGridExportGetLayeredView(handle, &layeredView);
            if (layeredView.mapping.maxBlocks > requestedCapacity)
                requestedCapacity = layeredView.mapping.maxBlocks;
        }
    }

    NvFlowConstantBufferDesc desca;
    desca.sizeInBytes = sizeof(VolumeRenderShaderParams);
    desca.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &desca);

    NvFlowFloat4 pts[] = {{-1.f, -1.f, -1.f, 1.f}, {1.f, -1.f, -1.f, 1.f},
                          {1.f, 1.f, -1.f, 1.f},   {-1.f, 1.f, -1.f, 1.f},
                          {-1.f, 1.f, 1.f, 1.f},   {1.f, 1.f, 1.f, 1.f},
                          {1.f, -1.f, 1.f, 1.f},   {-1.f, -1.f, 1.f, 1.f}};
    NvFlowVertexBufferDesc vbufDesc = {};
    vbufDesc.data = pts;
    vbufDesc.sizeInBytes = sizeof(pts);
    m_vertexBuffer = NvFlowCreateVertexBuffer(context, &vbufDesc);

    NvFlowUint indices[] = {
        0, 1, 3, 3, 1, 2, 5, 6, 4, 4, 6, 7, 1, 0, 6, 6, 0, 7, 2, 5, 3, 3,
        5, 4, 3, 4, 0, 0, 4, 7, 1, 6, 2, 2, 6, 5, 0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4, 0, 7, 1, 6, 2, 5, 3, 4, 0, 3, 1, 1, 3, 2,
    };
    NvFlowIndexBufferDesc ibufDesc = {};
    ibufDesc.format = eNvFlowFormat_r32_uint;
    ibufDesc.data = indices;
    ibufDesc.sizeInBytes = sizeof(indices);
    m_indexBuffer = NvFlowCreateIndexBuffer(context, &ibufDesc);

    NvFlowFloat4 rectPts[16] = {{-1.f, -1.f, -1.f, -1.f},
                                {1.f, -1.f, 1.f, -1.f},
                                {1.f, 1.f, 1.f, 1.f},
                                {-1.f, 1.f, -1.f, 1.f}};
    vbufDesc.data = rectPts;
    vbufDesc.sizeInBytes = sizeof(rectPts);
    m_compositeVertexBufferRect = NvFlowCreateVertexBuffer(context, &vbufDesc);
    m_compositeVertexBufferMultiRes = NvFlowCreateVertexBuffer(context, &vbufDesc);

    NvFlowUint rectIndices[54] = {0, 3, 1, 1, 3, 2};
    ibufDesc.format = eNvFlowFormat_r32_uint;
    ibufDesc.data = rectIndices;
    ibufDesc.sizeInBytes = sizeof(rectIndices);
    m_compositeIndexBufferRect = NvFlowCreateIndexBuffer(context, &ibufDesc);
    m_compositeIndexBufferMultiRes = NvFlowCreateIndexBuffer(context, &ibufDesc);

    NvFlowInputElementDesc elementDesc[1];
    elementDesc[0].format = eNvFlowFormat_r32g32b32a32_float;
    elementDesc[0].semanticName = "POSITION";

    NvFlowGraphicsShaderDesc shaderDesc = {};
    shaderDesc.vs = g_compositeVS;
    shaderDesc.vs_length = sizeof(g_compositeVS);
    shaderDesc.ps = g_compositePS;
    shaderDesc.ps_length = sizeof(g_compositePS);
    shaderDesc.label = L"compositePS";
    shaderDesc.inputElementDescs = elementDesc;
    shaderDesc.numInputElements = countof(elementDesc);
    shaderDesc.blendState.enable = 1;
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
    m_compositeShader = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_compositePS_LMS;
    shaderDesc.ps_length = sizeof(g_compositePS_LMS);
    shaderDesc.label = L"compositePS_LMS";
    m_compositeShader_LMS = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_compositeSmoothPS;
    shaderDesc.ps_length = sizeof(g_compositeSmoothPS);
    shaderDesc.label = L"compositeSmoothPS";
    m_compositeSmoothShader = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_compositeSmoothPS_LMS;
    shaderDesc.ps_length = sizeof(g_compositeSmoothPS_LMS);
    shaderDesc.label = L"compositeSmoothPS_LMS";
    m_compositeSmoothShader_LMS = NvFlowCreateGraphicsShader(context, &shaderDesc);

    for (uint32_t j = 0; j < 2; ++j) {
        shaderDesc.depthState.depthEnable = 1;
        shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
        shaderDesc.depthState.depthFunc =
            j ? eNvFlowComparison_GreaterEqual : eNvFlowComparison_LessEqual;
        shaderDesc.ps = g_compositeDepthEstimatePS;
        shaderDesc.ps_length = sizeof(g_compositeDepthEstimatePS);
        shaderDesc.label = L"compositeDepthEstimatePS";
        m_compositeDepthEstimate[j] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_compositeDepthEstimatePS_LMS;
        shaderDesc.ps_length = sizeof(g_compositeDepthEstimatePS_LMS);
        shaderDesc.label = L"compositeDepthEstimatePS_LMS";
        m_compositeDepthEstimate_LMS[j] = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    shaderDesc.depthState.depthEnable = 0;
    shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
    shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
    shaderDesc.ps = g_compositeDepthDebugPS;
    shaderDesc.ps_length = sizeof(g_compositeDepthDebugPS);
    shaderDesc.label = L"compositeDepthDebugPS";
    m_compositeDepthDebug = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_compositeDepthDebugPS_LMS;
    shaderDesc.ps_length = sizeof(g_compositeDepthDebugPS_LMS);
    shaderDesc.label = L"compositeDepthDebugPS_LMS";
    m_compositeDepthDebug_LMS = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.vs = g_volumeRenderVS;
    shaderDesc.vs_length = sizeof(g_volumeRenderVS);
    shaderDesc.ps = g_volumeRenderPS;
    shaderDesc.ps_length = sizeof(g_volumeRenderPS);
    shaderDesc.label = L"volumeRenderPS";
    shaderDesc.blendState.srcBlendColor = eNvFlowBlend_DstAlpha;
    shaderDesc.blendState.dstBlendColor = eNvFlowBlend_One;
    shaderDesc.blendState.srcBlendAlpha = eNvFlowBlend_Zero;
    shaderDesc.blendState.dstBlendAlpha = eNvFlowBlend_SrcAlpha;
    shaderDesc.numInputElements = 1;
    shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r16g16b16a16_float;
    shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
    shaderDesc.depthClipEnable = 1;

    for (uint32_t i = 0; i < 2; ++i) {
        if (i == 0) {
            shaderDesc.depthState.depthEnable = 0;
            shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
            shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
            shaderDesc.depthStencilFormat = eNvFlowFormat_unknown;
        } else {
            shaderDesc.depthState.depthEnable = 1;
            shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_Zero;
            shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
            shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        }

        shaderDesc.vs = g_volumeRenderVS;
        shaderDesc.vs_length = sizeof(g_volumeRenderVS);
        shaderDesc.ps = g_volumeRenderPS;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS);
        shaderDesc.label = L"volumeRenderPS";
        m_volumeRender[i][0][0] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS_colormap;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS_colormap);
        shaderDesc.label = L"volumeRenderPS_colormap";
        m_volumeRender[i][0][1] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS_debug;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS_debug);
        shaderDesc.label = L"volumeRenderPS_debug";
        m_volumeRender[i][0][2] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS_raw;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS_raw);
        shaderDesc.label = L"volumeRenderPS_raw";
        m_volumeRender[i][0][3] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_volumeRenderVS2;
        shaderDesc.vs_length = sizeof(g_volumeRenderVS2);
        shaderDesc.ps = g_volumeRenderPS2;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS2);
        shaderDesc.label = L"volumeRenderPS2";
        m_volumeRender[i][1][0] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS2_colormap;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS2_colormap);
        shaderDesc.label = L"volumeRenderPS2_colormap";
        m_volumeRender[i][1][1] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS2_debug;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS2_debug);
        shaderDesc.label = L"volumeRenderPS2_debug";
        m_volumeRender[i][1][2] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS2_raw;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS2_raw);
        shaderDesc.label = L"volumeRenderPS2_raw";
        m_volumeRender[i][1][3] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_volumeRenderVS3;
        shaderDesc.vs_length = sizeof(g_volumeRenderVS3);
        shaderDesc.ps = g_volumeRenderPS3;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS3);
        shaderDesc.label = L"volumeRenderPS3";
        m_volumeRender[i][2][0] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS3_colormap;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS3_colormap);
        shaderDesc.label = L"volumeRenderPS3_colormap";
        m_volumeRender[i][2][1] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS3_debug;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS3_debug);
        shaderDesc.label = L"volumeRenderPS3_debug";
        m_volumeRender[i][2][2] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS3_raw;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS3_raw);
        shaderDesc.label = L"volumeRenderPS3_raw";
        m_volumeRender[i][2][3] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_volumeRenderVS4;
        shaderDesc.vs_length = sizeof(g_volumeRenderVS4);
        shaderDesc.ps = g_volumeRenderPS4;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS4);
        shaderDesc.label = L"volumeRenderPS";
        m_volumeRender[i][3][0] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS4_colormap;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS4_colormap);
        shaderDesc.label = L"volumeRenderPS4_colormap";
        m_volumeRender[i][3][1] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS4_debug;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS4_debug);
        shaderDesc.label = L"volumeRenderPS4_debug";
        m_volumeRender[i][3][2] = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.ps = g_volumeRenderPS4_raw;
        shaderDesc.ps_length = sizeof(g_volumeRenderPS4_raw);
        shaderDesc.label = L"volumeRenderPS4_raw";
        m_volumeRender[i][3][3] = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    shaderDesc.depthState.depthEnable = 0;
    shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
    shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
    shaderDesc.depthStencilFormat = eNvFlowFormat_unknown;
    shaderDesc.vs = g_volumeRenderBoxVS;
    shaderDesc.vs_length = sizeof(g_volumeRenderBoxVS);
    shaderDesc.ps = g_volumeRenderBoxPS;
    shaderDesc.ps_length = sizeof(g_volumeRenderBoxPS);
    shaderDesc.label = L"volumeRenderBoxPS";
    shaderDesc.depthClipEnable = 1;
    m_volumeRenderBox = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.vs = g_depthDownsampleVS;
    shaderDesc.vs_length = sizeof(g_depthDownsampleVS);
    shaderDesc.blendState.enable = 0;
    shaderDesc.depthState.depthEnable = 0;
    shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
    shaderDesc.numRenderTargets = 1;
    shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r32_float;
    shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
    shaderDesc.depthClipEnable = 1;

    shaderDesc.ps = g_depthDownsamplePS;
    shaderDesc.ps_length = sizeof(g_depthDownsamplePS);
    shaderDesc.label = L"depthDownsamplePS";
    m_depthDownsampleShader[0][0] = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_depthDownsamplePS_depthMask;
    shaderDesc.ps_length = sizeof(g_depthDownsamplePS_depthMask);
    shaderDesc.label = L"depthDownsamplePS_depthMask";
    m_depthDownsampleShader[0][1] = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_depthDownsamplePS_LMS;
    shaderDesc.ps_length = sizeof(g_depthDownsamplePS_LMS);
    shaderDesc.label = L"depthDownsamplePS_LMS";
    m_depthDownsampleShader[1][0] = NvFlowCreateGraphicsShader(context, &shaderDesc);

    shaderDesc.ps = g_depthDownsamplePS_LMS_depthMask;
    shaderDesc.ps_length = sizeof(g_depthDownsamplePS_LMS_depthMask);
    shaderDesc.label = L"depthDownsamplePS_LMS_depthMask";
    m_depthDownsampleShader[1][1] = NvFlowCreateGraphicsShader(context, &shaderDesc);

    {
        NvFlowGraphicsShaderDesc shaderDesc = {};
        shaderDesc.vs = g_volumeRenderDebugVS;
        shaderDesc.vs_length = sizeof(g_volumeRenderDebugVS);
        shaderDesc.ps = g_volumeRenderDebugPS;
        shaderDesc.ps_length = sizeof(g_volumeRenderDebugPS);
        shaderDesc.label = L"volumeRenderDebugPS";
        shaderDesc.numInputElements = countof(elementDesc);
        shaderDesc.inputElementDescs = elementDesc;
        shaderDesc.blendState.enable = 0;
        shaderDesc.depthState.depthEnable = 0;
        shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
        shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
        shaderDesc.numRenderTargets = 1;
        shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r8g8b8a8_unorm;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        shaderDesc.uavTarget = 0;
        shaderDesc.depthClipEnable = 1;
        shaderDesc.lineList = 1;
        m_volumeRenderDebug = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_volumeRenderDebugEmitBoundsVS;
        shaderDesc.vs_length = sizeof(g_volumeRenderDebugEmitBoundsVS);
        shaderDesc.ps = g_volumeRenderDebugEmitBoundsPS;
        shaderDesc.ps_length = sizeof(g_volumeRenderDebugEmitBoundsPS);
        shaderDesc.label = L"volumeRenderDebugEmitBoundsPS";
        m_volumeRenderDebugEmitBounds = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_volumeRenderDebugShapesSimpleVS;
        shaderDesc.vs_length = sizeof(g_volumeRenderDebugShapesSimpleVS);
        shaderDesc.ps = g_volumeRenderDebugShapesSimplePS;
        shaderDesc.ps_length = sizeof(g_volumeRenderDebugShapesSimplePS);
        shaderDesc.label = L"volumeRenderDebugShapesSimplePS";
        m_volumeRenderDebugShapesSimple = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    NvFlowComputeShaderDesc csDesc;
    csDesc.cs = g_volumeRenderSortCS;
    csDesc.cs_length = sizeof(g_volumeRenderSortCS);
    csDesc.label = L"volumeRenderSortCS";
    m_sortShader = NvFlowCreateComputeShader(context, &csDesc);

    {
        NvFlowGraphicsShaderDesc shaderDesc = {};
        shaderDesc.numInputElements = countof(elementDesc);
        shaderDesc.inputElementDescs = elementDesc;
        shaderDesc.blendState.enable = 0;
        shaderDesc.depthState.depthEnable = 0;
        shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
        shaderDesc.uavTarget = 0;
        shaderDesc.depthClipEnable = 1;
        shaderDesc.vs = g_multiResColorUpsampleVS;
        shaderDesc.vs_length = sizeof(g_multiResColorUpsampleVS);
        shaderDesc.ps = g_multiResColorUpsamplePS;
        shaderDesc.ps_length = sizeof(g_multiResColorUpsamplePS);
        shaderDesc.label = L"multiResColorUpsamplePS";
        shaderDesc.numRenderTargets = 1;
        shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r16g16b16a16_float;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        m_multiResColorUpsampleShader = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_multiResDepthDownsampleVS;
        shaderDesc.vs_length = sizeof(g_multiResDepthDownsampleVS);
        shaderDesc.ps = g_multiResDepthDownsamplePS;
        shaderDesc.ps_length = sizeof(g_multiResDepthDownsamplePS);
        shaderDesc.label = L"multiResDepthDownsamplePS";
        shaderDesc.numRenderTargets = 1;
        shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r32_float;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        m_multiResDepthDownsampleShader = NvFlowCreateGraphicsShader(context, &shaderDesc);

        shaderDesc.vs = g_multiResDepthUpsampleVS;
        shaderDesc.vs_length = sizeof(g_multiResDepthUpsampleVS);
        shaderDesc.ps = g_multiResDepthUpsamplePS;
        shaderDesc.ps_length = sizeof(g_multiResDepthUpsamplePS);
        shaderDesc.label = L"multiResDepthUpsamplePS";
        shaderDesc.numRenderTargets = 1;
        shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r32_float;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        m_multiResDepthUpsampleShader = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    {
        NvFlowGraphicsShaderDesc shaderDesc = {};
        shaderDesc.numInputElements = countof(elementDesc);
        shaderDesc.inputElementDescs = elementDesc;
        shaderDesc.blendState.enable = 0;
        shaderDesc.depthState.depthEnable = 1;
        shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
        shaderDesc.depthState.depthFunc = eNvFlowComparison_GreaterEqual;
        shaderDesc.uavTarget = 0;
        shaderDesc.depthClipEnable = 1;
        shaderDesc.vs = g_volumeRenderDepthVS;
        shaderDesc.vs_length = sizeof(g_volumeRenderDepthVS);
        shaderDesc.ps = g_volumeRenderDepthPS;
        shaderDesc.ps_length = sizeof(g_volumeRenderDepthPS);
        shaderDesc.label = L"volumeRenderDepthPS";
        shaderDesc.numRenderTargets = 0;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        m_volumeRenderDepth[0] = NvFlowCreateGraphicsShader(context, &shaderDesc);
        shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
        m_volumeRenderDepth[1] = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    {
        NvFlowGraphicsShaderDesc shaderDesc = {};
        shaderDesc.numInputElements = countof(elementDesc);
        shaderDesc.inputElementDescs = elementDesc;
        shaderDesc.blendState.enable = 0;
        shaderDesc.depthState.depthEnable = 1;
        shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
        shaderDesc.depthState.depthFunc = eNvFlowComparison_Always;
        shaderDesc.uavTarget = 0;
        shaderDesc.depthClipEnable = 1;
        shaderDesc.vs = g_rayMarchMaskVS;
        shaderDesc.vs_length = sizeof(g_rayMarchMaskVS);
        shaderDesc.ps = g_rayMarchMaskPS;
        shaderDesc.ps_length = sizeof(g_rayMarchMaskPS);
        shaderDesc.label = L"rayMarchMaskPS";
        shaderDesc.numRenderTargets = 0;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d16_unorm;
        m_rayMarchMask = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    {
        NvFlowGraphicsShaderDesc shaderDesc = {};
        shaderDesc.numInputElements = countof(elementDesc);
        shaderDesc.inputElementDescs = elementDesc;
        shaderDesc.vs = g_volumeRenderDepthEstimateVS;
        shaderDesc.vs_length = sizeof(g_volumeRenderDepthEstimateVS);
        shaderDesc.ps = g_volumeRenderDepthEstimatePS;
        shaderDesc.ps_length = sizeof(g_volumeRenderDepthEstimatePS);
        shaderDesc.label = L"volumeRenderDepthEstimatePS";
        shaderDesc.blendState.enable = 1;
        shaderDesc.blendState.srcBlendColor = eNvFlowBlend_One;
        shaderDesc.blendState.dstBlendColor = eNvFlowBlend_One;
        shaderDesc.blendState.blendOpColor = eNvFlowBlendOp_Add;
        shaderDesc.blendState.srcBlendAlpha = eNvFlowBlend_One;
        shaderDesc.blendState.dstBlendAlpha = eNvFlowBlend_One;
        shaderDesc.blendState.blendOpAlpha = eNvFlowBlendOp_Add;
        shaderDesc.depthState.depthEnable = 0;
        shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
        shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
        shaderDesc.numRenderTargets = 1;
        shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r16g16_float;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        shaderDesc.uavTarget = 0;
        shaderDesc.depthClipEnable = 1;
        m_volumeRenderDepthEstimate = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    RadixSortDesc radixSortDesc = {};
    radixSortDesc.maxSortBlocks = (requestedCapacity + 1023) / 1024;
    m_sort = createRadixSort(context, &radixSortDesc);

    RadixSortCPUDesc radixSortCPUDesc = {};
    radixSortCPUDesc.maxElements = requestedCapacity;
    m_sortCPU = createRadixSortCPU(&radixSortCPUDesc);

    m_blockListSorted.reserve(requestedCapacity);

    {
        NvFlowBufferDesc bufDesc = {};
        bufDesc.format = eNvFlowFormat_r32_uint;
        bufDesc.dim = requestedCapacity;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_blockListUpload = NvFlowCreateBuffer(context, &bufDesc);
    }

    {
        NvFlowGridImportDesc gridImportDesc = {};
        gridImportDesc.gridExport = m_desc.gridExport;
        m_gridImport = NvFlowCreateGridImport(context, &gridImportDesc);
    }

    {
        NvFlowComputeShaderDesc csDesc = {};
        csDesc.cs = g_volumeRenderLightingCS;
        csDesc.cs_length = sizeof(g_volumeRenderLightingCS);
        csDesc.label = L"volumeRenderLightingCS";
        m_lightingShader = NvFlowCreateComputeShader(context, &csDesc);

        csDesc.cs = g_volumeRenderLightingCS_SST;
        csDesc.cs_length = sizeof(g_volumeRenderLightingCS_SST);
        csDesc.label = L"volumeRenderLightingCS_SST";
        m_lightingShader_SST = NvFlowCreateComputeShader(context, &csDesc);

        csDesc.cs = g_volumeRenderLightingCS_VTR;
        csDesc.cs_length = sizeof(g_volumeRenderLightingCS_VTR);
        csDesc.label = L"volumeRenderLightingCS_VTR";
        m_lightingShader_VTR = NvFlowCreateComputeShader(context, &csDesc);
    }
}

VolumeRender::~VolumeRender() {
    SafeRelease(m_constantBuffer);
    SafeRelease(m_vertexBuffer);
    SafeRelease(m_indexBuffer);
    SafeRelease(m_compositeVertexBufferRect);
    SafeRelease(m_compositeIndexBufferRect);
    SafeRelease(m_compositeVertexBufferMultiRes);
    SafeRelease(m_compositeIndexBufferMultiRes);

    m_debugEmitBoundsBuffer.release();
    m_debugSphereBuffer.release();
    m_debugCapsuleBuffer.release();
    m_debugBoxBuffer.release();
    m_debugSimpleShapeMeshes.release();

    SafeRelease(m_compositeShader);
    SafeRelease(m_compositeShader_LMS);
    SafeRelease(m_compositeSmoothShader);
    SafeRelease(m_compositeSmoothShader_LMS);

    SafeRelease(m_compositeDepthEstimate);
    SafeRelease(m_compositeDepthEstimate_LMS);

    SafeRelease(m_compositeDepthDebug);
    SafeRelease(m_compositeDepthDebug_LMS);

    for (uint32_t i = 0; i < 2; ++i)
        SafeRelease(m_depthDownsampleShader[i]);

    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t ii = 0; ii < 4; ++ii)
            SafeRelease(m_volumeRender[k][ii]);
    }

    SafeRelease(m_volumeRenderDepthEstimate);
    SafeRelease(m_volumeRenderBox);
    SafeRelease(m_volumeRenderDebug);
    SafeRelease(m_volumeRenderDebugEmitBounds);
    SafeRelease(m_volumeRenderDebugShapesSimple);
    SafeRelease(m_sortShader);
    SafeRelease(m_multiResColorUpsampleShader);
    SafeRelease(m_multiResDepthDownsampleShader);
    SafeRelease(m_multiResDepthUpsampleShader);
    SafeRelease(m_volumeRenderDepth);
    SafeRelease(m_rayMarchMask);
    for (auto &buffer : m_offscreenBuffers) {
        buffer.release();
    }
    SafeRelease(m_depthMask);
    SafeRelease(m_depthEstimate);
    SafeRelease(m_sort);
    SafeRelease(m_sortCPU);
    SafeRelease(m_blockListUpload);
    SafeRelease(m_gridImport);
    SafeRelease(m_lightingShader);
    SafeRelease(m_lightingShader_SST);
    SafeRelease(m_lightingShader_VTR);
}

#pragma endregion "VolumeRender Constructor"

const float VolumeRender::DebugSimpleShapeMeshes::pts[] = {
    0.0,         1.0,         0.0,         1.0, 0.0,         0.98078501,  0.19509,     1.0,
    0.0,         0.92387998,  0.38268301,  1.0, 0.0,         0.83147001,  0.55557001,  1.0,
    0.0,         0.70710701,  0.70710701,  1.0, 0.0,         0.55557001,  0.83147001,  1.0,
    0.0,         0.38268301,  0.92387998,  1.0, 0.0,         0.19509,     0.98078501,  1.0,
    0.0,         -0.0,        1.0,         1.0, 0.0,         -0.19509,    0.98078501,  1.0,
    0.0,         -0.38268399, 0.92387998,  1.0, 0.0,         -0.55557001, 0.83147001,  1.0,
    0.0,         -0.70710701, 0.70710701,  1.0, 0.0,         -0.83147001, 0.55557001,  1.0,
    0.0,         -0.92387998, 0.38268301,  1.0, 0.0,         -0.98078501, 0.19509,     1.0,
    0.0,         -1.0,        -0.0,        1.0, 0.0,         -0.98078501, -0.19509,    1.0,
    0.0,         -0.92387998, -0.38268301, 1.0, 0.0,         -0.83147001, -0.55557001, 1.0,
    0.0,         -0.70710701, -0.70710701, 1.0, 0.0,         -0.55557001, -0.83147001, 1.0,
    0.0,         -0.38268301, -0.92387998, 1.0, 0.0,         -0.19509,    -0.98078501, 1.0,
    0.0,         0.0,         -1.0,        1.0, 0.0,         0.19509,     -0.98078501, 1.0,
    0.0,         0.38268399,  -0.92387903, 1.0, 0.0,         0.55557001,  -0.831469,   1.0,
    0.0,         0.70710701,  -0.70710701, 1.0, 0.0,         0.83147001,  -0.55557001, 1.0,
    0.0,         0.92387998,  -0.38268301, 1.0, 0.0,         0.98078501,  -0.19509,    1.0,
    0.0,         0.0,         1.0,         1.0, 0.19509,     0.0,         0.98078501,  1.0,
    0.38268301,  0.0,         0.92387998,  1.0, 0.55557001,  0.0,         0.83147001,  1.0,
    0.70710701,  0.0,         0.70710701,  1.0, 0.83147001,  0.0,         0.55557001,  1.0,
    0.92387998,  0.0,         0.38268301,  1.0, 0.98078501,  0.0,         0.19509,     1.0,
    1.0,         0.0,         -0.0,        1.0, 0.98078501,  0.0,         -0.19509,    1.0,
    0.92387998,  0.0,         -0.38268399, 1.0, 0.83147001,  0.0,         -0.55557001, 1.0,
    0.70710701,  0.0,         -0.70710701, 1.0, 0.55557001,  0.0,         -0.83147001, 1.0,
    0.38268301,  0.0,         -0.92387998, 1.0, 0.19509,     0.0,         -0.98078501, 1.0,
    -0.0,        0.0,         -1.0,        1.0, -0.19509,    0.0,         -0.98078501, 1.0,
    -0.38268301, 0.0,         -0.92387998, 1.0, -0.55557001, 0.0,         -0.83147001, 1.0,
    -0.70710701, 0.0,         -0.70710701, 1.0, -0.83147001, 0.0,         -0.55557001, 1.0,
    -0.92387998, 0.0,         -0.38268301, 1.0, -0.98078501, 0.0,         -0.19509,    1.0,
    -1.0,        0.0,         0.0,         1.0, -0.98078501, 0.0,         0.19509,     1.0,
    -0.92387903, 0.0,         0.38268399,  1.0, -0.831469,   0.0,         0.55557001,  1.0,
    -0.70710701, 0.0,         0.70710701,  1.0, -0.55557001, 0.0,         0.83147001,  1.0,
    -0.38268301, 0.0,         0.92387998,  1.0, -0.19509,    0.0,         0.98078501,  1.0,
    1.0,         0.0,         0.0,         1.0, 0.98078501,  0.19509,     0.0,         1.0,
    0.92387998,  0.38268301,  0.0,         1.0, 0.83147001,  0.55557001,  0.0,         1.0,
    0.70710701,  0.70710701,  0.0,         1.0, 0.55557001,  0.83147001,  0.0,         1.0,
    0.38268301,  0.92387998,  0.0,         1.0, 0.19509,     0.98078501,  0.0,         1.0,
    -0.0,        1.0,         0.0,         1.0, -0.19509,    0.98078501,  0.0,         1.0,
    -0.38268399, 0.92387998,  0.0,         1.0, -0.55557001, 0.83147001,  0.0,         1.0,
    -0.70710701, 0.70710701,  0.0,         1.0, -0.83147001, 0.55557001,  0.0,         1.0,
    -0.92387998, 0.38268301,  0.0,         1.0, -0.98078501, 0.19509,     0.0,         1.0,
    -1.0,        -0.0,        0.0,         1.0, -0.98078501, -0.19509,    0.0,         1.0,
    -0.92387998, -0.38268301, 0.0,         1.0, -0.83147001, -0.55557001, 0.0,         1.0,
    -0.70710701, -0.70710701, 0.0,         1.0, -0.55557001, -0.83147001, 0.0,         1.0,
    -0.38268301, -0.92387998, 0.0,         1.0, -0.19509,    -0.98078501, 0.0,         1.0,
    0.0,         -1.0,        0.0,         1.0, 0.19509,     -0.98078501, 0.0,         1.0,
    0.38268399,  -0.92387903, 0.0,         1.0, 0.55557001,  -0.831469,   0.0,         1.0,
    0.70710701,  -0.70710701, 0.0,         1.0, 0.83147001,  -0.55557001, 0.0,         1.0,
    0.92387998,  -0.38268301, 0.0,         1.0, 0.98078501,  -0.19509,    0.0,         1.0,
    -1.0,        1.0,         0.0,         1.0, -1.0,        0.98078501,  0.19509,     1.0,
    -1.0,        0.92387998,  0.38268301,  1.0, -1.0,        0.83147001,  0.55557001,  1.0,
    -1.0,        0.70710701,  0.70710701,  1.0, -1.0,        0.55557001,  0.83147001,  1.0,
    -1.0,        0.38268301,  0.92387998,  1.0, -1.0,        0.19509,     0.98078501,  1.0,
    -1.0,        -0.0,        1.0,         1.0, -1.0,        -0.19509,    0.98078501,  1.0,
    -1.0,        -0.38268399, 0.92387998,  1.0, -1.0,        -0.55557001, 0.83147001,  1.0,
    -1.0,        -0.70710701, 0.70710701,  1.0, -1.0,        -0.83147001, 0.55557001,  1.0,
    -1.0,        -0.92387998, 0.38268301,  1.0, -1.0,        -0.98078501, 0.19509,     1.0,
    -1.0,        -1.0,        -0.0,        1.0, -1.0,        -0.98078501, -0.19509,    1.0,
    -1.0,        -0.92387998, -0.38268301, 1.0, -1.0,        -0.83147001, -0.55557001, 1.0,
    -1.0,        -0.70710701, -0.70710701, 1.0, -1.0,        -0.55557001, -0.83147001, 1.0,
    -1.0,        -0.38268301, -0.92387998, 1.0, -1.0,        -0.19509,    -0.98078501, 1.0,
    -1.0,        0.0,         -1.0,        1.0, -1.0,        0.19509,     -0.98078501, 1.0,
    -1.0,        0.38268399,  -0.92387903, 1.0, -1.0,        0.55557001,  -0.831469,   1.0,
    -1.0,        0.70710701,  -0.70710701, 1.0, -1.0,        0.83147001,  -0.55557001, 1.0,
    -1.0,        0.92387998,  -0.38268301, 1.0, -1.0,        0.98078501,  -0.19509,    1.0,
    1.0,         1.0,         0.0,         1.0, 1.0,         0.98078501,  0.19509,     1.0,
    1.0,         0.92387998,  0.38268301,  1.0, 1.0,         0.83147001,  0.55557001,  1.0,
    1.0,         0.70710701,  0.70710701,  1.0, 1.0,         0.55557001,  0.83147001,  1.0,
    1.0,         0.38268301,  0.92387998,  1.0, 1.0,         0.19509,     0.98078501,  1.0,
    1.0,         -0.0,        1.0,         1.0, 1.0,         -0.19509,    0.98078501,  1.0,
    1.0,         -0.38268399, 0.92387998,  1.0, 1.0,         -0.55557001, 0.83147001,  1.0,
    1.0,         -0.70710701, 0.70710701,  1.0, 1.0,         -0.83147001, 0.55557001,  1.0,
    1.0,         -0.92387998, 0.38268301,  1.0, 1.0,         -0.98078501, 0.19509,     1.0,
    1.0,         -1.0,        -0.0,        1.0, 1.0,         -0.98078501, -0.19509,    1.0,
    1.0,         -0.92387998, -0.38268301, 1.0, 1.0,         -0.83147001, -0.55557001, 1.0,
    1.0,         -0.70710701, -0.70710701, 1.0, 1.0,         -0.55557001, -0.83147001, 1.0,
    1.0,         -0.38268301, -0.92387998, 1.0, 1.0,         -0.19509,    -0.98078501, 1.0,
    1.0,         0.0,         -1.0,        1.0, 1.0,         0.19509,     -0.98078501, 1.0,
    1.0,         0.38268399,  -0.92387903, 1.0, 1.0,         0.55557001,  -0.831469,   1.0,
    1.0,         0.70710701,  -0.70710701, 1.0, 1.0,         0.83147001,  -0.55557001, 1.0,
    1.0,         0.92387998,  -0.38268301, 1.0, 1.0,         0.98078501,  -0.19509,    1.0,
    2.0,         0.0,         0.0,         1.0, 1.980785,    0.0,         0.19509,     1.0,
    1.92388,     0.0,         0.38268301,  1.0, 1.83147,     0.0,         0.55557001,  1.0,
    1.7071069,   0.0,         0.70710701,  1.0, 1.55557,     0.0,         0.83147001,  1.0,
    1.382683,    0.0,         0.92387998,  1.0, 1.1950901,   0.0,         0.98078501,  1.0,
    1.0,         0.0,         1.0,         1.0, -1.0,        0.0,         1.0,         1.0,
    -1.1950901,  0.0,         0.98078501,  1.0, -1.382684,   0.0,         0.92387998,  1.0,
    -1.55557,    0.0,         0.83147001,  1.0, -1.7071069,  0.0,         0.70710701,  1.0,
    -1.83147,    0.0,         0.55557001,  1.0, -1.92388,    0.0,         0.38268301,  1.0,
    -1.980785,   0.0,         0.19509,     1.0, -2.0,        0.0,         -0.0,        1.0,
    -1.980785,   0.0,         -0.19509,    1.0, -1.92388,    0.0,         -0.38268301, 1.0,
    -1.83147,    0.0,         -0.55557001, 1.0, -1.7071069,  0.0,         -0.70710701, 1.0,
    -1.55557,    0.0,         -0.83147001, 1.0, -1.382683,   0.0,         -0.92387998, 1.0,
    -1.1950901,  0.0,         -0.98078501, 1.0, -1.0,        0.0,         -1.0,        1.0,
    1.0,         0.0,         -1.0,        1.0, 1.1950901,   0.0,         -0.98078501, 1.0,
    1.382684,    0.0,         -0.92387903, 1.0, 1.55557,     0.0,         -0.831469,   1.0,
    1.7071069,   0.0,         -0.70710701, 1.0, 1.83147,     0.0,         -0.55557001, 1.0,
    1.92388,     0.0,         -0.38268301, 1.0, 1.980785,    0.0,         -0.19509,    1.0,
    2.0,         0.0,         0.0,         1.0, 1.980785,    0.19509,     0.0,         1.0,
    1.92388,     0.38268301,  0.0,         1.0, 1.83147,     0.55557001,  0.0,         1.0,
    1.7071069,   0.70710701,  0.0,         1.0, 1.55557,     0.83147001,  0.0,         1.0,
    1.382683,    0.92387998,  0.0,         1.0, 1.1950901,   0.98078501,  0.0,         1.0,
    1.0,         1.0,         0.0,         1.0, -1.0,        1.0,         0.0,         1.0,
    -1.1950901,  0.98078501,  0.0,         1.0, -1.382684,   0.92387998,  0.0,         1.0,
    -1.55557,    0.83147001,  0.0,         1.0, -1.7071069,  0.70710701,  0.0,         1.0,
    -1.83147,    0.55557001,  0.0,         1.0, -1.92388,    0.38268301,  0.0,         1.0,
    -1.980785,   0.19509,     0.0,         1.0, -2.0,        -0.0,        0.0,         1.0,
    -1.980785,   -0.19509,    0.0,         1.0, -1.92388,    -0.38268301, 0.0,         1.0,
    -1.83147,    -0.55557001, 0.0,         1.0, -1.7071069,  -0.70710701, 0.0,         1.0,
    -1.55557,    -0.83147001, 0.0,         1.0, -1.382683,   -0.92387998, 0.0,         1.0,
    -1.1950901,  -0.98078501, 0.0,         1.0, -1.0,        -1.0,        0.0,         1.0,
    1.0,         -1.0,        0.0,         1.0, 1.1950901,   -0.98078501, 0.0,         1.0,
    1.382684,    -0.92387903, 0.0,         1.0, 1.55557,     -0.831469,   0.0,         1.0,
    1.7071069,   -0.70710701, 0.0,         1.0, 1.83147,     -0.55557001, 0.0,         1.0,
    1.92388,     -0.38268301, 0.0,         1.0, 1.980785,    -0.19509,    0.0,         1.0,
    -1.0,        -1.0,        -1.0,        1.0, 1.0,         -1.0,        -1.0,        1.0,
    1.0,         1.0,         -1.0,        1.0, -1.0,        1.0,         -1.0,        1.0,
    -1.0,        1.0,         1.0,         1.0, 1.0,         1.0,         1.0,         1.0,
    1.0,         -1.0,        1.0,         1.0, -1.0,        -1.0,        1.0,         1.0};

uint32_t VolumeRender::DebugSimpleShapeMeshes::indices[480] = {
    0u,   1u,   1u,   2u,   2u,   3u,   3u,   4u,   4u,   5u,   5u,   6u,   6u,   7u,
    7u,   8u,   8u,   9u,   9u,   10u,  10u,  11u,  11u,  12u,  12u,  13u,  13u,  14u,
    14u,  15u,  15u,  16u,  16u,  17u,  17u,  18u,  18u,  19u,  19u,  20u,  20u,  21u,
    21u,  22u,  22u,  23u,  23u,  24u,  24u,  25u,  25u,  26u,  26u,  27u,  27u,  28u,
    28u,  29u,  29u,  30u,  30u,  31u,  31u,  0u,   32u,  33u,  33u,  34u,  34u,  35u,
    35u,  36u,  36u,  37u,  37u,  38u,  38u,  39u,  39u,  40u,  40u,  41u,  41u,  42u,
    42u,  43u,  43u,  44u,  44u,  45u,  45u,  46u,  46u,  47u,  47u,  48u,  48u,  49u,
    49u,  50u,  50u,  51u,  51u,  52u,  52u,  53u,  53u,  54u,  54u,  55u,  55u,  56u,
    56u,  57u,  57u,  58u,  58u,  59u,  59u,  60u,  60u,  61u,  61u,  62u,  62u,  63u,
    63u,  32u,  64u,  65u,  65u,  66u,  66u,  67u,  67u,  68u,  68u,  69u,  69u,  70u,
    70u,  71u,  71u,  72u,  72u,  73u,  73u,  74u,  74u,  75u,  75u,  76u,  76u,  77u,
    77u,  78u,  78u,  79u,  79u,  80u,  80u,  81u,  81u,  82u,  82u,  83u,  83u,  84u,
    84u,  85u,  85u,  86u,  86u,  87u,  87u,  88u,  88u,  89u,  89u,  90u,  90u,  91u,
    91u,  92u,  92u,  93u,  93u,  94u,  94u,  95u,  95u,  64u,  0u,   1u,   1u,   2u,
    2u,   3u,   3u,   4u,   4u,   5u,   5u,   6u,   6u,   7u,   7u,   8u,   8u,   9u,
    9u,   10u,  10u,  11u,  11u,  12u,  12u,  13u,  13u,  14u,  14u,  15u,  15u,  16u,
    16u,  17u,  17u,  18u,  18u,  19u,  19u,  20u,  20u,  21u,  21u,  22u,  22u,  23u,
    23u,  24u,  24u,  25u,  25u,  26u,  26u,  27u,  27u,  28u,  28u,  29u,  29u,  30u,
    30u,  31u,  31u,  0u,   32u,  33u,  33u,  34u,  34u,  35u,  35u,  36u,  36u,  37u,
    37u,  38u,  38u,  39u,  39u,  40u,  40u,  41u,  41u,  42u,  42u,  43u,  43u,  44u,
    44u,  45u,  45u,  46u,  46u,  47u,  47u,  48u,  48u,  49u,  49u,  50u,  50u,  51u,
    51u,  52u,  52u,  53u,  53u,  54u,  54u,  55u,  55u,  56u,  56u,  57u,  57u,  58u,
    58u,  59u,  59u,  60u,  60u,  61u,  61u,  62u,  62u,  63u,  63u,  32u,  64u,  65u,
    65u,  66u,  66u,  67u,  67u,  68u,  68u,  69u,  69u,  70u,  70u,  71u,  71u,  72u,
    72u,  73u,  73u,  74u,  74u,  75u,  75u,  76u,  76u,  77u,  77u,  78u,  78u,  79u,
    79u,  80u,  80u,  81u,  81u,  82u,  82u,  83u,  83u,  84u,  84u,  85u,  85u,  86u,
    86u,  87u,  87u,  88u,  88u,  89u,  89u,  90u,  90u,  91u,  91u,  92u,  92u,  93u,
    93u,  94u,  94u,  95u,  95u,  96u,  96u,  97u,  97u,  64u,  98u,  99u,  99u,  100u,
    100u, 101u, 101u, 102u, 102u, 103u, 103u, 104u, 104u, 105u, 105u, 106u, 106u, 107u,
    107u, 108u, 108u, 109u, 109u, 110u, 110u, 111u, 111u, 112u, 112u, 113u, 113u, 114u,
    114u, 115u, 115u, 116u, 116u, 117u, 117u, 118u, 118u, 119u, 119u, 120u, 120u, 121u,
    121u, 122u, 122u, 123u, 123u, 124u, 124u, 125u, 125u, 126u, 126u, 127u, 127u, 128u,
    128u, 129u, 129u, 130u, 130u, 131u, 131u, 98u,  0u,   1u,   1u,   2u,   2u,   3u,
    3u,   0u,   4u,   5u,   5u,   6u,   6u,   7u,   7u,   4u,   0u,   7u,   1u,   6u,
    2u,   5u,   3u,   4u};

NvFlowVolumeRender *FlowCreateVolumeRender(NvFlowContext *context,
                                           const NvFlowVolumeRenderDesc *desc) {
    return new VolumeRender(context, desc);
}

}  // namespace NvFlow