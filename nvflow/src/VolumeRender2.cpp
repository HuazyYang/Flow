#include "VolumeRender.h"
#include "RenderMaterialPool.h"
#include "VolumeRenderUtils.h"

namespace NvFlow {

namespace {

NvFlowUint3 NvFlow_tableVal_to_coord(unsigned int val) {
    unsigned int valInv = ~val;
    return make_uint3(valInv & 0x3FF, (valInv >> 10) & 0x3FF, (valInv >> 20) & 0x3FF);
}

unsigned int NvFlow_flip(float zfloat) {
    unsigned int mask = -(*(unsigned int *)&zfloat >> 31) | 0x80000000u;
    return mask ^ *(unsigned int *)&zfloat;
}

}  // namespace

uint64_t VolumeRender::getGPUBytesUsed() {
    return 0;
}

NvFlowGridExport *VolumeRender::lightGridExport(NvFlowContext *context,
                                                NvFlowGridExport *gridExport,
                                                const NvFlowVolumeLightingParams *params) {
    NvFlowContextProfileGroupBegin(context, L"GridLightGridView");

    auto currentChannel = params->renderChannel;
    NvFlowGridExportHandle exportHandle =
        NvFlowGridExportGetHandle(gridExport, context, currentChannel);
    if (!exportHandle.numLayerViews)
        return gridExport;

    NvFlowGridImportParams importParams;
    importParams.gridExport = gridExport;
    importParams.channel = currentChannel;
    importParams.importMode = eNvFlowGridImportModeLinear;
    auto importHandle = NvFlowGridImportGetHandle(m_gridImport, context, &importParams);

    NvFlowGridExportLayeredView exportLayeredView = {};
    NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

    NvFlowGridImportLayeredView importLayeredView = {};
    NvFlowGridImportGetLayeredView(importHandle, &importLayeredView);

    NvFlowGridExportLayerView exportLayerView = {};
    NvFlowGridImportLayerView importLayerView = {};

    struct ShaderParams {
        NvFlowShaderLinearParams exportParams;
        NvFlowShaderLinearParams importParams;
        NvFlowUint4 renderMode;
        float alphaBias_layer0;
        float intensityBias_layer0;
        float pad1;
        float pad2;
        NvFlowFloat4 colorMapCompMask_layer0;
        NvFlowFloat4 colorMapRange_layer0;
        NvFlowFloat4 alphaCompMask_layer0;
        NvFlowFloat4 intensityCompMask_layer0;
    };

    for (uint32_t layerIdx = 0; layerIdx < importHandle.numLayerViews; ++layerIdx) {
        ZeroMemory(&exportLayerView, sizeof(exportLayerView));
        ZeroMemory(&importLayerView, sizeof(importLayerView));

        NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);
        NvFlowGridImportGetLayerView(importHandle, layerIdx, &importLayerView);

        auto renderMaterial =
            params->materialPool->getRenderMaterialHandle(exportLayerView.mapping.material);

        auto material = params->materialPool->getMaterialParams(renderMaterial);

        auto colorMap = params->materialPool->getColorMap(renderMaterial);

        auto mapped = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        mapped->exportParams = importLayeredView.mapping.shaderParams;
        mapped->importParams = exportLayeredView.mapping.shaderParams;
        mapped->renderMode = make_uint4(params->renderMode);
        mapped->alphaBias_layer0 = material->alphaBias;
        mapped->intensityBias_layer0 = material->intensityBias;
        mapped->pad1 = 0.f;
        mapped->pad2 = 0.f;
        mapped->colorMapCompMask_layer0 = material->colorMapCompMask;
        mapped->alphaCompMask_layer0 = material->alphaCompMask;
        mapped->intensityCompMask_layer0 = material->intensityCompMask;

        float colorMapOffset = material->colorMapMinX;
        float colorMapScale = 1.f / (material->colorMapMaxX - material->colorMapMinX);
        NvFlowFloat4 colorMapRange;
        colorMapRange.x = colorMapOffset;
        colorMapRange.y = colorMapScale;
        colorMapRange.z = material->colorMapMinX;
        colorMapRange.w = material->colorMapMaxX;
        mapped->colorMapRange_layer0 = colorMapRange;

        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_lightingShader;
        if (importLayeredView.mapping.shaderParams.isVTR.x) {
            dparams.shader = m_lightingShader_VTR;
            dparams.gridDim.x = (importLayeredView.mapping.shaderParams.blockDim.x *
                                     importLayerView.mapping.numBlocks +
                                 7) >>
                                3;
            dparams.gridDim.y =
                (importLayeredView.mapping.shaderParams.blockDim.y + 7) >> 3;
            dparams.gridDim.z =
                (importLayeredView.mapping.shaderParams.blockDim.z + 7) >> 3;
        } else {
            dparams.shader = m_lightingShader_SST;
            dparams.gridDim.x =
                (importLayeredView.mapping.shaderParams.linearBlockDim.z *
                     importLayeredView.mapping.shaderParams.linearBlockDim.y *
                     importLayeredView.mapping.shaderParams.linearBlockDim.x +
                 127) /
                128;
            dparams.gridDim.y = 1;
            dparams.gridDim.z = 1;
        }
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = exportLayerView.mapping.blockList;
        dparams.readOnly[1] = exportLayerView.mapping.blockTable;
        dparams.readOnly[2] = exportLayerView.data;
        dparams.readOnly[3] = importLayerView.mapping.blockList;
        dparams.readOnly[4] = importLayerView.mapping.blockTable;
        if (colorMap)
            dparams.readOnly[5] = colorMap;
        dparams.readWrite[0] = importLayerView.dataRW;
        NvFlowContextDispatch(context, &dparams);
    }

    NvFlowGridImportGetGridExport(m_gridImport, context);
    NvFlowContextProfileGroupEnd(context);

    return NvFlowGridImportGetGridExport(m_gridImport, context);
}

void VolumeRender::renderGridExport(NvFlowContext *context, NvFlowGridExport *gridExport,
                                    const NvFlowVolumeRenderParams *params) {
    if (gridExport) {
        auto channel = params->renderChannel;
        NvFlowGridExportHandle exportHandle =
            NvFlowGridExportGetHandle(gridExport, context, channel);
        if (exportHandle.numLayerViews) {
            NvFlowContextProfileGroupBegin(context, L"GridVolumeRender");

            bool renderDepth = !params->colorCompositeOnly;
            bool renderDebug = !params->preColorCompositeOnly;
            bool renderDepthEstimate = renderDepth && params->generateDepth;

            NvFlowGridExportLayeredView layeredView = {};
            NvFlowGridExportGetLayeredView(exportHandle, &layeredView);
            bool enableVTR = layeredView.mapping.shaderParams.isVTR.x;

            NvFlowDim blockDim = make_dim(layeredView.mapping.shaderParams.blockDim);
            NvFlowDim gridDim = make_dim(layeredView.mapping.shaderParams.gridDim);
            NvFlowFloat4 vdim = make_float4(blockDim * gridDim, 0u);
            NvFlowFloat4 vGridDimInv = 1.f / make_float4(gridDim, 1u);

            NvFlowFloat4x4 modelViewMatrix =
                layeredView.mapping.modelMatrix * params->viewMatrix;
            NvFlowFloat4x4 modelViewMatrixInv = inverse(modelViewMatrix);
            NvFlowFloat4x4 projectionMatrixInv = inverse(params->projectionMatrix);
            NvFlowFloat4x4 viewProj = params->viewMatrix * params->projectionMatrix;
            NvFlowFloat4x4 viewProjT = transpose(viewProj);
            NvFlowFloat4x4 modelViewProj = modelViewMatrix * params->projectionMatrix;
            NvFlowFloat4x4 modelViewProjT = transpose(modelViewProj);

            NvFlowFloat4 rayOriginVirtual;
            NvFlowFloat4 rayForwardDirVirtual;
            VolumeRenderUtils::compute_rayVirtual(&rayOriginVirtual, &rayForwardDirVirtual,
                                                  modelViewMatrixInv, vdim,
                                                  params->projectionMatrix);

            BlockListRangeList blockListRangeList = {};
            bool renderEmpty = 0;

            if (renderDepth || params->debugMode) {
                if (exportHandle.numLayerViews == 1) {
                    NvFlowGridExportLayerView layerView;
                    NvFlowGridExportGetLayerView(exportHandle, 0, &layerView);
                    blockListRangeList = generateBlockListRangeListSingle(
                        context, 0, layerView.mapping.numBlocks,
                        layerView.mapping.blockList,
                        layeredView.mapping.shaderParams.blockDim, rayOriginVirtual,
                        rayForwardDirVirtual);

                    renderEmpty = layerView.mapping.numBlocks == 0;
                } else {
                    blockListRangeList = generateBlockListRangeListLayered(
                        context, layeredView.mapping.layeredBlockListCPU,
                        layeredView.mapping.layeredNumBlocks, exportHandle.numLayerViews,
                        layeredView.mapping.shaderParams.blockDim, rayOriginVirtual,
                        rayForwardDirVirtual);

                    renderEmpty = layeredView.mapping.layeredNumBlocks == 0;
                }
            }

            if (!renderEmpty) {
                NvFlowRenderTargetView *rtv = 0;
                NvFlowDepthStencilView *dsv = 0;
                NvFlowRenderTarget *rt = 0;
                NvFlowDepthStencil *ds = 0;
                uint32_t rtv_width = 0;
                uint32_t rtv_height = 0;
                extractRenderTargets(&rtv, &dsv, &rt, &ds, &rtv_width, &rtv_height, params);

                resizeTargets(context, rtv_width, rtv_height, params);

                NvFlowResource *dsvResource =
                    dsv ? NvFlowDepthStencilViewGetResource(dsv) : nullptr;
                auto &buf = m_offscreenBuffers[0];

                NvFlowFloat4 depthInvTransform = make_float4(1.f, 0.f, 1.f, 0.f);
                VolumeRenderUtils::compute_depthInvTransform(
                    &depthInvTransform, rayForwardDirVirtual, buf.m_viewport,
                    projectionMatrixInv, vdim, modelViewMatrix);

                float tlimitMin = 0.f, tlimitMax = 0.f;
                VolumeRenderUtils::compute_tlimit(&tlimitMin, &tlimitMax, modelViewProj,
                                                  depthInvTransform);
                if (renderDepth) {
                    downsampleDepth(context, dsvResource, ds, &blockListRangeList,
                                    modelViewProjT, depthInvTransform, vGridDimInv, params);
                }

                if (renderDebug) {
                    debugRender(context, rt, ds, &blockListRangeList, gridExport, viewProjT,
                                modelViewProjT, vGridDimInv, params);
                }

                if (renderDepth) {
                    if (params->multiResRayMarch) {
                        rayMarchMultiRes(context, &blockListRangeList, exportHandle,
                                         modelViewProjT, depthInvTransform,
                                         rayOriginVirtual, rayForwardDirVirtual,
                                         &layeredView.mapping.shaderParams, gridDim,
                                         enableVTR, projectionMatrixInv, params);
                    } else {
                        rayMarch(
                            context, NvFlowColorBufferGetRenderTarget(buf.m_colorBuffer),
                            nullptr, true,
                            NvFlowColorBufferGetResource(buf.m_depthMaxBuffer), nullptr,
                            buf.m_viewport, buf.screenPercentX, buf.screenPercentY,
                            &blockListRangeList, exportHandle, modelViewProjT,
                            depthInvTransform, make_float4(0.f), rayOriginVirtual,
                            rayForwardDirVirtual, &layeredView.mapping.shaderParams,
                            gridDim, enableVTR, params);
                    }
                }

                if (renderDebug) {
                    composite(context, rt, ds, dsvResource, depthInvTransform, tlimitMin,
                              tlimitMax, params);
                }

                NvFlowContextRestoreResourceState(context, dsvResource);

                if (renderDepthEstimate) {
                    resizeDepth(context, rtv_width, rtv_height, params);

                    NvFlowColorBufferDesc desc;
                    NvFlowColorBufferGetDesc(m_depthEstimate, &desc);
                    auto RenderTarget = NvFlowColorBufferGetRenderTarget(m_depthEstimate);
                    NvFlowRenderTargetDesc depthEstimate_desc;
                    NvFlowRenderTargetGetDesc(RenderTarget, &depthEstimate_desc);
                    m_depthEstimateViewport = depthEstimate_desc.viewport;
                    m_depthEstimateViewport.width = desc.width * params->screenPercentage;
                    m_depthEstimateViewport.height = desc.height * params->screenPercentage;
                    m_depthEstimateScreenPercentX =
                        m_depthEstimateViewport.width / desc.width;
                    m_depthEstimateScreenPercentY =
                        m_depthEstimateViewport.height / desc.height;

                    rayMarchDepthEstimate(
                        context, NvFlowColorBufferGetRenderTarget(m_depthEstimate),
                        m_depthEstimateViewport, m_depthEstimateScreenPercentX,
                        m_depthEstimateScreenPercentY, exportHandle, modelViewProjT,
                        rayOriginVirtual, rayForwardDirVirtual,
                        &layeredView.mapping.shaderParams, gridDim, enableVTR, params);

                    compositeDepthEstimate(context, ds, dsv, depthInvTransform, tlimitMin,
                                           tlimitMax, params);

                    NvFlowContextRestoreResourceState(context, dsvResource);
                }

                if (renderDebug && params->generateDepth &&
                    params->generateDepthDebugMode) {
                    compositeDepthDebug(context, rt, ds, dsvResource, depthInvTransform,
                                        tlimitMin, tlimitMax, params);
                }
            }

            NvFlowContextProfileGroupEnd(context);
        }
    }
}

void VolumeRender::renderTexture3D(NvFlowContext *context, NvFlowTexture3D *density,
                                   const NvFlowVolumeRenderParams *params) {
    auto rtv = params->renderTargetView;
    auto dsv = params->depthStencilView;
    auto defMatHandle = params->materialPool->getDefaultRenderMaterial();
    auto defMaterial = params->materialPool->getMaterialParams(defMatHandle);
    auto &buf = m_offscreenBuffers[0];

    auto rt = NvFlowRenderTargetViewGetRenderTarget(rtv);
    auto ds = NvFlowDepthStencilViewGetDepthStencil(dsv);
    NvFlowRenderTargetDesc desc;
    NvFlowRenderTargetGetDesc(rt, &desc);
    uint32_t width = desc.viewport.width;
    uint32_t height = desc.viewport.height;

    bool resizeTex = 0;
    if (params->downsampleFactor) {
        if (params->downsampleFactor == eNvFlowVolumeRenderDownsample2x2 &&
            (buf.m_width != width / 2 || buf.m_height != height / 2)) {
            buf.m_width = width / 2;
            buf.m_height = height / 2;
            resizeTex = 1;
        }
    } else if (buf.m_width != width || buf.m_height != height) {
        buf.m_width = width;
        buf.m_height = height;
        resizeTex = 1;
    }

    if (resizeTex) {
        SafeRelease(buf.m_colorBuffer);
        SafeRelease(buf.m_depthMaxBuffer);

        NvFlowColorBufferDesc bufDesc;
        bufDesc.format = eNvFlowFormat_r16g16b16a16_float;
        bufDesc.width = buf.m_width;
        bufDesc.height = buf.m_height;
        buf.m_colorBuffer = NvFlowCreateColorBuffer(context, &bufDesc);
        bufDesc.format = eNvFlowFormat_r32_float;
        buf.m_depthMaxBuffer = NvFlowCreateColorBuffer(context, &bufDesc);
    }

    NvFlowTexture3DDesc texDesc;
    NvFlowTexture3DGetDesc(density, &texDesc);

    auto modelView = params->modelMatrix * params->viewMatrix;
    auto modelViewInv = inverse(modelView);
    auto modelViewProj = modelView * params->projectionMatrix;
    auto modelViewProjT = transpose(modelViewProj);
    auto modelViewProjInvT = inverse(modelViewProjT);

    auto rayOriginW = transform4(make_float4(0.f, 0.f, 0.f, 1.f), modelViewInv);
    rayOriginW = rayOriginW / rayOriginW.w;
    auto screenCorner = make_float4(texDesc.dim.x, texDesc.dim.y, texDesc.dim.z, 0.f);
    auto rayOrigin = screenCorner * (0.5f * rayOriginW + 0.5f);

    auto rayDirV = make_float4(0.f, 0.f, 1.f, 0.f);
    if (params->projectionMatrix.z.w < 0.f)
        rayDirV = make_float4(0.f, 0.f, -1.f, 0.f);
    auto rayDirW = transform4(rayDirV, modelViewInv);
    rayDirW = 0.5f * rayDirW;
    auto rayForwardDir = screenCorner * rayDirW;
    (NvFlowFloat3 &)rayForwardDir = 0.75f * normalize((const NvFlowFloat3 &)rayForwardDir);

    auto RenderTarget = NvFlowColorBufferGetRenderTarget(buf.m_colorBuffer);
    NvFlowRenderTargetDesc rtvDesc;
    NvFlowRenderTargetGetDesc(RenderTarget, &rtvDesc);

    auto mapped =
        (VolumeRender2ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (mapped) {
        mapped->modelViewProj = modelViewProjT;
        mapped->modelViewProjInv = modelViewProjInvT;

        auto &viewport = rtvDesc.viewport;
        NvFlowFloat4 viewportInvScale;
        viewportInvScale.x = 2.f / viewport.width;
        viewportInvScale.y = -2.f / viewport.height;
        viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
        viewportInvScale.w = 0.f;
        mapped->viewportInvScale = viewportInvScale;

        NvFlowFloat4 viewportInvOffset;
        viewportInvOffset.x = -2.f / viewport.width * viewport.topLeftX - 1.f;
        viewportInvOffset.y = 2.f / viewport.height * viewport.topLeftY + 1.f;
        viewportInvOffset.z = -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
        viewportInvOffset.w = 1.f;
        mapped->viewportInvOffset = viewportInvOffset;

        mapped->dimInv = 1.f / make_float4(texDesc.dim, 1.f);
        mapped->minCoord = make_float4(0.f);
        mapped->maxCoord = make_float4(texDesc.dim, 1);
        mapped->rayOrigin = rayOrigin;
        mapped->rayForwardDir = rayForwardDir;
        mapped->renderMode = make_uint4(params->renderMode);
        mapped->alphaScale = make_float4(defMaterial->alphaScale);

        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    NvFlowFloat4 color = make_float4(0.f, 0.f, 0.f, 1.f);
    NvFlowContextSetRenderTarget(context, RenderTarget, 0);
    NvFlowContextClearRenderTarget(context, RenderTarget, color);

    NvFlowDrawParams drawParams = {};
    drawParams.shader = m_volumeRenderBox;
    drawParams.rootConstantBuffer = m_constantBuffer;
    drawParams.ps_readOnly[0] = NvFlowTexture3DGetResource(density);
    NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
    NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);
    NvFlowContextDrawIndexedInstanced(context, 0x24, 1, &drawParams);

    // Composite
    {
        struct CompositeShaderParams {
            NvFlowFloat4 uvScale;
        };

        NvFlowContextSetRenderTarget(context, rt, ds);

        auto mapped =
            (CompositeShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            mapped->uvScale = make_float4(0.5f, -0.5f, 0.5f, 0.5f);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowRenderTargetDesc rt_desc;
        NvFlowDepthStencilDesc ds_desc;
        NvFlowRenderTargetGetDesc(rt, &rt_desc);
        NvFlowDepthStencilGetDesc(ds, &ds_desc);

        NvFlowGraphicsShaderSetFormats(context, m_compositeShader, rt_desc.rt_format,
                                       ds_desc.ds_format);

        NvFlowDrawParams drawParams;
        drawParams.shader = m_compositeShader;
        drawParams.rootConstantBuffer = m_constantBuffer;
        drawParams.ps_readOnly[0] = NvFlowColorBufferGetResource(buf.m_colorBuffer);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
        NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);

        auto Resource = NvFlowDepthStencilViewGetResource(dsv);
        NvFlowContextRestoreResourceState(context, Resource);
    }
}

void VolumeRender::composite(NvFlowContext *context, NvFlowRenderTarget *rt,
                             NvFlowDepthStencil *ds, NvFlowResource *dsvResource,
                             const NvFlowFloat4 &depthInvTransform, float tlimitMin,
                             float tlimitMax, const NvFlowVolumeRenderParams *params) {
    auto &offscreenBuf = m_offscreenBuffers[0];
    auto colorBuffer = offscreenBuf.m_colorBuffer;
    auto depthBuffer = offscreenBuf.m_depthMaxBuffer;
    NvFlowRenderTargetDesc rt_desc;
    NvFlowDepthStencilDesc ds_desc;
    NvFlowColorBufferDesc depthBuffer_desc;

    NvFlowRenderTargetGetDesc(rt, &rt_desc);
    NvFlowDepthStencilGetDesc(ds, &ds_desc);
    NvFlowColorBufferGetDesc(depthBuffer, &depthBuffer_desc);

    struct CompositeParams {
        using float2 = NvFlowFloat2;
#include "compositeShaderParams.h"
    };

    auto mapped = (CompositeParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (mapped) {
        mapped->uvScale.x = 0.5f * offscreenBuf.screenPercentX;
        mapped->uvScale.y = -0.5f * offscreenBuf.screenPercentY;
        mapped->uvScale.z = 0.5f * offscreenBuf.screenPercentX;
        mapped->uvScale.w = 0.5f * offscreenBuf.screenPercentY;
        mapped->compositeMode = 0;
        mapped->depthAlphaThreshold = params->depthAlphaThreshold;
        mapped->depthIntensityThreshold = params->depthIntensityThreshold;
        mapped->pad1 = 0.f;

        NvFlowFloat2 uvs, uvo;
        uvs.x = ds_desc.viewport.width / float(ds_desc.width);
        uvs.y = ds_desc.viewport.height / float(ds_desc.height);
        uvo.x = ds_desc.viewport.topLeftX / float(ds_desc.width);
        uvo.y = ds_desc.viewport.topLeftY / float(ds_desc.height);

        const auto &viewport = rt_desc.viewport;
        NvFlowFloat4 viewportInvScale;
        viewportInvScale.x = 2.f / viewport.width;
        viewportInvScale.y = -(2.f / viewport.height);
        viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
        viewportInvScale.w = 0.f;

        NvFlowFloat4 viewportInvOffset;
        viewportInvOffset.x = -2.f / float(viewport.width) * viewport.topLeftX - 1.f;
        viewportInvOffset.y = 2.f / float(viewport.height) * viewport.topLeftY + 1.f;
        viewportInvOffset.z = -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
        viewportInvOffset.w = 1.f;

        NvFlowFloat4 scene_uvScale;
        scene_uvScale.x = 0.5f * uvs.x * viewportInvScale.x;
        scene_uvScale.y = (-0.5f * uvs.y) * viewportInvScale.y;
        scene_uvScale.z = (0.5f * uvs.x + uvo.x) + (0.5f * uvs.x * viewportInvOffset.x);
        scene_uvScale.y = (0.5f * uvs.y + uvo.y) + (-0.5f * uvs.y * viewportInvOffset.y);
        mapped->scene_uvScale = scene_uvScale;

        NvFlowFloat4 uvStepSize;
        uvStepSize.x = 1.f / depthBuffer_desc.width;
        uvStepSize.y = 1.f / depthBuffer_desc.height;
        uvStepSize.z = depthBuffer_desc.width;
        uvStepSize.w = depthBuffer_desc.height;
        mapped->uvStepSize = uvStepSize;

        mapped->depthInvTransform = depthInvTransform;
        mapped->tlimitRange = make_float4(tlimitMin, tlimitMax, tlimitMin, tlimitMax);
        auto &lms = params->lensMatchedShading;
        mapped->warpLeft = lms.warpLeft;
        mapped->warpRight = lms.warpRight;
        mapped->warpUp = lms.warpUp;
        mapped->warpDown = lms.warpDown;

        float uvX = lms.sizeLeft / float(lms.sizeLeft + lms.sizeRight);
        float uvY = lms.sizeUp / float(lms.sizeUp + lms.sizeDown);
        mapped->leftX.x = (1.f / (1.f + lms.warpLeft)) / uvX;
        mapped->leftX.y = uvX;
        mapped->rightX.x = (1.f / (1.f + lms.warpRight)) / (1.f - uvX);
        mapped->rightX.y = uvX;
        mapped->upY.x = (1.f / (1.f + lms.warpUp)) / uvY;
        mapped->upY.y = uvY;
        mapped->downY.x = (1.f / (1.f + lms.warpDown)) / (1.f - uvY);
        mapped->downY.y = uvY;
        mapped->screenPercent.x = offscreenBuf.screenPercentX;
        mapped->screenPercent.y = offscreenBuf.screenPercentY;
        mapped->screenPercent.z = 1.f / offscreenBuf.screenPercentX;
        mapped->screenPercent.w = 1.f / offscreenBuf.screenPercentY;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    NvFlowContextSetRenderTarget(context, rt, nullptr);
    bool shouldSmooth = params->smoothColorUpsample;
    auto drawParams_shader = shouldSmooth ? m_compositeSmoothShader : m_compositeShader;
    if (params->lensMatchedShading.enabled) {
        drawParams_shader =
            shouldSmooth ? m_compositeSmoothShader_LMS : m_compositeShader_LMS;
    }

    NvFlowDrawParams drawParams = {};
    drawParams.shader = drawParams_shader;
    NvFlowGraphicsShaderSetFormats(context, drawParams_shader, rt_desc.rt_format,
                                   ds_desc.ds_format);
    drawParams.rootConstantBuffer = m_constantBuffer;
    drawParams.ps_readOnly[0] = NvFlowColorBufferGetResource(colorBuffer);
    drawParams.ps_readOnly[1] = NvFlowColorBufferGetResource(depthBuffer);
    drawParams.ps_readOnly[2] = dsvResource;
    drawParams.ps_readOnly[3] = NvFlowDepthBufferGetResource(m_depthMask);

    if (params->multiRes.enabled) {
        generateMultiResMesh(context, &params->multiRes);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferMultiRes,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferMultiRes, 0);
        NvFlowContextDrawIndexedInstanced(context, 54, 1, &drawParams);
    } else {
        generateDefaultMesh(context);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
        NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
    }

    NvFlowContextSetRenderTarget(context, rt, ds);
}

void VolumeRender::compositeDepthDebug(NvFlowContext *context, NvFlowRenderTarget *rt,
                                       NvFlowDepthStencil *ds, NvFlowResource *dsvResource,
                                       const NvFlowFloat4 &depthInvTransform,
                                       float tlimitMin, float tlimitMax,
                                       const NvFlowVolumeRenderParams *params) {
    if (m_depthEstimate && params->generateDepth && params->generateDepthDebugMode) {
        auto &offscreenBuf = m_offscreenBuffers[0];
        auto colorBuffer = offscreenBuf.m_colorBuffer;
        auto depthEstimateBuffer = m_depthEstimate;
        NvFlowRenderTargetDesc rt_desc;
        NvFlowDepthStencilDesc ds_desc;
        NvFlowRenderTargetGetDesc(rt, &rt_desc);
        NvFlowDepthStencilGetDesc(ds, &ds_desc);
        NvFlowColorBufferDesc depthEstimateBuffer_desc;
        NvFlowColorBufferGetDesc(depthEstimateBuffer, &depthEstimateBuffer_desc);

        struct CompositeParams {
            using float2 = NvFlowFloat2;
#include "compositeShaderParams.h"
        };

        auto mapped = (CompositeParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            mapped->uvScale.x = 0.5f * m_depthEstimateScreenPercentX;
            mapped->uvScale.y = -0.5f * m_depthEstimateScreenPercentY;
            mapped->uvScale.z = 0.5f * m_depthEstimateScreenPercentX;
            mapped->uvScale.w = 0.5f * m_depthEstimateScreenPercentY;
            mapped->compositeMode = params->generateDepthDebugMode;
            mapped->depthAlphaThreshold = params->depthAlphaThreshold;
            mapped->depthIntensityThreshold = params->depthIntensityThreshold;
            mapped->pad1 = 0.f;

            NvFlowFloat2 uvs, uvo;
            uvs.x = ds_desc.viewport.width / float(ds_desc.width);
            uvs.y = ds_desc.viewport.height / float(ds_desc.height);
            uvo.x = ds_desc.viewport.topLeftX / float(ds_desc.width);
            uvo.y = ds_desc.viewport.topLeftY / float(ds_desc.height);

            const auto &viewport = ds_desc.viewport;
            NvFlowFloat4 viewportInvScale;
            viewportInvScale.x = 2.f / viewport.width;
            viewportInvScale.y = -(2.f / viewport.height);
            viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
            viewportInvScale.w = 0.f;

            NvFlowFloat4 viewportInvOffset;
            viewportInvOffset.x = -2.f / float(viewport.width) * viewport.topLeftX - 1.f;
            viewportInvOffset.y = 2.f / float(viewport.height) * viewport.topLeftY + 1.f;
            viewportInvOffset.z =
                -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
            viewportInvOffset.w = 1.f;

            NvFlowFloat4 scene_uvScale;
            scene_uvScale.x = 0.5f * uvs.x * viewportInvScale.x;
            scene_uvScale.y = (-0.5f * uvs.y) * viewportInvScale.y;
            scene_uvScale.z = (0.5f * uvs.x + uvo.x) + (0.5f * uvs.x * viewportInvOffset.x);
            scene_uvScale.y =
                (0.5f * uvs.y + uvo.y) + (-0.5f * uvs.y * viewportInvOffset.y);
            mapped->scene_uvScale = scene_uvScale;

            NvFlowFloat4 uvStepSize;
            uvStepSize.x = 1.f / depthEstimateBuffer_desc.width;
            uvStepSize.y = 1.f / depthEstimateBuffer_desc.height;
            uvStepSize.z = depthEstimateBuffer_desc.width;
            uvStepSize.w = depthEstimateBuffer_desc.height;
            mapped->uvStepSize = uvStepSize;

            mapped->depthInvTransform = depthInvTransform;
            mapped->tlimitRange = make_float4(tlimitMin, tlimitMax, tlimitMin, tlimitMax);
            auto &lms = params->lensMatchedShading;
            mapped->warpLeft = lms.warpLeft;
            mapped->warpRight = lms.warpRight;
            mapped->warpUp = lms.warpUp;
            mapped->warpDown = lms.warpDown;

            float uvX = lms.sizeLeft / float(lms.sizeLeft + lms.sizeRight);
            float uvY = lms.sizeUp / float(lms.sizeUp + lms.sizeDown);
            mapped->leftX.x = (1.f / (1.f + lms.warpLeft)) / uvX;
            mapped->leftX.y = uvX;
            mapped->rightX.x = (1.f / (1.f + lms.warpRight)) / (1.f - uvX);
            mapped->rightX.y = uvX;
            mapped->upY.x = (1.f / (1.f + lms.warpUp)) / uvY;
            mapped->upY.y = uvY;
            mapped->downY.x = (1.f / (1.f + lms.warpDown)) / (1.f - uvY);
            mapped->downY.y = uvY;
            mapped->screenPercent.x = m_depthEstimateScreenPercentX;
            mapped->screenPercent.y = m_depthEstimateScreenPercentY;
            mapped->screenPercent.z = 1.f / m_depthEstimateScreenPercentX;
            mapped->screenPercent.w = 1.f / m_depthEstimateScreenPercentY;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowContextSetRenderTarget(context, rt, nullptr);
        NvFlowGraphicsShader *drawParams_shader = params->lensMatchedShading.enabled
                                                      ? m_compositeDepthDebug_LMS
                                                      : m_compositeDepthDebug;
        NvFlowDrawParams drawParams = {};
        drawParams.shader = drawParams_shader;
        NvFlowGraphicsShaderSetFormats(context, drawParams_shader, rt_desc.rt_format,
                                       ds_desc.ds_format);
        drawParams.rootConstantBuffer = m_constantBuffer;
        drawParams.ps_readOnly[0] =
            colorBuffer ? NvFlowColorBufferGetResource(colorBuffer) : nullptr;
        drawParams.ps_readOnly[1] = depthEstimateBuffer
                                        ? NvFlowColorBufferGetResource(depthEstimateBuffer)
                                        : nullptr;

        if (params->multiRes.enabled) {
            generateMultiResMesh(context, &params->multiRes);
            NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferMultiRes,
                                         sizeof(NvFlowFloat4), 0);
            NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferMultiRes, 0);
            NvFlowContextDrawIndexedInstanced(context, 54, 1, &drawParams);
        } else {
            generateDefaultMesh(context);
            NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                         sizeof(NvFlowFloat4), 0);
            NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
            NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
        }

        NvFlowContextSetRenderTarget(context, rt, ds);
    }
}

void VolumeRender::compositeDepthEstimate(NvFlowContext *context, NvFlowDepthStencil *ds,
                                          NvFlowDepthStencilView *dsv,
                                          const NvFlowFloat4 &depthInvTransform,
                                          float tlimitMin, float tlimitMax,
                                          const NvFlowVolumeRenderParams *params) {
    auto &offscreenBuf = m_offscreenBuffers[0];
    auto colorBuffer = offscreenBuf.m_colorBuffer;
    auto depthEstimateBuffer = m_depthEstimate;
    NvFlowDepthStencilDesc ds_desc;
    NvFlowDepthStencilGetDesc(ds, &ds_desc);
    NvFlowColorBufferDesc depthEstimateBuffer_desc;
    NvFlowColorBufferGetDesc(depthEstimateBuffer, &depthEstimateBuffer_desc);

    struct CompositeParams {
        using float2 = NvFlowFloat2;
#include "compositeShaderParams.h"
    };

    auto mapped = (CompositeParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (mapped) {
        mapped->uvScale.x = 0.5f * m_depthEstimateScreenPercentX;
        mapped->uvScale.y = -0.5f * m_depthEstimateScreenPercentY;
        mapped->uvScale.z = 0.5f * m_depthEstimateScreenPercentX;
        mapped->uvScale.w = 0.5f * m_depthEstimateScreenPercentY;
        mapped->compositeMode = params->generateDepthDebugMode;
        mapped->depthAlphaThreshold = params->depthAlphaThreshold;
        mapped->depthIntensityThreshold = params->depthIntensityThreshold;
        mapped->pad1 = 0.f;

        NvFlowFloat2 uvs, uvo;
        uvs.x = ds_desc.viewport.width / float(ds_desc.width);
        uvs.y = ds_desc.viewport.height / float(ds_desc.height);
        uvo.x = ds_desc.viewport.topLeftX / float(ds_desc.width);
        uvo.y = ds_desc.viewport.topLeftY / float(ds_desc.height);

        const auto &viewport = ds_desc.viewport;
        NvFlowFloat4 viewportInvScale;
        viewportInvScale.x = 2.f / viewport.width;
        viewportInvScale.y = -(2.f / viewport.height);
        viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
        viewportInvScale.w = 0.f;

        NvFlowFloat4 viewportInvOffset;
        viewportInvOffset.x = -2.f / float(viewport.width) * viewport.topLeftX - 1.f;
        viewportInvOffset.y = 2.f / float(viewport.height) * viewport.topLeftY + 1.f;
        viewportInvOffset.z = -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
        viewportInvOffset.w = 1.f;

        NvFlowFloat4 scene_uvScale;
        scene_uvScale.x = 0.5f * uvs.x * viewportInvScale.x;
        scene_uvScale.y = (-0.5f * uvs.y) * viewportInvScale.y;
        scene_uvScale.z = (0.5f * uvs.x + uvo.x) + (0.5f * uvs.x * viewportInvOffset.x);
        scene_uvScale.y = (0.5f * uvs.y + uvo.y) + (-0.5f * uvs.y * viewportInvOffset.y);
        mapped->scene_uvScale = scene_uvScale;

        NvFlowFloat4 uvStepSize;
        uvStepSize.x = 1.f / depthEstimateBuffer_desc.width;
        uvStepSize.y = 1.f / depthEstimateBuffer_desc.height;
        uvStepSize.z = depthEstimateBuffer_desc.width;
        uvStepSize.w = depthEstimateBuffer_desc.height;
        mapped->uvStepSize = uvStepSize;

        mapped->depthInvTransform = depthInvTransform;
        mapped->tlimitRange = make_float4(tlimitMin, tlimitMax, tlimitMin, tlimitMax);
        auto &lms = params->lensMatchedShading;
        mapped->warpLeft = lms.warpLeft;
        mapped->warpRight = lms.warpRight;
        mapped->warpUp = lms.warpUp;
        mapped->warpDown = lms.warpDown;

        float uvX = lms.sizeLeft / float(lms.sizeLeft + lms.sizeRight);
        float uvY = lms.sizeUp / float(lms.sizeUp + lms.sizeDown);
        mapped->leftX.x = (1.f / (1.f + lms.warpLeft)) / uvX;
        mapped->leftX.y = uvX;
        mapped->rightX.x = (1.f / (1.f + lms.warpRight)) / (1.f - uvX);
        mapped->rightX.y = uvX;
        mapped->upY.x = (1.f / (1.f + lms.warpUp)) / uvY;
        mapped->upY.y = uvY;
        mapped->downY.x = (1.f / (1.f + lms.warpDown)) / (1.f - uvY);
        mapped->downY.y = uvY;
        mapped->screenPercent.x = m_depthEstimateScreenPercentX;
        mapped->screenPercent.y = m_depthEstimateScreenPercentY;
        mapped->screenPercent.z = 1.f / m_depthEstimateScreenPercentX;
        mapped->screenPercent.w = 1.f / m_depthEstimateScreenPercentY;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    NvFlowContextSetRenderTarget(context, nullptr, ds);
    bool isReverseZ = 0;
    float z0 = (depthInvTransform.x * 0.f + depthInvTransform.y) /
               (depthInvTransform.z * 0.f + depthInvTransform.w);
    float z1 = (depthInvTransform.x * 1.f + depthInvTransform.y) /
               (depthInvTransform.z * 1.f + depthInvTransform.w);
    isReverseZ = z0 > z1;
    bool generateDepthDebugMode = z0 > z1;
    auto drawParams_shader = m_compositeDepthEstimate[isReverseZ];
    if (params->lensMatchedShading.enabled) {
        generateDepthDebugMode = isReverseZ;
        drawParams_shader = m_compositeDepthEstimate_LMS[isReverseZ];
    }

    NvFlowDrawParams drawParams = {};
    drawParams.shader = drawParams_shader;

    NvFlowGraphicsShaderDesc drawParams_shader_desc;
    NvFlowGraphicsShaderGetDesc(drawParams_shader, &drawParams_shader_desc);
    NvFlowGraphicsShaderSetFormats(context, drawParams_shader,
                                   drawParams_shader_desc.renderTargetFormat[0],
                                   ds_desc.ds_format);
    drawParams.rootConstantBuffer = m_constantBuffer;
    drawParams.ps_readOnly[0] =
        colorBuffer ? NvFlowColorBufferGetResource(colorBuffer) : nullptr;
    drawParams.ps_readOnly[1] = NvFlowColorBufferGetResource(depthEstimateBuffer);

    if (params->multiRes.enabled) {
        generateMultiResMesh(context, &params->multiRes);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferMultiRes,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferMultiRes, 0);
        NvFlowContextDrawIndexedInstanced(context, 54, 1, &drawParams);
    } else {
        generateDefaultMesh(context);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
        NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
    }

    NvFlowContextSetRenderTarget(context, nullptr, ds);
}

void VolumeRender::debugRender(NvFlowContext *context, NvFlowRenderTarget *rt,
                               NvFlowDepthStencil *ds,
                               const BlockListRangeList *blockListRangeList,
                               NvFlowGridExport *gridExport, const NvFlowFloat4x4 &viewProj,
                               const NvFlowFloat4x4 &modelViewProj,
                               const NvFlowFloat4 &vGridDimInv,
                               const NvFlowVolumeRenderParams *params) {
    if (params->debugMode) {
        NvFlowGridExportDebugVisView debugVisView = {};
        NvFlowGridExportGetDebugVisView(gridExport, &debugVisView);
        if (debugVisView.debugVisFlags & eNvFlowGridDebugVisBlocks) {
            NvFlowContextSetRenderTarget(context, rt, ds);

            struct ShaderParams {
                NvFlowFloat4x4 modelViewProj;
                NvFlowFloat4 vGridDimInv;
            };

            auto mapped =
                (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
            if (mapped) {
                mapped->modelViewProj = modelViewProj;
                mapped->vGridDimInv = vGridDimInv;
                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }

            NvFlowDrawParams drawParams = {};
            drawParams.shader = m_volumeRenderDebug;
            drawParams.rootConstantBuffer = m_constantBuffer;
            drawParams.vs_readOnly[0] = blockListRangeList->allBlockList;
            NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
            NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0x90);
            NvFlowContextDrawIndexedInstanced(
                context, 0x18, blockListRangeList->allNumBlocks + 1, &drawParams);
        }

        if ((debugVisView.debugVisFlags & eNvFlowGridDebugVisEmitBounds) && gridExport &&
            debugVisView.numBounds) {
            NvFlowContextSetRenderTarget(context, rt, ds);

            m_debugEmitBoundsBuffer.reserve(context, eNvFlowFormat_r32g32b32a32_float,
                                            4 * debugVisView.numBounds);

            NvFlowFloat4x4 *mappedData = (NvFlowFloat4x4 *)NvFlowBufferMap(
                context, m_debugEmitBoundsBuffer.m_buffer);
            if (mappedData) {
                for (uint32_t i = 0; i < debugVisView.numBounds; ++i) {
                    mappedData[i] = debugVisView.bounds[i];
                }
                NvFlowBufferUnmap(context, m_debugEmitBoundsBuffer.m_buffer);
            }

            struct ShaderParams {
                NvFlowFloat4x4 modelViewProj;
            };

            auto mapped =
                (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
            if (mapped) {
                mapped->modelViewProj = viewProj;
                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }

            NvFlowDrawParams drawParams = {};
            drawParams.shader = m_volumeRenderDebugEmitBounds;
            drawParams.rootConstantBuffer = m_constantBuffer;
            drawParams.vs_readOnly[0] =
                NvFlowBufferGetResource(m_debugEmitBoundsBuffer.m_buffer);
            NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
            NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0x90);
            NvFlowContextDrawIndexedInstanced(context, 0x18, debugVisView.numBounds,
                                              &drawParams);
        }

        if ((debugVisView.debugVisFlags & eNvFlowGridDebugVisShapesSimple) && gridExport) {
            const int numPasses = 3;
            const NvFlowShapeType shapeTypes[] = {
                eNvFlowShapeTypeSphere, eNvFlowShapeTypeCapsule, eNvFlowShapeTypeBox};
            DebugUploadBuffer *const uploadBuffers[] = {
                &m_debugSphereBuffer, &m_debugCapsuleBuffer, &m_debugBoxBuffer};
            NvFlowGridExportSimpleShape *shapeArr[] = {
                debugVisView.spheres, debugVisView.capsules, debugVisView.boxes};
            uint32_t numShapeArr[] = {debugVisView.numSpheres, debugVisView.numCapsules,
                                      debugVisView.numBoxes};

            for (int passID = 0; passID < numPasses; ++passID) {
                auto shapeType = shapeTypes[passID];
                auto uploadBuffer = uploadBuffers[passID];
                auto shapes = shapeArr[passID];
                auto numShapes = numShapeArr[passID];
                if (numShapes) {
                    NvFlowContextSetRenderTarget(context, rt, ds);
                    uploadBuffer->reserve(context, eNvFlowFormat_r32g32b32a32_float,
                                          5 * numShapes);

                    auto mappedData = (NvFlowGridExportSimpleShape *)NvFlowBufferMap(
                        context, uploadBuffer->m_buffer);
                    if (mappedData) {
                        for (uint32_t i = 0; i < numShapes; ++i)
                            mappedData[i] = shapes[i];
                        NvFlowBufferUnmap(context, uploadBuffer->m_buffer);
                    }

                    struct ShaderParams {
                        NvFlowFloat4x4 modelViewProj;
                        NvFlowUint4 shapeType;
                    };
                    auto mapped =
                        (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
                    if (mapped) {
                        mapped->modelViewProj = viewProj;
                        mapped->shapeType = make_uint4((uint32_t)shapeType, 0, 0, 0);
                        NvFlowConstantBufferUnmap(context, m_constantBuffer);
                    }

                    NvFlowDrawParams drawParams = {};
                    drawParams.shader = m_volumeRenderDebugShapesSimple;
                    drawParams.rootConstantBuffer = m_constantBuffer;
                    drawParams.vs_readOnly[0] =
                        NvFlowBufferGetResource(uploadBuffer->m_buffer);
                    m_debugSimpleShapeMeshes.drawMesh(context, shapeType, numShapes,
                                                      &drawParams);
                }
            }
        }
    }
}

void VolumeRender::downsampleDepth(NvFlowContext *context, NvFlowResource *dsvResource,
                                   NvFlowDepthStencil *ds,
                                   const BlockListRangeList *blockListRangeList,
                                   const NvFlowFloat4x4 &modelViewProjT,
                                   const NvFlowFloat4 &depthInvTransform,
                                   const NvFlowFloat4 &vGridDimInv,
                                   const NvFlowVolumeRenderParams *params) {
    bool shouldGenDepthMask = false;
    if (params->multiResRayMarch > eNvFlowMultiResRayMarchDisabled)
        shouldGenDepthMask = params->multiResRayMarch < eNvFlowMultiResRayMarch16x16;
    if (params->smoothColorUpsample)
        shouldGenDepthMask = 1;

    bool isReverseZ = false;
    float z0 = (depthInvTransform.x * 0.f + depthInvTransform.y) /
               (depthInvTransform.z * 0.f + depthInvTransform.w);
    float z1 = (depthInvTransform.x * 1.f + depthInvTransform.y) /
               (depthInvTransform.z * 1.f + depthInvTransform.w);
    isReverseZ = z0 > z1;

    if (shouldGenDepthMask) {
        struct ShaderParams {
            NvFlowFloat4x4 modelViewProj;
            NvFlowFloat4 vGridDimInv;
        };

        auto mapped = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            mapped->modelViewProj = modelViewProjT;
            mapped->vGridDimInv = vGridDimInv;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        auto DepthStencil = NvFlowDepthBufferGetDepthStencil(m_depthMask);
        NvFlowContextSetRenderTarget(context, nullptr, DepthStencil);
        NvFlowContextClearDepthStencil(context, DepthStencil, isReverseZ ? 1.f : 0.f);
        NvFlowContextSetViewport(context, &m_offscreenBuffers[0].m_viewport);

        NvFlowDrawParams drawParams = {};
        drawParams.shader = m_volumeRenderDepth[isReverseZ];
        drawParams.rootConstantBuffer = m_constantBuffer;
        drawParams.vs_readOnly[0] = blockListRangeList->allBlockList;
        NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);
        if (params->projectionMatrix.z.w < 0.f)
            drawParams.frontCounterClockwise = 1;
        NvFlowContextDrawIndexedInstanced(context, 0x24, blockListRangeList->allNumBlocks,
                                          &drawParams);
    }

    NvFlowDepthStencilDesc ds_desc;
    NvFlowDepthStencilGetDesc(ds, &ds_desc);

    struct ShaderParams {
        using float2 = NvFlowFloat2;
#include "depthDownsampleShaderParams.h"
    };

    auto mapped = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (mapped) {
        NvFlowFloat2 uvs, uvo;
        uvs.x = ds_desc.viewport.width / float(ds_desc.width);
        uvs.y = ds_desc.viewport.height / float(ds_desc.height);
        uvo.x = ds_desc.viewport.topLeftX / float(ds_desc.width);
        uvo.y = ds_desc.viewport.topLeftY / float(ds_desc.height);

        NvFlowFloat4 uvScale;
        if (params->lensMatchedShading.enabled) {
            uvScale = make_float4(0.5f, -0.5f, 0.5f, 0.5f);
        } else {
            uvScale.x = 0.5f * uvs.x;
            uvScale.y = -0.5f * uvs.x;
            uvScale.z = 0.5f * uvs.x + uvo.x;
            uvScale.w = 0.5f * uvs.y + uvo.y;
        }
        mapped->uvScale = uvScale;

        const auto &viewport = m_offscreenBuffers[0].m_viewport;
        NvFlowFloat4 viewportInvScale;
        viewportInvScale.x = 2.f / viewport.width;
        viewportInvScale.y = -(2.f / viewport.height);
        viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
        viewportInvScale.w = 0.f;
        mapped->viewportInvScale = viewportInvScale;

        NvFlowFloat4 viewportInvOffset;
        viewportInvOffset.x = -2.f / float(viewport.width) * viewport.topLeftX - 1.f;
        viewportInvOffset.y = 2.f / float(viewport.height) * viewport.topLeftY + 1.f;
        viewportInvOffset.z = -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
        viewportInvOffset.w = 1.f;
        mapped->viewportInvOffset = viewportInvOffset;

        NvFlowFloat4 depth_uvScale;
        depth_uvScale.x = 0.5f * m_offscreenBuffers[0].screenPercentX;
        depth_uvScale.y = -0.5f * m_offscreenBuffers[0].screenPercentY;
        depth_uvScale.z = 0.5f * m_offscreenBuffers[0].screenPercentX;
        depth_uvScale.w = 0.5f * m_offscreenBuffers[0].screenPercentY;
        mapped->depth_uvScale = depth_uvScale;

        mapped->depth_reverseZ = make_uint4((uint32_t)isReverseZ, 0, 0, 0);

        auto &lms = params->lensMatchedShading;
        mapped->warpLeft = lms.warpLeft;
        mapped->warpRight = lms.warpRight;
        mapped->warpUp = lms.warpUp;
        mapped->warpDown = lms.warpDown;

        float uvX = lms.sizeLeft / float(lms.sizeLeft + lms.sizeRight);
        float uvY = lms.sizeUp / float(lms.sizeUp + lms.sizeDown);
        mapped->leftX.x = (uvs.x * (1.f + lms.warpLeft)) * uvX;
        mapped->leftX.y = uvs.x * uvX + uvo.x;
        mapped->rightX.x = (uvs.x * (1.f + lms.warpRight)) * (1.f - uvX);
        mapped->rightX.y = uvs.x * uvX + uvo.x;
        mapped->upY.x = (uvs.y * (1.f + lms.warpUp)) * uvY;
        mapped->upY.y = uvs.y * uvY + uvo.y;
        mapped->downY.x = (uvs.y * (1.f + lms.warpDown)) * (1.f - uvY);
        mapped->downY.y = uvs.y * uvY + uvo.y;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    auto depthBuffer = m_offscreenBuffers[0].m_depthMaxBuffer;
    auto RenderTarget = NvFlowColorBufferGetRenderTarget(depthBuffer);
    NvFlowContextSetRenderTarget(context, RenderTarget, nullptr);
    NvFlowContextSetViewport(context, &m_offscreenBuffers[0].m_viewport);

    NvFlowDrawParams drawParams = {};
    drawParams.shader =
        m_depthDownsampleShader[params->lensMatchedShading.enabled][shouldGenDepthMask];
    drawParams.rootConstantBuffer = m_constantBuffer;
    drawParams.ps_readOnly[0] = dsvResource;
    drawParams.ps_readOnly[1] =
        shouldGenDepthMask ? NvFlowDepthBufferGetResource(m_depthMask) : nullptr;

    if (params->multiRes.enabled) {
        generateMultiResMesh(context, &params->multiRes);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferMultiRes,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferMultiRes, 0);
        NvFlowContextDrawIndexedInstanced(context, 0x36, 1, &drawParams);
    } else {
        generateDefaultMesh(context);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
        NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
    }
}

void VolumeRender::extractRenderTargets(NvFlowRenderTargetView **rtv,
                                        NvFlowDepthStencilView **dsv,
                                        NvFlowRenderTarget **rt, NvFlowDepthStencil **ds,
                                        unsigned int *rtv_width, unsigned int *rtv_height,
                                        const NvFlowVolumeRenderParams *params) {
    *rtv = params->renderTargetView;
    *dsv = params->depthStencilView;
    NvFlowRenderTarget *RenderTarget =
        rtv ? NvFlowRenderTargetViewGetRenderTarget(*rtv) : nullptr;
    NvFlowDepthStencil *DepthStencil =
        *dsv ? NvFlowDepthStencilViewGetDepthStencil(*dsv) : nullptr;
    *rt = RenderTarget;
    *ds = DepthStencil;

    NvFlowRenderTargetDesc rt_desc;
    NvFlowDepthStencilDesc ds_desc;
    NvFlowRenderTargetGetDesc(RenderTarget, &rt_desc);
    NvFlowDepthStencilGetDesc(DepthStencil, &ds_desc);

    if (params->multiRes.enabled && params->multiRes.viewport.width > 0.f) {
        if (*rt) {
            rt_desc.viewport.topLeftX = params->multiRes.viewport.topLeftX;
            rt_desc.viewport.topLeftY = params->multiRes.viewport.topLeftY;
            rt_desc.viewport.width = params->multiRes.viewport.width;
            rt_desc.viewport.height = params->multiRes.viewport.height;
        }
        if (*ds) {
            ds_desc.viewport.topLeftX = params->multiRes.viewport.topLeftX;
            ds_desc.viewport.topLeftY = params->multiRes.viewport.topLeftY;
            ds_desc.viewport.width = params->multiRes.viewport.width;
            ds_desc.viewport.height = params->multiRes.viewport.height;
        }
    }

    if (params->lensMatchedShading.enabled &&
        params->lensMatchedShading.viewport.width > 0.f) {
        if (*rt) {
            rt_desc.viewport.topLeftX = params->lensMatchedShading.viewport.topLeftX;
            rt_desc.viewport.topLeftY = params->lensMatchedShading.viewport.topLeftY;
            rt_desc.viewport.width = params->lensMatchedShading.viewport.width;
            rt_desc.viewport.height = params->lensMatchedShading.viewport.height;
        }
        if (*ds) {
            ds_desc.viewport.topLeftX = params->lensMatchedShading.viewport.topLeftX;
            ds_desc.viewport.topLeftY = params->lensMatchedShading.viewport.topLeftY;
            ds_desc.viewport.width = params->lensMatchedShading.viewport.width;
            ds_desc.viewport.height = params->lensMatchedShading.viewport.height;
        }
    }

    if (*rt) {
        *rtv_width = int(rt_desc.viewport.width);
        *rtv_height = int(rt_desc.viewport.height);
    } else {
        *rtv_width = int(ds_desc.viewport.width);
        *rtv_height = int(ds_desc.viewport.height);
    }

    if (params->multiRes.enabled && params->multiRes.viewport.width > 0.f) {
        *rtv_width = int(params->multiRes.nonMultiResWidth);
        *rtv_height = int(params->multiRes.nonMultiResHeight);
    }
    if (params->lensMatchedShading.enabled &&
        params->lensMatchedShading.viewport.width > 0.f) {
        *rtv_width = int(params->lensMatchedShading.nonLMSWidth);
        *rtv_height = int(params->lensMatchedShading.nonLMSHeight);
    }
    NvFlowRenderTargetSetViewport(*rt, &rt_desc.viewport);
    NvFlowDepthStencilSetViewport(*ds, &ds_desc.viewport);
}

uint32_t VolumeRender::genBlockDistKey(uint32_t val, const NvFlowUint4 &blockDim,
                                       const NvFlowFloat4 &rayOriginVirtual) {
    NvFlowUint3 vBlockIdx = NvFlow_tableVal_to_coord(val);
    NvFlowFloat3 dr;
    dr.x = float(blockDim.x * vBlockIdx.x) + 0.5f * float(blockDim.x) - rayOriginVirtual.x;
    dr.y = float(blockDim.y * vBlockIdx.y) + 0.5f * float(blockDim.y) - rayOriginVirtual.y;
    dr.z = float(blockDim.z * vBlockIdx.z) + 0.5f * float(blockDim.z) - rayOriginVirtual.z;
    return NvFlow_flip(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
}

uint32_t VolumeRender::genBlockDistKeyCoarse(uint32_t val, const NvFlowUint4 &blockDim,
                                             const NvFlowFloat4 &rayOriginVirtual) {
    NvFlowUint3 vBlockIdx = NvFlow_tableVal_to_coord(val);
    vBlockIdx.x = vBlockIdx.x & ~3u;
    vBlockIdx.y = vBlockIdx.y & ~3u;
    vBlockIdx.z = vBlockIdx.z & ~3u;
    NvFlowFloat3 dr;
    dr.x =
        float(blockDim.x * vBlockIdx.x) + 0.5f * float(4 * blockDim.x) - rayOriginVirtual.x;
    dr.y =
        float(blockDim.y * vBlockIdx.y) + 0.5f * float(4 * blockDim.y) - rayOriginVirtual.y;
    dr.z =
        float(blockDim.z * vBlockIdx.z) + 0.5f * float(4 * blockDim.z) - rayOriginVirtual.z;
    return NvFlow_flip(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
}

uint32_t VolumeRender::genBlockIdxKeyCoarse(uint32_t val, const NvFlowUint4 &blockDim) {
    NvFlowUint3 vBlockIdx = NvFlow_tableVal_to_coord(val);
    return (4 * (vBlockIdx.x / 4)) | ((vBlockIdx.y / 4) << 12) |
           ((vBlockIdx.z / 4) << 22);
}

VolumeRender::BlockListRangeList VolumeRender::generateBlockListRangeListLayered(
    NvFlowContext *context, NvFlowUint2 *layeredBlockListCPU, uint32_t layeredNumBlocks,
    uint32_t numLayerViews, const NvFlowUint4 &blockDim,
    const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual) {
    BlockListRangeList rangeList;

    if (layeredNumBlocks) {
        NvFlowContextProfileItemBegin(context, L"sortCPU");
        for (int passID = 0; passID < 4; ++passID) {
            auto keyVal = m_sortCPU->getBuffer().keyVal;

            switch (passID) {
                case 0:
                    for (uint32_t i = 0; i < layeredNumBlocks; ++i) {
                        auto val = layeredBlockListCPU[i];
                        keyVal[i] = NvFlowUint2{val.y, i};
                    }
                    break;
                case 1:
                    for (uint32_t i = 0; i < layeredNumBlocks; ++i) {
                        auto srcKeyVal = keyVal[i];
                        uint32_t key = genBlockDistKey(layeredBlockListCPU[srcKeyVal.y].x,
                                                       blockDim, rayOriginVirtual);
                        keyVal[i] = NvFlowUint2{key, srcKeyVal.y};
                    }
                    break;
                case 2:
                    for (uint32_t i = 0; i < layeredNumBlocks; ++i) {
                        auto srcKeyVal = keyVal[i];
                        uint32_t key = genBlockIdxKeyCoarse(
                            layeredBlockListCPU[srcKeyVal.y].x, blockDim);
                        keyVal[i] = NvFlowUint2{key, srcKeyVal.y};
                    }
                    break;
                case 3:
                    for (uint32_t i = 0; i < layeredNumBlocks; ++i) {
                        auto srcKeyVal = keyVal[i];
                        uint32_t key = genBlockDistKeyCoarse(
                            layeredBlockListCPU[srcKeyVal.y].x, blockDim, rayOriginVirtual);
                        keyVal[i] = NvFlowUint2{key, srcKeyVal.y};
                    }
                    break;
            }

            RadixSortCPUParams sortParams;
            sortParams.numElements = layeredNumBlocks;
            m_sortCPU->sort(context, &sortParams);
        }
        NvFlowContextProfileItemEnd(context);

        NvFlowContextProfileItemBegin(context, L"layeredBlockList");
        auto keyVal = m_sortCPU->getBuffer().keyVal; // v53
        m_blockListLayers.clear();
        m_blockListRanges.clear();
        m_blockListSorted.clear();
        m_layerCounters.resize(numLayerViews);

        uint32_t dstIdx = 0;             // v46
        uint32_t coarseBlockIdx = 0; // v47
        NvFlowUint2 blockListVal = {};    // v48
        uint32_t currentCoarseBlockIdx = 0; // v49
        uint32_t currentValue = 0;          // v50
        uint32_t startBlockListIdx = 0; // v51
        uint32_t idx = 0; // v52

        auto genKey = [this, &blockListVal, layeredBlockListCPU, keyVal, &idx,
                       &coarseBlockIdx, blockDim]() {
            uint32_t y = keyVal[idx].y;
            NvFlowUint2 val = layeredBlockListCPU[y];
            blockListVal = val;
            coarseBlockIdx = genBlockIdxKeyCoarse(blockListVal.x, blockDim);
        };

        auto resetLayerCounters = [numLayerViews, this, &currentCoarseBlockIdx,
                                   &coarseBlockIdx, &startBlockListIdx]() {
            for (uint32_t counterIdx = 0; counterIdx < numLayerViews; ++counterIdx)
                m_layerCounters[counterIdx] = 0;

            currentCoarseBlockIdx = coarseBlockIdx;
            startBlockListIdx = m_blockListSorted.size();
        };

        auto coarseFlush = [this, numLayerViews, &startBlockListIdx]() {
            uint32_t rangeAllocIdx = m_blockListRanges.allocateBack();
            auto &range = m_blockListRanges[rangeAllocIdx];
            uint32_t numNonZero = 0;
            for (uint32_t counterIdx = 0; counterIdx < numLayerViews; ++counterIdx) {
                if (m_layerCounters[counterIdx])
                    ++numNonZero;
            }

            range.layerListStart = m_blockListLayers.size();
            range.layerListCount = numNonZero;
            for (uint32_t i = 0; i < numLayerViews; ++i) {
                if (m_layerCounters[i]) {
                    auto layerIdx = m_blockListLayers.allocateBack();
                    m_blockListLayers[layerIdx] = i;
                }
            }

            range.blockListStart = startBlockListIdx;
            range.blockListCount = m_blockListSorted.size() - startBlockListIdx;
            range.blockList = 0;
        };

        auto pushBlock = [&dstIdx, this, &blockListVal, &currentValue]() {
            dstIdx = m_blockListSorted.allocateBack();
            m_blockListSorted[dstIdx] = blockListVal.x;
            currentValue = blockListVal.x;
        };

        genKey();
        resetLayerCounters();
        pushBlock();

        ++idx;
        while (idx < layeredNumBlocks) {
            genKey();
            if (coarseBlockIdx != currentCoarseBlockIdx) {
                coarseFlush();
                resetLayerCounters();
            }
            if (blockListVal.y < m_layerCounters.size()) {
                ++m_layerCounters[blockListVal.y];
            }

            if (blockListVal.x != currentValue)
                pushBlock();

            ++idx;
        }

        coarseFlush();

        auto mappedBlockIdx = (uint32_t *)NvFlowBufferMap(context, m_blockListUpload);
        if (mappedBlockIdx) {
            for (uint32_t i = 0; i < m_blockListSorted.size(); ++i)
                mappedBlockIdx[i] = m_blockListSorted[i];
            NvFlowBufferUnmap(context, m_blockListUpload);
        }

        for (uint32_t rangeIdx = 0; rangeIdx < m_blockListRanges.size(); ++rangeIdx) {
            auto Resource = NvFlowBufferGetResource(m_blockListUpload);
            m_blockListRanges[rangeIdx].blockList = Resource;
        }

        rangeList.layerLists = m_blockListLayers.data();
        rangeList.numLayerLists = m_blockListLayers.size();
        rangeList.ranges = m_blockListRanges.data();
        rangeList.numRanges = m_blockListRanges.size();
        rangeList.allBlockList = NvFlowBufferGetResource(m_blockListUpload);
        rangeList.allNumBlocks = m_blockListSorted.size();

        NvFlowContextProfileItemEnd(context);
    } else {
        ZeroMemory(&rangeList, sizeof(rangeList));
    }
    return rangeList;
}

VolumeRender::BlockListRangeList VolumeRender::generateBlockListRangeListSingle(
    NvFlowContext *context, unsigned int layerIdx, unsigned int numBlocks,
    NvFlowResource *blockList, const NvFlowUint4 &blockDim,
    const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual) {
    BlockListRangeList rangeList;

    uint32_t numSortBlocks = (numBlocks + 1023) >> 10;
    auto sortBuffer = m_sort->getBuffer();

    struct ShaderParams {
        NvFlowUint4 blockDim;
        NvFlowFloat4 rayOriginVirtual;
        NvFlowFloat4 rayForwardDirVirtual;
        NvFlowUint4 numBlocks;
    };

    auto mapped = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (mapped) {
        mapped->blockDim = blockDim;
        mapped->rayOriginVirtual = rayOriginVirtual;
        mapped->rayForwardDirVirtual = rayForwardDirVirtual;
        mapped->numBlocks = make_uint4(numBlocks, 0, 0, 0);
        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    NvFlowDispatchParams dparams = {};
    dparams.shader = m_sortShader;
    dparams.gridDim = make_dim(((numSortBlocks << 10) + 255) / 256, 1, 1);
    dparams.rootConstantBuffer = m_constantBuffer;
    dparams.readOnly[0] = blockList;
    dparams.readWrite[0] = NvFlowBufferGetResourceRW(sortBuffer.key);
    dparams.readWrite[1] = NvFlowBufferGetResourceRW(sortBuffer.val);
    NvFlowContextDispatch(context, &dparams);

    RadixSortParams sortParams;
    sortParams.numSortBlocks = numSortBlocks;
    m_sort->sort(context, &sortParams);

    sortBuffer = m_sort->getBuffer();
    auto sortedBlockList = NvFlowBufferGetResource(sortBuffer.val);

    m_blockListLayers.resize(1);
    m_blockListLayers[0] = layerIdx;
    m_blockListRanges.resize(1);
    BlockListRange range;
    range.layerListStart = 0;
    range.layerListCount = 1;
    range.blockListStart = 0;
    range.blockListCount = numBlocks;
    range.blockList = sortedBlockList;
    m_blockListRanges[0] = range;

    rangeList.layerLists = m_blockListLayers.data();
    rangeList.numLayerLists = 1;
    rangeList.ranges = m_blockListRanges.data();
    rangeList.numRanges = 1;
    rangeList.allBlockList = sortedBlockList;
    rangeList.allNumBlocks = numBlocks;

    return rangeList;
}

void VolumeRender::generateDefaultMesh(NvFlowContext *context) {}

void VolumeRender::generateMultiResMesh(NvFlowContext *context,
                                        const NvFlowVolumeRenderMultiResParams *multiRes) {
    float region01x =
        multiRes->densityScaleX[0] * (multiRes->centerX - 0.5f * multiRes->centerWidth);
    float region12x = multiRes->densityScaleX[1] * multiRes->centerWidth;
    float region23x = multiRes->densityScaleX[2] *
                      (1.f - (multiRes->centerX + 0.5f * multiRes->centerWidth));
    float region01y =
        multiRes->densityScaleY[0] * (multiRes->centerY - 0.5f * multiRes->centerHeight);
    float region12y = multiRes->densityScaleY[1] * multiRes->centerHeight;
    float region23y = multiRes->densityScaleY[2] *
                      (1.f - (multiRes->centerY + 0.5f * multiRes->centerHeight));

    float regionSumX = region01x + region12x + region23x;
    float regionSumY = region01y + region12y + region23y;

    float pos1x = region01x / regionSumX;
    float pos2x = (region01x + region12x) / regionSumX;
    float pos1y = region01y / regionSumX;
    float pos2y = (region01y + region12y) / regionSumY;

    float uv1x = multiRes->centerX - 0.5f * multiRes->centerWidth;
    float uv2x = multiRes->centerX + 0.5f * multiRes->centerWidth;
    float uv1y = multiRes->centerY - 0.5f * multiRes->centerHeight;
    float uv2y = multiRes->centerY + 0.5f * multiRes->centerHeight;

    pos1x = 2.f * pos1x - 1.f;
    pos2x = 2.f * pos2x - 1.f;
    pos1y = -2.f * pos1y + 1.f;
    pos2y = -2.f * pos2y + 1.f;
    uv1x = 2.f * uv1x - 1.f;
    uv2x = 2.f * uv2x - 1.f;
    uv1y = -2.f * uv1y + 1.f;
    uv2y = -2.f * uv2y + 1.f;

    NvFlowFloat4 pts[] = {
        {-1.f, 1.f, -1.f, 1.f},     {pos1x, 1.f, uv1x, 1.f},    {pos2x, 1.f, uv2x, 1.f},
        {1.f, 1.f, 1.f, 1.f},       {-1.f, pos1y, -1.f, uv1y},  {pos1x, pos1y, uv1x, uv1y},
        {pos2x, pos1y, uv2x, uv1y}, {1.f, pos1y, 1.f, uv1y},    {-1.f, pos2y, -1.f, uv2y},
        {pos1x, pos2y, uv1x, uv2y}, {pos2x, pos2y, uv2x, uv2y}, {1.f, pos2y, 1.f, uv2y},
        {-1.f, -1.f, -1.f, -1.f},   {pos1x, -1.f, uv1x, -1.f},  {pos2x, -1.f, uv2x, -1.f},
        {1.f, -1.f, 1.f, -1.f},
    };

    NvFlowUint indices[] = {
        4,  0, 5,  5,  0, 1, 5,  1, 6,  6,  1, 2,  6,  2,  7,  7,  2,  3,
        8,  4, 9,  9,  4, 5, 9,  5, 10, 10, 5, 6,  10, 6,  11, 11, 6,  7,
        12, 8, 13, 13, 8, 9, 13, 9, 14, 14, 9, 10, 14, 10, 15, 15, 10, 11,
    };

    auto vbdata = NvFlowVertexBufferMap(context, m_compositeVertexBufferMultiRes);
    if (vbdata) {
        memcpy(vbdata, pts, sizeof(pts));
        NvFlowVertexBufferUnmap(context, m_compositeVertexBufferMultiRes);
    }

    auto ibdata = NvFlowIndexBufferMap(context, m_compositeIndexBufferMultiRes);
    if (ibdata) {
        memcpy(ibdata, indices, sizeof(indices));
        NvFlowIndexBufferUnmap(context, m_compositeIndexBufferMultiRes);
    }
}

void VolumeRender::rayMarch(
    NvFlowContext *context, NvFlowRenderTarget *dstTarget, NvFlowDepthStencil *rayMarchMask,
    bool clearRenderTarget, NvFlowResource *depthMaxBuffer, NvFlowResource *depthMinBuffer,
    const NvFlowViewport &offscreenViewport, float screenPercentX, float screenPercentY,
    const BlockListRangeList *blockListRangeList,
    const NvFlowGridExportHandle &exportHandle, const NvFlowFloat4x4 &modelViewProjT,
    const NvFlowFloat4 &depthMaxInvTransform, const NvFlowFloat4 &depthMinInvTransform,
    const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual,
    const NvFlowShaderLinearParams *linearParams, const NvFlowDim &gridDim,
    const bool enableVTR, const NvFlowVolumeRenderParams *params) {
    auto renderMode = params->renderMode;
    NvFlowGridExportLayeredView exportLayeredView = {};
    NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

    NvFlowContextSetRenderTarget(context, dstTarget, rayMarchMask);
    if (clearRenderTarget) {
        NvFlowFloat4 color = make_float4(0.f, 0.f, 0.f, 1.f);
        NvFlowContextClearRenderTarget(context, dstTarget, color);
    }
    NvFlowContextSetViewport(context, &offscreenViewport);

    for (uint32_t rangeIdx = 0; rangeIdx < blockListRangeList->numRanges; ++rangeIdx) {
        auto &blockListRange = blockListRangeList->ranges[rangeIdx];
        uint32_t totalRenderMaterials = 0;

        for (uint32_t perLayerIdx = 0; perLayerIdx < blockListRange.layerListCount;
             ++perLayerIdx) {
            auto layerIdx =
                blockListRangeList->layerLists[perLayerIdx + blockListRange.layerListStart];
            NvFlowGridExportLayerView exportLayerView;
            NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);
            totalRenderMaterials += params->materialPool->getRenderMaterialHandles(
                0, 0, exportLayerView.mapping.material);
        }

        m_drawRenderMaterials.resize(totalRenderMaterials);
        m_drawLayerIndices.resize(totalRenderMaterials);

        uint32_t dstMatIdx = 0;
        for (uint32_t i = 0; i < blockListRange.layerListCount; ++i) {
            auto layerIdx =
                blockListRangeList->layerLists[i + blockListRange.layerListStart];
            NvFlowGridExportLayerView exportLayerView;
            NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);
            uint32_t startDstMatIdx = dstMatIdx;
            uint32_t maxNumMaterials = totalRenderMaterials - dstMatIdx;

            uint32_t numMaterials = params->materialPool->getRenderMaterialHandles(
                &m_drawRenderMaterials[dstMatIdx], maxNumMaterials,
                exportLayerView.mapping.material);

            dstMatIdx += numMaterials;
            for (uint32_t idx = startDstMatIdx; idx < dstMatIdx; ++idx) {
                m_drawLayerIndices[idx] = layerIdx;
            }
        }

        uint32_t numBatches = (totalRenderMaterials + 3) / 4;
        for (uint32_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            uint32_t drawRenderMatIdx = 4 * batchIdx;
            uint32_t drawRenderMatCount;
            if (batchIdx == numBatches - 1)
                drawRenderMatCount = totalRenderMaterials - drawRenderMatIdx;
            else
                drawRenderMatCount = 4;

            NvFlowUint layerIndices[4] = {};
            NvFlowGridExportLayerView exportLayerViews[4] = {};
            NvFlowRenderMaterialHandle renderMaterials[4] = {};
            NvFlowRenderMaterialParams materials[4] = {};
            NvFlowResource *colorMaps[4] = {};
            NvFlowResource *blockTables[4] = {};
            NvFlowResource *datas[4] = {};

            for (uint32_t j = 0; j < drawRenderMatCount; ++j) {
                uint32_t layerIdx = m_drawLayerIndices[j + drawRenderMatIdx];
                layerIndices[j] = layerIdx;

                NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerViews[j]);

                auto &renderMaterial = m_drawRenderMaterials[j + drawRenderMatIdx];
                renderMaterials[j] = renderMaterial;

                auto material = params->materialPool->getMaterialParams(renderMaterial);
                materials[j] = *material;
                auto colorMap = params->materialPool->getColorMap(renderMaterial);
                colorMaps[j] = colorMap;

                blockTables[j] = exportLayerViews[j].mapping.blockTable;
                datas[j] = exportLayerViews[j].data;
            }

            auto mapped = (VolumeRenderShaderParams *)NvFlowConstantBufferMap(
                context, m_constantBuffer);
            if (mapped) {
                mapped->modelViewProj = modelViewProjT;
                auto &viewport = offscreenViewport;
                NvFlowFloat4 viewportInvScale;
                viewportInvScale.x = 2.f / viewport.width;
                viewportInvScale.y = -2.f / viewport.height;
                viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
                viewportInvScale.w = 0.f;
                mapped->viewportInvScale = viewportInvScale;
                NvFlowFloat4 viewportInvOffset;
                viewportInvOffset.x = -2.f / viewport.width * viewport.topLeftX - 1.f;
                viewportInvOffset.y = 2.f / viewport.height * viewport.topLeftY + 1.f;
                viewportInvOffset.z =
                    -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
                viewportInvOffset.w = 1.f;
                mapped->viewportInvOffset = viewportInvOffset;

                mapped->valueParams = *linearParams;
                (NvFlowDim &)mapped->vGridDim = gridDim;
                mapped->vGridDim.w = 0.f;
                (NvFlowFloat3 &)mapped->vGridDimInv = 1.f / make_float3(gridDim);
                mapped->vGridDimInv.w = 0.f;

                mapped->rayOriginVirtual = rayOriginVirtual;
                mapped->rayForwardDirVirtual = rayForwardDirVirtual;

                mapped->blockListStartCount.x = blockListRange.blockListStart;
                mapped->blockListStartCount.y = blockListRange.blockListCount;
                mapped->blockListStartCount.z = 0.f;
                mapped->blockListStartCount.w = 0.f;
                mapped->renderMode = make_uint4(renderMode);

                mapped->eyePad = 0.f;
                mapped->eyeFade = 16.f;
                mapped->eyeFadeInv = 1.f / 16.f;
                mapped->eyeFadeOffset = -16.f;

                NvFlowFloat4 depthUVTransform;
                depthUVTransform.x = 0.5f * screenPercentX;
                depthUVTransform.y = -0.5f * screenPercentY;
                depthUVTransform.z = 0.5f * screenPercentX;
                depthUVTransform.w = 0.5f * screenPercentY;
                mapped->depthUVTransform = depthUVTransform;
                mapped->depthMaxInvTransform = depthMaxInvTransform;
                mapped->depthMinInvTransform = depthMinInvTransform;

                mapped->alphaScale_layer0 = materials[0].alphaScale;
                mapped->alphaScale_layer1 = materials[1].alphaScale;
                mapped->alphaScale_layer2 = materials[2].alphaScale;
                mapped->alphaScale_layer3 = materials[3].alphaScale;

                mapped->additiveFactor_layer0 = materials[0].additiveFactor;
                mapped->additiveFactor_layer1 = materials[1].additiveFactor;
                mapped->additiveFactor_layer2 = materials[2].additiveFactor;
                mapped->additiveFactor_layer3 = materials[3].additiveFactor;

                mapped->alphaBias_layer0 = materials[0].alphaBias;
                mapped->alphaBias_layer1 = materials[1].alphaBias;
                mapped->alphaBias_layer2 = materials[2].alphaBias;
                mapped->alphaBias_layer3 = materials[3].alphaBias;

                mapped->intensityBias_layer0 = materials[0].intensityBias;
                mapped->intensityBias_layer1 = materials[1].intensityBias;
                mapped->intensityBias_layer2 = materials[2].intensityBias;
                mapped->intensityBias_layer3 = materials[3].intensityBias;

                mapped->colorMapCompMask_layer0 = materials[0].colorMapCompMask;
                mapped->colorMapCompMask_layer1 = materials[1].colorMapCompMask;
                mapped->colorMapCompMask_layer2 = materials[2].colorMapCompMask;
                mapped->colorMapCompMask_layer3 = materials[3].colorMapCompMask;

                NvFlowFloat4 *colorMapRange_layers[] = {
                    &mapped->colorMapRange_layer0,
                    &mapped->colorMapRange_layer1,
                    &mapped->colorMapRange_layer2,
                    &mapped->colorMapRange_layer3,
                };
                NvFlowFloat4 colorMapRange;
                for (uint32_t k = 0; k < 4; ++k) {
                    colorMapRange.x = materials[k].colorMapMinX;
                    colorMapRange.y =
                        1.f / (materials[k].colorMapMaxX - materials[k].colorMapMinX);
                    colorMapRange.z = materials[k].colorMapMinX;
                    colorMapRange.w = materials[k].colorMapMaxX;
                    *colorMapRange_layers[k] = colorMapRange;
                }

                mapped->alphaCompMask_layer0 = materials[0].alphaCompMask;
                mapped->alphaCompMask_layer1 = materials[1].alphaCompMask;
                mapped->alphaCompMask_layer2 = materials[2].alphaCompMask;
                mapped->alphaCompMask_layer3 = materials[3].alphaCompMask;

                mapped->intensityCompMask_layer0 = materials[0].intensityCompMask;
                mapped->intensityCompMask_layer1 = materials[1].intensityCompMask;
                mapped->intensityCompMask_layer2 = materials[2].intensityCompMask;
                mapped->intensityCompMask_layer3 = materials[3].intensityCompMask;

                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }

            bool depthTest = rayMarchMask != 0;
            uint32_t volumeRenderShader = 0;
            switch (renderMode) {
                case eNvFlowVolumeRenderMode_colormap:
                    volumeRenderShader = 1;
                    break;
                case eNvFlowVolumeRenderMode_raw:
                    volumeRenderShader = 3;
                    break;
                case eNvFlowVolumeRenderMode_rainbow:
                case eNvFlowVolumeRenderMode_debug:
                    volumeRenderShader = 2;
                    break;
                default:
                    volumeRenderShader = 0;
                    break;
            }

            NvFlowDrawParams drawParams = {};
            drawParams.shader =
                m_volumeRender[depthTest][drawRenderMatCount - 1][volumeRenderShader];
            drawParams.rootConstantBuffer = m_constantBuffer;
            drawParams.vs_readOnly[0] = blockListRange.blockList;
            drawParams.vs_readOnly[1] = blockTables[0];
            if (drawRenderMatCount > 1)
                drawParams.vs_readOnly[2] = blockTables[1];
            if (drawRenderMatCount > 2)
                drawParams.vs_readOnly[3] = blockTables[2];
            if (drawRenderMatCount > 3)
                drawParams.vs_readOnly[4] = blockTables[4];
            drawParams.ps_readOnly[0] = depthMaxBuffer;
            drawParams.ps_readOnly[1] = depthMinBuffer;
            drawParams.ps_readOnly[2] = datas[0];
            drawParams.ps_readOnly[3] = colorMaps[0];
            if (drawRenderMatCount > 1) {
                drawParams.ps_readOnly[4] = datas[1];
                drawParams.ps_readOnly[5] = colorMaps[1];
            }
            if (drawRenderMatCount > 2) {
                drawParams.ps_readOnly[6] = datas[2];
                drawParams.ps_readOnly[7] = colorMaps[2];
            }
            if (drawRenderMatCount > 3) {
                drawParams.ps_readOnly[8] = datas[3];
                drawParams.ps_readOnly[9] = colorMaps[3];
            }
            NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
            NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);

            if (params->projectionMatrix.z.w < 0.f)
                drawParams.frontCounterClockwise = 1;

            NvFlowContextDrawIndexedInstanced(context, 0x24, blockListRange.blockListCount,
                                              &drawParams);
        }
    }
}

void VolumeRender::rayMarchDepthEstimate(
    NvFlowContext *context, NvFlowRenderTarget *dstTarget,
    const NvFlowViewport &offscreenViewport, float screenPercentX, float screenPercentY,
    const NvFlowGridExportHandle &exportHandle, const NvFlowFloat4x4 &modelViewProjT,
    const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual,
    const NvFlowShaderLinearParams *linearParams, const NvFlowDim &gridDim, bool enableVTR,
    const NvFlowVolumeRenderParams *params) {
    auto renderMode = params->renderMode;
    NvFlowGridExportLayeredView exportLayeredView = {};
    NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);
    NvFlowFloat4 color = make_float4(0.f, 0.f, 0.f, 1.f);
    NvFlowContextSetRenderTarget(context, dstTarget, nullptr);
    NvFlowContextClearRenderTarget(context, dstTarget, color);
    NvFlowContextSetViewport(context, &offscreenViewport);

    for (uint32_t layerIdx = 0; layerIdx < exportHandle.numLayerViews; ++layerIdx) {
        NvFlowGridExportLayeredView dst = {};
        NvFlowGridExportGetLayeredView(exportHandle, &dst);
        NvFlowGridExportLayerView exportLayerView = {};
        NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);
        uint32_t numMaterials = params->materialPool->getRenderMaterialHandles(
            0, 0, exportLayerView.mapping.material);

        m_depthEstimateRenderMaterials.resize(numMaterials);
        params->materialPool->getRenderMaterialHandles(
            m_depthEstimateRenderMaterials.data(), numMaterials,
            exportLayerView.mapping.material);

        for (uint32_t materialIdx = 0; materialIdx < numMaterials; ++materialIdx) {
            auto &material = m_depthEstimateRenderMaterials[materialIdx];
            auto &materialParams = *params->materialPool->getMaterialParams(material);
            NvFlowResource *colorMap = params->materialPool->getColorMap(material);
            auto blockList = exportLayerView.mapping.blockList;
            auto blockTable = exportLayerView.mapping.blockTable;
            auto data = exportLayerView.data;

            uint32_t blockListStart = 0;
            uint32_t blockListCount = exportLayerView.mapping.numBlocks;

            auto mapped = (VolumeRenderShaderParams *)NvFlowConstantBufferMap(
                context, m_constantBuffer);
            if (mapped) {
                mapped->modelViewProj = modelViewProjT;
                auto &viewport = offscreenViewport;
                NvFlowFloat4 viewportInvScale;
                viewportInvScale.x = 2.f / viewport.width;
                viewportInvScale.y = -2.f / viewport.height;
                viewportInvScale.z = 1.f / (viewport.maxDepth - viewport.minDepth);
                viewportInvScale.w = 0.f;
                mapped->viewportInvScale = viewportInvScale;
                NvFlowFloat4 viewportInvOffset;
                viewportInvOffset.x = -2.f / viewport.width * viewport.topLeftX - 1.f;
                viewportInvOffset.y = 2.f / viewport.height * viewport.topLeftY + 1.f;
                viewportInvOffset.z =
                    -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
                viewportInvOffset.w = 1.f;
                mapped->viewportInvOffset = viewportInvOffset;

                mapped->valueParams = *linearParams;
                (NvFlowDim &)mapped->vGridDim = gridDim;
                mapped->vGridDim.w = 0.f;
                (NvFlowFloat3 &)mapped->vGridDimInv = 1.f / make_float3(gridDim);
                mapped->vGridDimInv.w = 0.f;

                mapped->rayOriginVirtual = rayOriginVirtual;
                mapped->rayForwardDirVirtual = rayForwardDirVirtual;

                mapped->blockListStartCount.x = blockListStart;
                mapped->blockListStartCount.y = blockListCount;
                mapped->blockListStartCount.z = 0.f;
                mapped->blockListStartCount.w = 0.f;
                mapped->renderMode = make_uint4(renderMode);

                mapped->eyePad = 0.f;
                mapped->eyeFade = 16.f;
                mapped->eyeFadeInv = 1.f / 16.f;
                mapped->eyeFadeOffset = -16.f;

                NvFlowFloat4 depthUVTransform;
                depthUVTransform.x = 0.5f * screenPercentX;
                depthUVTransform.y = -0.5f * screenPercentY;
                depthUVTransform.z = 0.5f * screenPercentX;
                depthUVTransform.w = 0.5f * screenPercentY;
                mapped->depthUVTransform = depthUVTransform;
                mapped->depthMaxInvTransform = make_float4(0.f);
                mapped->depthMinInvTransform = make_float4(0.f);

                mapped->alphaScale_layer0 = materialParams.alphaScale;
                mapped->alphaScale_layer1 = materialParams.alphaScale;
                mapped->alphaScale_layer2 = materialParams.alphaScale;
                mapped->alphaScale_layer3 = materialParams.alphaScale;
                mapped->additiveFactor_layer0 = materialParams.additiveFactor;
                mapped->additiveFactor_layer1 = materialParams.additiveFactor;
                mapped->additiveFactor_layer2 = materialParams.additiveFactor;
                mapped->additiveFactor_layer3 = materialParams.additiveFactor;
                mapped->alphaBias_layer0 = materialParams.alphaBias;
                mapped->alphaBias_layer1 = materialParams.alphaBias;
                mapped->alphaBias_layer2 = materialParams.alphaBias;
                mapped->alphaBias_layer3 = materialParams.alphaBias;
                mapped->intensityBias_layer0 = materialParams.intensityBias;
                mapped->intensityBias_layer1 = materialParams.intensityBias;
                mapped->intensityBias_layer2 = materialParams.intensityBias;
                mapped->intensityBias_layer3 = materialParams.intensityBias;

                mapped->colorMapCompMask_layer0 = materialParams.colorMapCompMask;
                mapped->colorMapCompMask_layer1 = materialParams.colorMapCompMask;
                mapped->colorMapCompMask_layer2 = materialParams.colorMapCompMask;
                mapped->colorMapCompMask_layer3 = materialParams.colorMapCompMask;

                NvFlowFloat4 *colorMapRange_layers[] = {
                    &mapped->colorMapRange_layer0,
                    &mapped->colorMapRange_layer1,
                    &mapped->colorMapRange_layer2,
                    &mapped->colorMapRange_layer3,
                };
                NvFlowFloat4 colorMapRange;
                for (uint32_t perLayerIdx = 0; perLayerIdx < 4; ++perLayerIdx) {
                    colorMapRange.x = materialParams.colorMapMinX;
                    colorMapRange.y =
                        1.f / (materialParams.colorMapMaxX - materialParams.colorMapMinX);
                    colorMapRange.z = materialParams.colorMapMinX;
                    colorMapRange.w = materialParams.colorMapMaxX;
                    *colorMapRange_layers[perLayerIdx] = colorMapRange;
                }

                mapped->alphaCompMask_layer0 = materialParams.alphaCompMask;
                mapped->alphaCompMask_layer1 = materialParams.alphaCompMask;
                mapped->alphaCompMask_layer2 = materialParams.alphaCompMask;
                mapped->alphaCompMask_layer3 = materialParams.alphaCompMask;

                mapped->intensityCompMask_layer0 = materialParams.intensityCompMask;
                mapped->intensityCompMask_layer1 = materialParams.intensityCompMask;
                mapped->intensityCompMask_layer2 = materialParams.intensityCompMask;
                mapped->intensityCompMask_layer3 = materialParams.intensityCompMask;

                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }

            NvFlowDrawParams drawParams = {};
            drawParams.shader = m_volumeRenderDepthEstimate;
            drawParams.rootConstantBuffer = m_constantBuffer;
            drawParams.vs_readOnly[0] = blockList;
            drawParams.vs_readOnly[1] = blockTable;
            drawParams.ps_readOnly[2] = data;
            drawParams.ps_readOnly[3] = colorMap;
            NvFlowContextSetVertexBuffer(context, m_vertexBuffer, 0x10u, 0);
            NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);
            if (params->projectionMatrix.z.w < 0.0)
                drawParams.frontCounterClockwise = 1;
            NvFlowContextDrawIndexedInstanced(context, 0x24u, blockListCount, &drawParams);
        }
    }
}

void VolumeRender::rayMarchMultiRes(
    NvFlowContext *context, const BlockListRangeList *blockListRangeList,
    const NvFlowGridExportHandle &exportHandle, const NvFlowFloat4x4 &modelViewProjT,
    const NvFlowFloat4 &depthInvTransform, const NvFlowFloat4 &rayOriginVirtual,
    const NvFlowFloat4 &rayForwardDirVirtual, const NvFlowShaderLinearParams *linearParams,
    const NvFlowDim &gridDim, bool enableVTR, const NvFlowFloat4x4 &projectionMatrixInv,
    const NvFlowVolumeRenderParams *params) {
    uint32_t numLevels = 1;
    if (params->multiResRayMarch > eNvFlowMultiResRayMarchDisabled &&
        params->multiResRayMarch <= eNvFlowMultiResRayMarch16x16) {
        numLevels = params->multiResRayMarch + 1;
    }

    for (uint32_t i = 0; i < numLevels; ++i) {
        auto &offscreenBuf = m_offscreenBuffers[i];
        offscreenBuf.m_tmax = VolumeRenderUtils::compute_tmax(
            offscreenBuf.m_viewport.width, projectionMatrixInv,
            params->multiResSamplingScale);
    }

    struct ShaderParams {
        using float4 = NvFlowFloat4;
        float4 uvScale;
        float4 depthInvTransform;
        float tmax;
        float tbias;
        float pad0;
        float pad1;
    };

    struct UpsampleShaderParams {
        NvFlowFloat4 uvScale;
    };

    struct RayMarchShaderParams {
        NvFlowFloat4 uvScale;
        NvFlowFloat4 depthInvTransform;
    };

    NvFlowRenderTarget *RenderTarget;

    for (uint32_t i = 1; i < numLevels; ++i) {
        auto &dstBuf = m_offscreenBuffers[i];
        auto &srcBuf = m_offscreenBuffers[i - 1];
        auto mapped = (ShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            NvFlowFloat4 uvScale;
            uvScale.x = 0.5f * dstBuf.screenPercentX;
            uvScale.y = -0.5f * dstBuf.screenPercentY;
            uvScale.x = 0.5f * dstBuf.screenPercentX;
            uvScale.y = 0.5f * dstBuf.screenPercentY;
            mapped->uvScale = uvScale;
            if (i == 1)
                mapped->depthInvTransform = depthInvTransform;
            else {
                mapped->depthInvTransform = make_float4(1.f, 0.f, 0.f, 1.f);
            }
            mapped->tmax = dstBuf.m_tmax;
            mapped->tbias = 0.f;
            mapped->pad0 = 0.f;
            mapped->pad1 = 0.f;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        RenderTarget = NvFlowColorBufferGetRenderTarget(dstBuf.m_depthMaxBuffer);
        NvFlowContextSetRenderTarget(context, RenderTarget, 0);
        NvFlowContextSetViewport(context, &dstBuf.m_viewport);

        NvFlowDrawParams drawParams = {};
        drawParams.shader = m_multiResDepthDownsampleShader;
        drawParams.rootConstantBuffer = m_constantBuffer;
        drawParams.ps_readOnly[0] = NvFlowColorBufferGetResource(srcBuf.m_depthMaxBuffer);
        generateDefaultMesh(context);
        NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                     sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
        NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
    }

    for (uint32_t j = numLevels - 1; (j & 0x80000000) != 0; --j) {
        bool coarsestLevel = j == numLevels - 1;
        if (j == numLevels - 1) {
            auto &dstBuf = m_offscreenBuffers[j];
            NvFlowFloat4 color = make_float4(0.f, 0.f, 0.f, 1.f);
            RenderTarget = NvFlowColorBufferGetRenderTarget(dstBuf.m_colorBuffer);
            NvFlowContextClearRenderTarget(context, RenderTarget, color);
            RenderTarget = NvFlowColorBufferGetRenderTarget(dstBuf.m_depthMinBuffer);
            NvFlowContextClearRenderTarget(context, RenderTarget, color);
        }

        if (!coarsestLevel) {
            {
                auto &dstBuf = m_offscreenBuffers[j];
                auto &srcBuf = m_offscreenBuffers[j + 1];
                auto mapped = (UpsampleShaderParams *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                if (mapped) {
                    NvFlowFloat4 uvScale;
                    uvScale.x = 0.5f * dstBuf.screenPercentX;
                    uvScale.y = -0.5f * dstBuf.screenPercentY;
                    uvScale.x = 0.5f * dstBuf.screenPercentX;
                    uvScale.y = 0.5f * dstBuf.screenPercentY;
                    mapped->uvScale = uvScale;
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                RenderTarget = NvFlowColorBufferGetRenderTarget(dstBuf.m_colorBuffer);
                NvFlowContextSetRenderTarget(context, RenderTarget, 0);
                NvFlowContextSetViewport(context, &dstBuf.m_viewport);

                NvFlowDrawParams drawParams = {};
                drawParams.shader = m_multiResColorUpsampleShader;
                drawParams.rootConstantBuffer = m_constantBuffer;
                drawParams.ps_readOnly[0] =
                    NvFlowColorBufferGetResource(srcBuf.m_colorBuffer);
                drawParams.ps_readOnly[1] =
                    NvFlowColorBufferGetResource(srcBuf.m_depthMaxBuffer);
                generateDefaultMesh(context);
                NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                             sizeof(NvFlowFloat4), 0);
                NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
                NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
            }

            {
                auto &dstBuf = m_offscreenBuffers[j];
                auto &srcBuf = m_offscreenBuffers[j + 1];
                auto mapped = (UpsampleShaderParams *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                if (mapped) {
                    NvFlowFloat4 uvScale;
                    uvScale.x = 0.5f * dstBuf.screenPercentX;
                    uvScale.y = -0.5f * dstBuf.screenPercentY;
                    uvScale.x = 0.5f * dstBuf.screenPercentX;
                    uvScale.y = 0.5f * dstBuf.screenPercentY;
                    mapped->uvScale = uvScale;
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                RenderTarget = NvFlowColorBufferGetRenderTarget(dstBuf.m_depthMinBuffer);
                NvFlowContextSetRenderTarget(context, RenderTarget, 0);
                NvFlowContextSetViewport(context, &dstBuf.m_viewport);

                NvFlowDrawParams drawParams = {};
                drawParams.shader = m_multiResDepthUpsampleShader;
                drawParams.rootConstantBuffer = m_constantBuffer;
                drawParams.ps_readOnly[0] =
                    NvFlowColorBufferGetResource(srcBuf.m_depthMaxBuffer);
                generateDefaultMesh(context);
                NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                             sizeof(NvFlowFloat4), 0);
                NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
                NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
            }
        }

        auto &dstBuf = m_offscreenBuffers[j];
        bool finalLevel = j == 0;
        NvFlowFloat4 depthMaxInvTransform;
        if (finalLevel)
            depthMaxInvTransform = make_float4(1.f, 0.f, 0.f, 1.f);
        else
            depthMaxInvTransform = depthInvTransform;
        NvFlowFloat4 depthMinInvTransform = make_float4(1.f, 0.f, 0.f, 1.f);

        NvFlowDepthStencil *DepthStencil = 0;
        if (!coarsestLevel) {
            DepthStencil = NvFlowDepthBufferGetDepthStencil(dstBuf.m_rayMarchMask);
            NvFlowContextClearDepthStencil(context, DepthStencil, 1.f);
            auto mapped =
                (RayMarchShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
            if (mapped) {
                NvFlowFloat4 uvScale;
                uvScale.x = 0.5f * dstBuf.screenPercentX;
                uvScale.y = -0.5f * dstBuf.screenPercentY;
                uvScale.x = 0.5f * dstBuf.screenPercentX;
                uvScale.y = 0.5f * dstBuf.screenPercentY;
                mapped->uvScale = uvScale;
                mapped->depthInvTransform = depthInvTransform;
                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }
            NvFlowContextSetRenderTarget(context, nullptr, DepthStencil);
            NvFlowContextSetViewport(context, &dstBuf.m_viewport);

            NvFlowDrawParams drawParams = {};
            drawParams.shader = m_rayMarchMask;
            drawParams.rootConstantBuffer = m_constantBuffer;
            drawParams.ps_readOnly[0] =
                NvFlowColorBufferGetResource(dstBuf.m_depthMinBuffer);
            drawParams.ps_readOnly[1] = NvFlowDepthBufferGetResource(m_depthMask);
            generateDefaultMesh(context);
            NvFlowContextSetVertexBuffer(context, m_compositeVertexBufferRect,
                                         sizeof(NvFlowFloat4), 0);
            NvFlowContextSetIndexBuffer(context, m_compositeIndexBufferRect, 0);
            NvFlowContextDrawIndexedInstanced(context, 6, 1, &drawParams);
        }

        rayMarch(context, NvFlowColorBufferGetRenderTarget(dstBuf.m_colorBuffer),
                 DepthStencil, false, NvFlowColorBufferGetResource(dstBuf.m_depthMaxBuffer),
                 NvFlowColorBufferGetResource(dstBuf.m_depthMinBuffer), dstBuf.m_viewport,
                 dstBuf.screenPercentX, dstBuf.screenPercentY, blockListRangeList,
                 exportHandle, modelViewProjT, depthMaxInvTransform, depthMinInvTransform,
                 rayOriginVirtual, rayForwardDirVirtual, linearParams, gridDim, enableVTR,
                 params);
    }
}

void VolumeRender::resizeDepth(NvFlowContext *context, unsigned int rtv_width,
                               unsigned int rtv_height,
                               const NvFlowVolumeRenderParams *params) {
    uint32_t numLevels = 1;
    uint32_t width = rtv_width;
    uint32_t height = rtv_height;
    if (params->downsampleFactor == eNvFlowVolumeRenderDownsample2x2) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }

    if (params->multiResRayMarch > eNvFlowMultiResRayMarchDisabled &&
        params->multiResRayMarch <= eNvFlowMultiResRayMarch16x16) {
        numLevels = params->multiResRayMarch + 1;
    }

    for (uint32_t i = 1; i < numLevels; ++i) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }

    int widthScreenPercent = int(params->screenPercentage * width);
    int heightScreenPercent = int(params->screenPercentage * height);
    uint32_t k = 0;
    uint32_t w = width;
    uint32_t h = height;
    uint32_t depthWidth = (width + 3) / 4;
    uint32_t depthHeight = (height + 3) / 4;
    bool shouldAllocate = m_depthEstimate == nullptr;
    if (m_depthEstimate) {
        NvFlowColorBufferDesc depthEstimate_desc;
        NvFlowColorBufferGetDesc(m_depthEstimate, &depthEstimate_desc);
        if (depthEstimate_desc.width == depthWidth &&
            depthEstimate_desc.height == depthHeight)
            shouldAllocate = 0;
    }

    if (shouldAllocate) {
        SafeRelease(m_depthEstimate);
        NvFlowColorBufferDesc bufDesc;
        bufDesc.format = eNvFlowFormat_r16g16_float;
        bufDesc.width = depthWidth;
        bufDesc.height = depthHeight;
        m_depthEstimate = NvFlowCreateColorBuffer(context, &bufDesc);

        NvFlowColorBufferDesc desc;
        NvFlowColorBufferGetDesc(m_depthEstimate, &desc);
        auto RenderTarget = NvFlowColorBufferGetRenderTarget(m_depthEstimate);
        NvFlowRenderTargetDesc depthEstimate_rt_desc;
        NvFlowRenderTargetGetDesc(RenderTarget, &depthEstimate_rt_desc);
        m_depthEstimateViewport = depthEstimate_rt_desc.viewport;
        m_depthEstimateViewport.width = depthWidth * params->screenPercentage;
        m_depthEstimateViewport.height = depthHeight * params->screenPercentage;
        m_depthEstimateScreenPercentX = m_depthEstimateViewport.width / float(desc.width);
        m_depthEstimateScreenPercentY = m_depthEstimateViewport.height / float(desc.height);
    }
}

void VolumeRender::resizeTargets(NvFlowContext *context, unsigned int rtv_width,
                                 unsigned int rtv_height,
                                 const NvFlowVolumeRenderParams *params) {
    uint32_t numLevels = 1;
    uint32_t width = rtv_width;
    uint32_t height = rtv_height;
    if (params->downsampleFactor == eNvFlowVolumeRenderDownsample2x2) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }

    if (params->multiResRayMarch > eNvFlowMultiResRayMarchDisabled &&
        params->multiResRayMarch <= eNvFlowMultiResRayMarch16x16) {
        numLevels = params->multiResRayMarch + 1;
    }

    for (uint32_t i = 1; i < numLevels; ++i) {
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }

    int widthScreenPercent = int(params->screenPercentage * width);
    int heightScreenPercent = int(params->screenPercentage * height);

    while (m_offscreenBuffers.size() < numLevels)
        m_offscreenBuffers.allocateBack();
    while (m_offscreenBuffers.size() > numLevels) {
        m_offscreenBuffers.back().release();
        m_offscreenBuffers.pop_back();
    }

    for (uint32_t idx = 0; idx < numLevels; ++idx) {
        uint32_t k = 1u << (numLevels - idx - 1);
        uint32_t w = width * k;
        uint32_t h = height * k;
        auto &buf = m_offscreenBuffers[idx];
        auto &colorBuffer = buf.m_colorBuffer;
        auto &depthMaxBuffer = buf.m_depthMaxBuffer;
        auto &depthMinBuffer = buf.m_depthMinBuffer;
        if (buf.m_width != w || buf.m_height != h) {
            SafeRelease(colorBuffer);
            SafeRelease(depthMinBuffer);
            SafeRelease(depthMaxBuffer);
            buf.m_width = w;
            buf.m_height = h;

            NvFlowColorBufferDesc colorDesc = {};
            colorDesc.format = eNvFlowFormat_r16g16b16a16_float;
            colorDesc.width = w;
            colorDesc.height = h;
            colorBuffer = NvFlowCreateColorBuffer(context, &colorDesc);

            colorDesc.format = eNvFlowFormat_r32_float;
            depthMinBuffer = NvFlowCreateColorBuffer(context, &colorDesc);
            depthMaxBuffer = NvFlowCreateColorBuffer(context, &colorDesc);

            if (!idx) {
                SafeRelease(m_depthMask);
                NvFlowDepthBufferDesc bufDesc;
                bufDesc.format_resource = eNvFlowFormat_r32_typeless;
                bufDesc.format_dsv = eNvFlowFormat_d32_float;
                bufDesc.format_srv = eNvFlowFormat_r32_float;
                bufDesc.width = w;
                bufDesc.height = h;
                m_depthMask = NvFlowCreateDepthBuffer(context, &bufDesc);
            }
            auto &rayMarchMask = buf.m_rayMarchMask;
            SafeRelease(rayMarchMask);
            NvFlowDepthBufferDesc desc;
            desc.format_resource = eNvFlowFormat_r16_typeless;
            desc.format_dsv = eNvFlowFormat_d16_unorm;
            desc.format_srv = eNvFlowFormat_r16_unorm;
            desc.width = w;
            desc.height = h;
            rayMarchMask = NvFlowCreateDepthBuffer(context, &desc);
        }

        auto RenderTarget = NvFlowColorBufferGetRenderTarget(colorBuffer);
        NvFlowRenderTargetDesc colorBuffer_desc;
        NvFlowRenderTargetGetDesc(RenderTarget, &colorBuffer_desc);
        NvFlowColorBufferDesc buf_colorBuffer_desc;
        NvFlowColorBufferGetDesc(colorBuffer, &buf_colorBuffer_desc);
        buf.m_viewport = colorBuffer_desc.viewport;
        buf.m_viewport.width = widthScreenPercent * k;
        buf.m_viewport.height = heightScreenPercent * k;
        buf.screenPercentX = buf.m_viewport.width / float(buf_colorBuffer_desc.width);
        buf.screenPercentY = buf.m_viewport.height / float(buf_colorBuffer_desc.height);
    }
}

void VolumeRender::DebugUploadBuffer::reserve(NvFlowContext *context, NvFlowFormat format,
                                              uint32_t numElements) {
    if (numElements > m_bufferSize) {
        SafeRelease(m_buffer);

        for (m_bufferSize = 1024; m_bufferSize < numElements; m_bufferSize *= 2)
            ;

        NvFlowBufferDesc bufDesc = {};
        bufDesc.dim = m_bufferSize;
        bufDesc.format = format;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_buffer = NvFlowCreateBuffer(context, &bufDesc);
    }
}

void VolumeRender::DebugSimpleShapeMeshes::drawMesh(NvFlowContext *context,
                                                    NvFlowShapeType shapeType,
                                                    uint32_t numInstances,
                                                    const NvFlowDrawParams *drawParams) {
    if (!m_vertexBuffer && !m_indexBuffer) {
        NvFlowVertexBufferDesc vbufDesc = {};
        vbufDesc.data = pts;
        vbufDesc.sizeInBytes = sizeof(pts);
        m_vertexBuffer = NvFlowCreateVertexBuffer(context, &vbufDesc);

        NvFlowIndexBufferDesc ibufDesc = {};
        ibufDesc.format = eNvFlowFormat_r32_uint;
        ibufDesc.data = indices;
        ibufDesc.sizeInBytes = sizeof(indices);
        m_indexBuffer = NvFlowCreateIndexBuffer(context, &ibufDesc);
    }

    if (shapeType == eNvFlowShapeTypeSphere) {
        NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0);
        NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0);
        NvFlowContextDrawIndexedInstanced(context, 0xC0, numInstances, drawParams);
    } else if (shapeType == eNvFlowShapeTypeBox) {
        NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0xE40);
        NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0x720);
        NvFlowContextDrawIndexedInstanced(context, 0x18, numInstances, drawParams);
    } else if (shapeType == eNvFlowShapeTypeCapsule) {
        NvFlowContextSetVertexBuffer(context, m_vertexBuffer, sizeof(NvFlowFloat4), 0x600);
        NvFlowContextSetIndexBuffer(context, m_indexBuffer, 0x300);
        NvFlowContextDrawIndexedInstanced(context, 0x108, numInstances, drawParams);
    }
}

}  // namespace NvFlow