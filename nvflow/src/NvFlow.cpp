#include "NvFlowImpl.h"
#include "FlowMath.h"
#include "Grid.h"
#include "ShapeSDF.h"
#include "GridExport.h"
#include "GridImport.h"
#include "GridSummary.h"
#include "RenderMaterialPool.h"
#include "VolumeRender.h"
#include "VolumeShadow.h"
#include "CrossSection.h"
#include "GridProxy.h"
#include "Device.h"
#include "SDFGen.h"
#include "ParticleSurface.h"

using namespace NvFlow;

NV_FLOW_API void NvFlowGridDescDefaults(NvFlowGridDesc* desc) {
    if (desc) {
        desc->initialLocation = make_float3(0.f);
        desc->halfSize = make_float3(8.f);
        desc->virtualDim = make_dim(512);
        desc->densityMultiRes = eNvFlowMultiRes2x2x2;
        desc->residentScale = 0.0093750004f;
        desc->coarseResidentScaleFactor = 1.5f;
        desc->enableVTR = 0;
        desc->lowLatencyMapping = 0;
    }
}

NV_FLOW_API NvFlowGrid* NvFlowCreateGrid(NvFlowContext* context,
                                         const NvFlowGridDesc* desc) {
    return FlowCreateGrid(context, desc);
}

NV_FLOW_API void NvFlowReleaseGrid(NvFlowGrid* grid) {
    grid->release();
}

NV_FLOW_API void NvFlowGridResetDescDefaults(NvFlowGridResetDesc* desc) {
    if (desc) {
        desc->initialLocation = make_float3(0.f);
        desc->halfSize = make_float3(8.f);
    }
}

NV_FLOW_API void NvFlowGridReset(NvFlowGrid* grid, const NvFlowGridResetDesc* desc) {
    return grid->reset(desc);
}

NV_FLOW_API void NvFlowGridSetTargetLocation(NvFlowGrid* grid,
                                             NvFlowFloat3 targetLocation) {
    return grid->setTargetLocation(targetLocation);
}

NV_FLOW_API void NvFlowGridParamsDefaults(NvFlowGridParams* params) {
    if (params) {
        params->gravity = make_float3(0.f, -1.f, 0.f);
        params->singlePassAdvection = 1;
        params->pressureLegacyMode = 0;
        params->bigEffectMode = 0;
        params->bigEffectPredictTime = 0.1f;
        params->debugVisFlags = eNvFlowGridDebugVisBlocks;
    }
}

NV_FLOW_API void NvFlowGridSetParams(NvFlowGrid* grid, const NvFlowGridParams* params) {
    return grid->setParams(params);
}

NV_FLOW_API NvFlowResult NvFlowGridQuerySupport(NvFlowGrid* grid, NvFlowContext* context,
                                                NvFlowSupport* support) {
    return grid->querySupport(context, support);
}

NV_FLOW_API NvFlowResult NvFlowGridQueryTime(NvFlowGrid* grid, NvFlowQueryTime* gpuTime,
                                             NvFlowQueryTime* cpuTime) {
    return grid->queryTime(gpuTime, cpuTime);
}

NV_FLOW_API void NvFlowGridGPUMemUsage(NvFlowGrid* grid, NvFlowUint64* numBytes) {
    grid->GPUMemUsage(numBytes);
}

NV_FLOW_API void NvFlowGridUpdate(NvFlowGrid* grid, NvFlowContext* context, float dt) {
    grid->update(context, dt);
}

NV_FLOW_API NvFlowGridExport* NvFlowGridGetGridExport(NvFlowContext* context,
                                                      NvFlowGrid* grid) {
    return grid->getGridExport(context);
}

NV_FLOW_API void NvFlowGridMaterialParamsDefaults(NvFlowGridMaterialParams* params) {
    if (params) {
        params->velocity.damping = 0.1f;
        params->velocity.fade = 0.1f;
        params->velocity.macCormackBlendFactor = 0.5f;
        params->velocity.macCormackBlendThreshold = 0.001f;
        params->velocity.allocWeight = 0.0;
        params->velocity.allocThreshold = 0.0;
        params->smoke.damping = 0.1f;
        params->smoke.fade = 0.1f;
        params->smoke.macCormackBlendFactor = 0.5f;
        params->smoke.macCormackBlendThreshold = 0.001f;
        params->smoke.allocWeight = 0.0;
        params->smoke.allocThreshold = 0.0;
        params->temperature.damping = 0.0;
        params->temperature.fade = 0.0;
        params->temperature.macCormackBlendFactor = 0.5f;
        params->temperature.macCormackBlendThreshold = 0.001f;
        params->temperature.allocWeight = 1.0f;
        params->temperature.allocThreshold = 0.050000001f;
        params->fuel.damping = 0.0020000001f;
        params->fuel.fade = 0.0020000001f;
        params->fuel.macCormackBlendFactor = 0.5f;
        params->fuel.macCormackBlendThreshold = 0.001f;
        params->fuel.allocWeight = 0.0;
        params->fuel.allocThreshold = 0.0;
        params->vorticityStrength = 9.0f;
        params->vorticityVelocityMask = 1.0f;
        params->vorticityTemperatureMask = 0.0;
        params->vorticitySmokeMask = 0.0;
        params->vorticityFuelMask = 0.0;
        params->vorticityConstantMask = 0.0;
        params->ignitionTemp = 0.050000001f;
        params->burnPerTemp = 4.0f;
        params->fuelPerBurn = 1.0f;
        params->tempPerBurn = 5.0f;
        params->smokePerBurn = 3.0f;
        params->divergencePerBurn = 4.0f;
        params->buoyancyPerTemp = 4.0f;
        params->coolingRate = 1.5f;
    }
}

NV_FLOW_API NvFlowGridMaterialHandle NvFlowGridGetDefaultMaterial(NvFlowGrid* grid) {
    return grid->getDefaultMaterial();
}

NV_FLOW_API NvFlowGridMaterialHandle
NvFlowGridCreateMaterial(NvFlowGrid* grid, const NvFlowGridMaterialParams* params) {
    return grid->createMaterial(params);
}

NV_FLOW_API void NvFlowGridReleaseMaterial(NvFlowGrid* grid,
                                           NvFlowGridMaterialHandle material) {
    grid->releaseMaterial(material);
}

NV_FLOW_API void NvFlowGridSetMaterialParams(NvFlowGrid* grid,
                                             NvFlowGridMaterialHandle material,
                                             const NvFlowGridMaterialParams* params) {
    grid->setMaterialParams(material, params);
}

NV_FLOW_API void NvFlowShapeSDFDescDefaults(NvFlowShapeSDFDesc* desc) {
    if (desc) {
        desc->resolution = make_dim(16);
    }
}

NV_FLOW_API NvFlowShapeSDF* NvFlowCreateShapeSDF(NvFlowContext* context,
                                                 const NvFlowShapeSDFDesc* desc) {
    return FlowCreateShape(context, desc);
}

NV_FLOW_API NvFlowShapeSDF* NvFlowCreateShapeSDFFromTexture3D(NvFlowContext* context,
                                                              NvFlowTexture3D* texture) {
    return FlowCreateShape(context, texture);
}

NV_FLOW_API void NvFlowReleaseShapeSDF(NvFlowShapeSDF* shape) {
    shape->release();
}

NV_FLOW_API NvFlowShapeSDFData NvFlowShapeSDFMap(NvFlowShapeSDF* shape,
                                                 NvFlowContext* context) {
    return shape->map(context);
}

NV_FLOW_API void NvFlowShapeSDFUnmap(NvFlowShapeSDF* shape, NvFlowContext* context) {
    shape->unmap(context);
}

NV_FLOW_API void NvFlowGridEmitParamsDefaults(NvFlowGridEmitParams* params) {
    if (params) {
        params->shapeRangeOffset = 0;
        params->shapeRangeSize = 1;
        params->shapeType = eNvFlowShapeTypeSphere;
        params->shapeDistScale = 1.f;
        params->bounds = identity();
        params->localToWorld = identity();
        params->centerOfMass = make_float3(0.f);
        params->deltaTime = 0.f;
        params->emitMaterialIndex = -1;
        params->emitMode = eNvFlowGridEmitModeDefault;
        params->allocationScale = make_float3(1.f);
        params->allocationPredict = 0.125f;
        params->predictVelocity = make_float3(0.f);
        params->predictVelocityWeight = 0.f;
        params->minActiveDist = -1.f;
        params->maxActiveDist = 0.f;
        params->minEdgeDist = 0.f;
        params->maxEdgeDist = 0.1f;
        params->slipThickness = 0.f;
        params->slipFactor = 0.f;
        params->velocityLinear = make_float3(0.f);
        params->velocityAngular = make_float3(0.f);
        params->velocityCoupleRate = make_float3(0.5f);
        params->smoke = 0.5f;
        params->smokeCoupleRate = 0.5f;
        params->temperature = 2.f;
        params->temperatureCoupleRate = 0.5f;
        params->fuel = 1.f;
        params->fuelCoupleRate = 0.5f;
        params->fuelReleaseTemp = 0.1f;
        params->fuelRelease = 0.f;
    }
}

NV_FLOW_API void NvFlowGridEmit(NvFlowGrid* grid, const NvFlowShapeDesc* shapes,
                                NvFlowUint numShapes, const NvFlowGridEmitParams* params,
                                NvFlowUint numParams) {
    grid->emit(shapes, numShapes, params, numParams);
}

NV_FLOW_API void NvFlowGridUpdateEmitMaterials(NvFlowGrid* grid,
                                               NvFlowGridMaterialHandle* materials,
                                               NvFlowUint numMaterials) {
    grid->updateEmitMaterials(materials, numMaterials);
}

NV_FLOW_API void NvFlowGridUpdateEmitSDFs(NvFlowGrid* grid, NvFlowShapeSDF** sdfs,
                                          NvFlowUint numSdfs) {
    grid->updateEmitSDFs(sdfs, numSdfs);
}

NV_FLOW_API void NvFlowGridEmitCustomRegisterAllocFunc(NvFlowGrid* grid,
                                                       NvFlowGridEmitCustomAllocFunc func,
                                                       void* userdata) {
    grid->emitCustomRegisterAllocFunc(func, userdata);
}

NV_FLOW_API void NvFlowGridEmitCustomRegisterEmitFunc(NvFlowGrid* grid,
                                                      NvFlowGridTextureChannel channel,
                                                      NvFlowGridEmitCustomEmitFunc func,
                                                      void* userdata) {
    grid->emitCustomRegisterEmitFunc(channel, func, userdata);
}

NV_FLOW_API void NvFlowGridEmitCustomGetLayerParams(
    const NvFlowGridEmitCustomEmitParams* emitParams, NvFlowUint layerIdx,
    NvFlowGridEmitCustomEmitLayerParams* emitLayerParams) {
    emitParams->grid->emitCustomGetLayerParams(emitParams, layerIdx, emitLayerParams);
}

NV_FLOW_API NvFlowGridExportHandle
NvFlowGridExportGetHandle(NvFlowGridExport* gridExport, NvFlowContext* context,
                          NvFlowGridTextureChannel channel) {
    return gridExport->getHandle(context, channel);
}

NV_FLOW_API void NvFlowGridExportGetLayerView(NvFlowGridExportHandle handle,
                                              NvFlowUint layerIdx,
                                              NvFlowGridExportLayerView* layerView) {
    handle.gridExport->getLayerView(handle, layerIdx, layerView);
}

NV_FLOW_API void NvFlowGridExportGetLayeredView(NvFlowGridExportHandle handle,
                                                NvFlowGridExportLayeredView* layeredView) {
    handle.gridExport->getLayeredView(handle, layeredView);
}

NV_FLOW_API void NvFlowGridExportGetDebugVisView(NvFlowGridExport* gridExport,
                                                 NvFlowGridExportDebugVisView* view) {
    gridExport->getDebugVisView(view);
}

NV_FLOW_API NvFlowGridImport* NvFlowCreateGridImport(NvFlowContext* context,
                                                     const NvFlowGridImportDesc* desc) {
    return FlowCreateGridImport(context, desc);
}

NV_FLOW_API void NvFlowReleaseGridImport(NvFlowGridImport* gridImport) {
    gridImport->release();
}

NV_FLOW_API NvFlowGridImportHandle
NvFlowGridImportGetHandle(NvFlowGridImport* gridImport, NvFlowContext* context,
                          const NvFlowGridImportParams* params) {
    return gridImport->getHandle(context, params);
}

NV_FLOW_API void NvFlowGridImportGetLayerView(NvFlowGridImportHandle handle,
                                              NvFlowUint layerIdx,
                                              NvFlowGridImportLayerView* layerView) {
    handle.gridImport->getLayerView(handle, layerIdx, layerView);
}

NV_FLOW_API void NvFlowGridImportGetLayeredView(NvFlowGridImportHandle handle,
                                                NvFlowGridImportLayeredView* layeredView) {
    handle.gridImport->getLayeredView(handle, layeredView);
}

NV_FLOW_API void NvFlowGridImportReleaseChannel(NvFlowGridImport* gridImport,
                                                NvFlowContext* context,
                                                NvFlowGridTextureChannel channel) {
    gridImport->releaseChannel(context, channel);
}

NV_FLOW_API NvFlowGridExport* NvFlowGridImportGetGridExport(NvFlowGridImport* gridImport,
                                                            NvFlowContext* context) {
    return gridImport->getGridExport(context);
}

NV_FLOW_API NvFlowGridImportStateCPU* NvFlowCreateGridImportStateCPU(
    NvFlowGridImport* gridImport) {
    return gridImport->createImportStateCPU();
}

NV_FLOW_API void NvFlowReleaseGridImportStateCPU(NvFlowGridImportStateCPU* stateCPU) {
    stateCPU->release();
}

NV_FLOW_API void NvFlowGridImportUpdateStateCPU(NvFlowGridImportStateCPU* stateCPU,
                                                NvFlowContext* context,
                                                NvFlowGridExport* gridExport) {
    stateCPU->updateStateCPU(context, gridExport);
}

NV_FLOW_API NvFlowGridImportHandle
NvFlowGridImportStateCPUGetHandle(NvFlowGridImport* gridImport, NvFlowContext* context,
                                  const NvFlowGridImportStateCPUParams* params) {
    return gridImport->stateCPUGetHandle(context, params);
}

NV_FLOW_API NvFlowGridSummary* NvFlowCreateGridSummary(NvFlowContext* context,
                                                       const NvFlowGridSummaryDesc* desc) {
    return FlowCreateGridSummary(context, desc);
}

NV_FLOW_API void NvFlowReleaseGridSummary(NvFlowGridSummary* gridSummary) {
    gridSummary->release();
}

NV_FLOW_API NvFlowGridSummaryStateCPU* NvFlowCreateGridSummaryStateCPU(
    NvFlowGridSummary* gridSummary) {
    return gridSummary->createStateCPU();
}

NV_FLOW_API void NvFlowReleaseGridSummaryStateCPU(NvFlowGridSummaryStateCPU* stateCPU) {
    stateCPU->release();
}

NV_FLOW_API void NvFlowGridSummaryUpdate(NvFlowGridSummary* gridSummary,
                                         NvFlowContext* context,
                                         const NvFlowGridSummaryUpdateParams* params) {
    gridSummary->update(context, params);
}

NV_FLOW_API void NvFlowGridSummaryDebugRender(
    NvFlowGridSummary* gridSummary, NvFlowContext* context,
    const NvFlowGridSummaryDebugRenderParams* params) {
    gridSummary->debugRender(context, params);
}

NV_FLOW_API NvFlowUint NvFlowGridSummaryGetNumLayers(NvFlowGridSummaryStateCPU* stateCPU) {
    return stateCPU->getNumLayers();
}

NV_FLOW_API NvFlowGridMaterialHandle NvFlowGridSummaryGetLayerMaterial(
    NvFlowGridSummaryStateCPU* stateCPU, NvFlowUint layerIdx) {
    return stateCPU->getLayerMaterial(layerIdx);
}

NV_FLOW_API void NvFlowGridSummaryGetSummaries(NvFlowGridSummaryStateCPU* stateCPU,
                                               NvFlowGridSummaryResult** results,
                                               NvFlowUint* numResults,
                                               NvFlowUint layerIdx) {
    stateCPU->getSummaries((const NvFlowGridSummaryResult**)results, numResults, layerIdx);
}

NV_FLOW_API NvFlowRenderMaterialPool* NvFlowCreateRenderMaterialPool(
    NvFlowContext* context, const NvFlowRenderMaterialPoolDesc* desc) {
    return FlowCreateRenderMaterialPool(context, desc);
}

NV_FLOW_API void NvFlowReleaseRenderMaterialPool(NvFlowRenderMaterialPool* pool) {
    pool->release();
}

NV_FLOW_API void NvFlowRenderMaterialParamsDefaults(NvFlowRenderMaterialParams* params) {
    if (params) {
        ZeroMemory(&params->material, sizeof(params->material));
        params->alphaScale = 0.1f;
        params->additiveFactor = 0.f;
        params->colorMapCompMask = make_float4(1.f, 0.f, 0.f, 0.f);
        params->alphaCompMask = make_float4(0.f, 0.f, 0.f, 1.f);
        params->intensityCompMask = make_float4(0.f);
        params->colorMapMinX = 0.f;
        params->colorMapMaxX = 1.f;
        params->alphaBias = 0.f;
        params->intensityBias = 1.f;
    }
}

NV_FLOW_API NvFlowRenderMaterialHandle
NvFlowGetDefaultRenderMaterial(NvFlowRenderMaterialPool* pool) {
    return pool->getDefaultRenderMaterial();
}

NV_FLOW_API NvFlowRenderMaterialHandle
NvFlowCreateRenderMaterial(NvFlowContext* context, NvFlowRenderMaterialPool* pool,
                           const NvFlowRenderMaterialParams* params) {
    return pool->createRenderMaterial(context, params);
}

NV_FLOW_API void NvFlowReleaseRenderMaterial(NvFlowRenderMaterialHandle handle) {
    handle.pool->releaseRenderMaterial(handle);
}

NV_FLOW_API void NvFlowRenderMaterialUpdate(NvFlowRenderMaterialHandle handle,
                                            const NvFlowRenderMaterialParams* params) {
    handle.pool->renderMaterialUpdate(handle, params);
}

NV_FLOW_API NvFlowColorMapData
NvFlowRenderMaterialColorMap(NvFlowContext* context, NvFlowRenderMaterialHandle handle) {
    return handle.pool->renderMaterialColorMap(context, handle);
}

NV_FLOW_API void NvFlowRenderMaterialColorUnmap(NvFlowContext* context,
                                                NvFlowRenderMaterialHandle handle) {
    handle.pool->renderMaterialColorUnmap(context, handle);
}

NV_FLOW_API NvFlowVolumeRender* NvFlowCreateVolumeRender(
    NvFlowContext* context, const NvFlowVolumeRenderDesc* desc) {
    return FlowCreateVolumeRender(context, desc);
}

NV_FLOW_API void NvFlowReleaseVolumeRender(NvFlowVolumeRender* volumeRender) {
    volumeRender->release();
}

NV_FLOW_API void NvFlowVolumeRenderParamsDefaults(NvFlowVolumeRenderParams* params) {
    if (params) {
        params->projectionMatrix = identity();
        params->viewMatrix = identity();
        params->depthStencilView = 0;
        params->renderTargetView = 0;
        params->materialPool = 0;
        params->renderMode = eNvFlowVolumeRenderMode_colormap;
        params->renderChannel = eNvFlowGridTextureChannelDensity;
        params->debugMode = 0;
        params->downsampleFactor = eNvFlowVolumeRenderDownsample2x2;
        params->screenPercentage = 1.f;
        params->multiResRayMarch = eNvFlowMultiResRayMarchDisabled;
        params->multiResSamplingScale = 2.f;
        params->smoothColorUpsample = 0;
        params->preColorCompositeOnly = 0;
        params->colorCompositeOnly = 0;
        params->generateDepth = 0;
        params->generateDepthDebugMode = 0;
        params->depthAlphaThreshold = 0.89999998;
        params->depthIntensityThreshold = 4.0;
        params->multiRes.enabled = 0;
        params->multiRes.centerWidth = 0.40000001;
        params->multiRes.centerHeight = 0.40000001;
        params->multiRes.centerX = 0.5;
        params->multiRes.centerY = 0.5;
        params->multiRes.densityScaleX[0] = 0.5;
        params->multiRes.densityScaleX[1] = 1.f;
        params->multiRes.densityScaleX[2] = 0.5;
        params->multiRes.densityScaleY[0] = 0.5;
        params->multiRes.densityScaleY[1] = 1.f;
        params->multiRes.densityScaleY[2] = 0.5;
        params->multiRes.viewport.topLeftX = 0.0;
        params->multiRes.viewport.topLeftY = 0.0;
        params->multiRes.viewport.width = 0.0;
        params->multiRes.viewport.height = 0.0;
        params->multiRes.nonMultiResWidth = 0.0;
        params->multiRes.nonMultiResHeight = 0.0;
        params->lensMatchedShading.enabled = 0;
        params->lensMatchedShading.warpLeft = 0.47099999;
        params->lensMatchedShading.warpRight = 0.47099999;
        params->lensMatchedShading.warpUp = 0.47099999;
        params->lensMatchedShading.warpDown = 0.47099999;
        params->lensMatchedShading.sizeLeft = 552.09998;
        params->lensMatchedShading.sizeRight = 735.90002;
        params->lensMatchedShading.sizeUp = 847.0;
        params->lensMatchedShading.sizeDown = 584.40002;
        params->lensMatchedShading.viewport.topLeftX = 0.0;
        params->lensMatchedShading.viewport.topLeftY = 0.0;
        params->lensMatchedShading.viewport.width = 0.0;
        params->lensMatchedShading.viewport.height = 0.0;
        params->lensMatchedShading.nonLMSWidth = 0.0;
        params->lensMatchedShading.nonLMSHeight = 0.0;
    }
}

NV_FLOW_API NvFlowGridExport* NvFlowVolumeRenderLightGridExport(
    NvFlowVolumeRender* volumeRender, NvFlowContext* context, NvFlowGridExport* gridExport,
    const NvFlowVolumeLightingParams* params) {
    return volumeRender->lightGridExport(context, gridExport, params);
}

NV_FLOW_API void NvFlowVolumeRenderGridExport(NvFlowVolumeRender* volumeRender,
                                              NvFlowContext* context,
                                              NvFlowGridExport* gridExport,
                                              const NvFlowVolumeRenderParams* params) {
    volumeRender->renderGridExport(context, gridExport, params);
}

NV_FLOW_API void NvFlowVolumeRenderTexture3D(NvFlowVolumeRender* volumeRender,
                                             NvFlowContext* context,
                                             NvFlowTexture3D* density,
                                             const NvFlowVolumeRenderParams* params) {
    volumeRender->renderTexture3D(context, density, params);
}

NV_FLOW_API NvFlowVolumeShadow* NvFlowCreateVolumeShadow(
    NvFlowContext* context, const NvFlowVolumeShadowDesc* desc) {
    return FlowCreateVolumeShadow(context, desc);
}

NV_FLOW_API void NvFlowReleaseVolumeShadow(NvFlowVolumeShadow* volumeShadow) {
    volumeShadow->release();
}

NV_FLOW_API void NvFlowVolumeShadowUpdate(NvFlowVolumeShadow* volumeShadow,
                                          NvFlowContext* context,
                                          NvFlowGridExport* gridExport,
                                          const NvFlowVolumeShadowParams* params) {
    volumeShadow->update(context, gridExport, params);
}

NV_FLOW_API NvFlowGridExport* NvFlowVolumeShadowGetGridExport(
    NvFlowVolumeShadow* volumeShadow, NvFlowContext* context) {
    return volumeShadow->getGridExport(context);
}

NV_FLOW_API void NvFlowVolumeShadowDebugRender(
    NvFlowVolumeShadow* volumeShadow, NvFlowContext* context,
    const NvFlowVolumeShadowDebugRenderParams* params) {
    volumeShadow->debugRender(context, params);
}

NV_FLOW_API void NvFlowVolumeShadowGetStats(NvFlowVolumeShadow* volumeShadow,
                                            NvFlowVolumeShadowStats* stats) {
    volumeShadow->getStats(stats);
}

NV_FLOW_API void NvFlowCrossSectionParamsDefaults(NvFlowCrossSectionParams* params) {
    if (params) {
        params->gridExport = 0;
        params->gridExportDebugVis = 0;
        params->projectionMatrix = identity();
        params->viewMatrix = identity();
        params->depthStencilView = 0;
        params->renderTargetView = 0;
        params->materialPool = 0;
        params->renderMode = eNvFlowVolumeRenderMode_colormap;
        params->renderChannel = eNvFlowGridTextureChannelDensity;
        params->crossSectionAxis = 0;
        params->crossSectionPosition = make_float3(0.f);
        params->crossSectionScale = 2.f;
        params->intensityScale = 1.f;
        params->pointFilter = 0;
        params->velocityVectors = 1;
        params->velocityScale = 1.f;
        params->vectorLengthScale = 1.f;
        params->outlineCells = 0;
        params->fullscreen = 0;
        params->lineColor = make_float4(1.f);
        params->backgroundColor = make_float4(0.f, 0.f, 0.f, 1.f);
        params->cellColor = make_float4(1.f);
    }
}

NV_FLOW_API NvFlowCrossSection* NvFlowCreateCrossSection(
    NvFlowContext* context, const NvFlowCrossSectionDesc* desc) {
    return FlowCreateCrossSection(context, desc);
}

NV_FLOW_API void NvFlowReleaseCrossSection(NvFlowCrossSection* crossSection) {
    crossSection->release();
}

NV_FLOW_API void NvFlowCrossSectionRender(NvFlowCrossSection* crossSection,
                                          NvFlowContext* context,
                                          const NvFlowCrossSectionParams* params) {
    crossSection->render(context, params);
}

NV_FLOW_API NvFlowGridProxy* NvFlowCreateGridProxy(const NvFlowGridProxyDesc* desc) {
    return FlowCreateGridProxy(desc);
}

NV_FLOW_API void NvFlowReleaseGridProxy(NvFlowGridProxy* proxy) {
    proxy->release();
}

NV_FLOW_API void NvFlowGridProxyPush(NvFlowGridProxy* proxy, NvFlowGridExport* gridExport,
                                     const NvFlowGridProxyFlushParams* params) {
    proxy->push(gridExport, params);
}

NV_FLOW_API void NvFlowGridProxyFlush(NvFlowGridProxy* proxy,
                                      const NvFlowGridProxyFlushParams* params) {
    proxy->flush(params);
}

NV_FLOW_API NvFlowGridExport* NvFlowGridProxyGetGridExport(NvFlowGridProxy* proxy,
                                                           NvFlowContext* renderContext) {
    return proxy->getGridExport(renderContext);
}

NV_FLOW_API void NvFlowDeviceDescDefaults(NvFlowDeviceDesc* desc) {
    if (desc) {
        desc->mode = eNvFlowDeviceModeUnique;
        desc->autoSelectDevice = 1;
        desc->adapterIdx = 1;
    }
}

NV_FLOW_API bool NvFlowDedicatedDeviceAvailable(NvFlowContext* renderContext) {
    return FlowDedicatedDeviceAvailable(renderContext);
}

NV_FLOW_API bool NvFlowDedicatedDeviceQueueAvailable(NvFlowContext* renderContext) {
    return FlowDedicatedDeviceQueueAvailable(renderContext);
}

NV_FLOW_API NvFlowDevice* NvFlowCreateDevice(NvFlowContext* renderContext,
                                             const NvFlowDeviceDesc* desc) {
    return nullptr;
}

NV_FLOW_API void NvFlowReleaseDevice(NvFlowDevice* device) {}

NV_FLOW_API NvFlowDeviceQueue* NvFlowCreateDeviceQueue(NvFlowDevice* device,
                                                       const NvFlowDeviceQueueDesc* desc) {
    return nullptr;
}

NV_FLOW_API void NvFlowReleaseDeviceQueue(NvFlowDeviceQueue* deviceQueue) {
    return;
}

NV_FLOW_API NvFlowContext* NvFlowDeviceQueueCreateContext(NvFlowDeviceQueue* deviceQueue) {
    return nullptr;
}

NV_FLOW_API void NvFlowDeviceQueueUpdateContext(NvFlowDeviceQueue* deviceQueue,
                                                NvFlowContext* context,
                                                NvFlowDeviceQueueStatus* status) {
    return;
}

NV_FLOW_API void NvFlowDeviceQueueFlush(NvFlowDeviceQueue* deviceQueue,
                                        NvFlowContext* context) {
    return;
}

NV_FLOW_API void NvFlowDeviceQueueConditionalFlush(NvFlowDeviceQueue* deviceQueue,
                                                   NvFlowContext* context) {
    return;
}

NV_FLOW_API void NvFlowDeviceQueueWaitOnFence(NvFlowDeviceQueue* deviceQueue,
                                              NvFlowContext* context,
                                              NvFlowUint64 fenceValue) {
    return;
}

NV_FLOW_API NvFlowSDFGen* NvFlowCreateSDFGen(NvFlowContext* context,
                                             const NvFlowSDFGenDesc* desc) {
    return FlowCreateSDFGen(context, desc);
}

NV_FLOW_API void NvFlowReleaseSDFGen(NvFlowSDFGen* sdfGen) {
    sdfGen->release();
}

NV_FLOW_API void NvFlowSDFGenReset(NvFlowSDFGen* sdfGen, NvFlowContext* context) {
    sdfGen->reset(context);
}

NV_FLOW_API void NvFlowSDFGenVoxelize(NvFlowSDFGen* sdfGen, NvFlowContext* context,
                                      const NvFlowSDFGenMeshParams* params) {
    sdfGen->voxelize(context, params);
}

NV_FLOW_API void NvFlowSDFGenUpdate(NvFlowSDFGen* sdfGen, NvFlowContext* context) {
    sdfGen->update(context);
}

NV_FLOW_API NvFlowTexture3D* NvFlowSDFGenShape(NvFlowSDFGen* sdfGen,
                                               NvFlowContext* context) {
    return sdfGen->shape(context);
}

NV_FLOW_API NvFlowParticleSurface* NvFlowCreateParticleSurface(
    NvFlowContext* context, const NvFlowParticleSurfaceDesc* desc) {
    return FlowCreateParticleSurface(context, desc);
}

NV_FLOW_API void NvFlowReleaseParticleSurface(NvFlowParticleSurface* surface) {
    surface->release();
}

NV_FLOW_API void NvFlowParticleSurfaceUpdateParticles(
    NvFlowParticleSurface* surface, NvFlowContext* context,
    const NvFlowParticleSurfaceData* data) {
    surface->updateParticles(context, data);
}

NV_FLOW_API void NvFlowParticleSurfaceUpdateSurface(
    NvFlowParticleSurface* surface, NvFlowContext* context,
    const NvFlowParticleSurfaceParams* params) {
    surface->updateSurface(context, params);
}

NV_FLOW_API void NvFlowParticleSurfaceAllocFunc(
    NvFlowParticleSurface* surface, NvFlowContext* context,
    const NvFlowGridEmitCustomAllocParams* params) {
    surface->allocFunc(context, params);
}

NV_FLOW_API void NvFlowParticleSurfaceEmitVelocityFunc(
    NvFlowParticleSurface* surface, NvFlowContext* context, NvFlowUint* dataFrontIdx,
    const NvFlowGridEmitCustomEmitParams* params,
    const NvFlowParticleSurfaceEmitParams* emitParams) {
    surface->emitVelocityFunc(context, dataFrontIdx, params, emitParams);
}

NV_FLOW_API void NvFlowParticleSurfaceEmitDensityFunc(
    NvFlowParticleSurface* surface, NvFlowContext* context, NvFlowUint* dataFrontIdx,
    const NvFlowGridEmitCustomEmitParams* params,
    const NvFlowParticleSurfaceEmitParams* emitParams) {
    surface->emitDensityFunc(context, dataFrontIdx, params, emitParams);
}

NV_FLOW_API NvFlowGridExport* NvFlowParticleSurfaceDebugGridExport(
    NvFlowParticleSurface* surface, NvFlowContext* context) {
    return surface->debugGridExport(context);
}
