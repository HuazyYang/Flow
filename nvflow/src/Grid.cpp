#include "Grid.h"
#include "Object.h"
#include "BlockManager.h"
#include "SparseTextureMemoryPool.h"
#include "SparseTexturePool.h"
#include "GridExport.h"
#include "ShapeSDF.h"
#include "ClientHelper.h"

namespace NvFlow {

struct EmitCustomParams {
    SparseWritePointHandle frontWriteLayeredHandle;
    SparseWritePointHandle tempWriteLayeredHandle;
    SparseWritePointLayeredView layeredView;
    SparseTextureFront *field;
};

Grid *FlowCreateGrid(NvFlowContext *context, const NvFlowGridDesc *desc) {
    return new Grid(context, desc);
}

uint32_t Grid::addRef() {
    return Object::addRef();
}

uint32_t Grid::release() {
    return Object::release();
}

uint64_t Grid::getGPUBytesUsed() {
    return 0;
}

void Grid::reset(const NvFlowGridResetDesc *desc) {
    m_resetDesc = *desc;
    m_resetRequested = 1;
}

NvFlowResult Grid::querySupport(NvFlowContext *context, NvFlowSupport *support) {
    m_is_VTR_supported = NvFlowContextIsSparseTextureSupported(context);
    if (support)
        support->supportsVTR = m_is_VTR_supported;
    return eNvFlowSuccess;
}

NvFlowResult Grid::queryTime(NvFlowQueryTime *gpuTime, NvFlowQueryTime *cpuTime) {
    if (gpuTime)
        gpuTime->simulation = m_simTimeGPU;
    if (cpuTime)
        cpuTime->simulation = m_simTimeCPU;
    return eNvFlowSuccess;
}

Grid::Grid(NvFlowContext *context, const NvFlowGridDesc *desc)
    : m_is_VTR_supported(0),
      m_resetRequested(0),
      m_resetMode(0),
      m_resetDesc{},
      m_currentLocation{0.f, 0.f, 0.f},
      m_currentHalfSize{0.f, 0.f, 0.f},
      m_currentModelMatrix{identity()},
      m_oldLocation{0.f, 0.f, 0.f},
      m_targetLocation{0.f, 0.f, 0.f},
      m_blockManager(0),
      m_summaryCount(0),
      m_velocity{},
      m_densityCoarse{},
      m_velocityPool{0},
      m_velocityMemoryPool{0},
      m_pressure{},
      m_pressurePool{0},
      m_pressureMemoryPool{0},
      m_density{},
      m_densityPool{0},
      m_densityMemoryPool{0},
      m_advect{0},
      m_vorticityConfinement{0},
      m_emitter{0},
      m_pressureOp{0},
      m_gridExport{this},
      m_debugVis{},
      m_shapeRefs{0},
      m_emitCustomAlloc{},
      m_emitCustomEmit{},
      m_timer{0},
      m_simTimeCPU{0.f},
      m_simTimeGPU{0.f} {
    m_desc = *desc;
    NvFlowGridParamsDefaults(&m_params);

    NvFlowGridMaterialParams materialParams = {};
    NvFlowGridMaterialParamsDefaults(&materialParams);
    auto result = createMaterial(&materialParams);

    m_currentLocation = m_desc.initialLocation;
    m_currentHalfSize = m_desc.halfSize;

    updateModelMatrix();

    m_targetLocation = m_currentLocation;

    if (m_desc.enableVTR)
        m_is_VTR_supported = NvFlowContextIsSparseTextureSupported(context);

    bool enableVTR = m_desc.enableVTR & m_is_VTR_supported;
    m_desc.enableVTR = enableVTR;

    float densityResidentScale = m_desc.residentScale;
    float velocityResidentScale = m_desc.residentScale;
    NvFlowDim velocityVirtualDim = m_desc.virtualDim;
    NvFlowDim densityVirtualDim = m_desc.virtualDim;

    if (m_desc.densityMultiRes == eNvFlowMultiRes2x2x2) {
        densityVirtualDim = 2 * velocityVirtualDim;
        velocityResidentScale = velocityResidentScale * m_desc.coarseResidentScaleFactor;
    }

    BlockManagerDesc managerDesc;
    managerDesc.enableVTR = enableVTR;
    managerDesc.lowLatencyMapping = m_desc.lowLatencyMapping;
    managerDesc.velocityVirtualDim = velocityVirtualDim;
    managerDesc.densityVirtualDim = densityVirtualDim;
    m_blockManager = createBlockManager(context, &managerDesc);

    SparseTextureMemoryPoolDesc velocityMemPoolDesc;
    velocityMemPoolDesc.format = eNvFlowFormat_r16g16b16a16_float;
    velocityMemPoolDesc.poolGridDim = computeSparseTexturePoolGridDim(
        velocityMemPoolDesc.format, velocityVirtualDim, velocityResidentScale);
    velocityMemPoolDesc.enableVTR = enableVTR;
    m_velocityMemoryPool = createSparseTextureMemoryPool(context, &velocityMemPoolDesc);

    SparseTexturePoolDesc velocityPoolDesc = {};
    velocityPoolDesc.memoryPool = m_velocityMemoryPool;
    velocityPoolDesc.virtualDim = velocityVirtualDim;
    velocityPoolDesc.initialMapping = m_blockManager->getSparseMapping();
    m_velocityPool = createSparseTexturePool(context, &velocityPoolDesc);

    m_velocity.init(context, m_velocityPool);
    m_densityCoarse.init(context, m_velocityPool);

    SparseTextureMemoryPoolDesc texMemPoolDesc = {};
    texMemPoolDesc.format = eNvFlowFormat_r16g16_float;
    texMemPoolDesc.poolGridDim = computeSparseTexturePoolGridDim(
        texMemPoolDesc.format, velocityVirtualDim, velocityResidentScale);
    texMemPoolDesc.enableVTR = enableVTR;
    m_pressureMemoryPool = createSparseTextureMemoryPool(context, &texMemPoolDesc);

    SparseTexturePoolDesc texPoolDesc = {};
    texPoolDesc.memoryPool = m_pressureMemoryPool;
    texPoolDesc.initialMapping = m_blockManager->getSparseMapping();
    texPoolDesc.virtualDim = velocityVirtualDim;
    m_pressurePool = createSparseTexturePool(context, &texPoolDesc);

    m_pressure.init(context, m_pressurePool);
    SparseTextureHandle pback = m_pressure.acquireTexture(context);
    pback.releaseTexture();

    SparseTextureMemoryPoolDesc densityMemPoolDesc = {};
    densityMemPoolDesc.format = eNvFlowFormat_r16g16b16a16_float;
    densityMemPoolDesc.poolGridDim = computeSparseTexturePoolGridDim(
        densityMemPoolDesc.format, densityVirtualDim, densityResidentScale);
    densityMemPoolDesc.enableVTR = enableVTR;
    m_densityMemoryPool = createSparseTextureMemoryPool(context, &densityMemPoolDesc);

    SparseTexturePoolDesc densityPoolDesc = {};
    densityPoolDesc.memoryPool = m_densityMemoryPool;
    densityPoolDesc.virtualDim = densityVirtualDim;
    densityPoolDesc.initialMapping = m_blockManager->getSparseMapping();
    m_densityPool = createSparseTexturePool(context, &densityPoolDesc);

    m_density.init(context, m_densityPool);

    AdvectDesc advectDesc = {};
    advectDesc.enableVTR = enableVTR;
    advectDesc.velocity = &m_velocity;
    m_advect = createAdvect(context, &advectDesc);

    VorticityConfinementDesc vcDesc = {};
    vcDesc.enableVTR = enableVTR;
    vcDesc.velocityField = &m_density;
    m_vorticityConfinement = createVorticityConfinement(context, &vcDesc);

    EmitterDesc emitterDesc = {};
    emitterDesc.enableVTR = enableVTR;
    m_emitter = createEmitter(context, &emitterDesc);

    PressureDesc pressureDesc = {};
    pressureDesc.enableVTR = enableVTR;
    m_pressureOp = createPressure(context, &pressureDesc);

    uint32_t seedCount = 2;
    SparseTextureHandle densityTextures[2], velocityTextures[2];
    ZeroMemory(densityTextures, sizeof(densityTextures));
    ZeroMemory(velocityTextures, sizeof(velocityTextures));
    for (int i = 0; i < 2; ++i) {
        densityTextures[i] = m_density.acquireTexture(context);
        velocityTextures[i] = m_velocity.acquireTexture(context);
    }
    for (int k = 0; k < 2; ++k) {
        densityTextures[k].releaseTexture();
        velocityTextures[k].releaseTexture();
    }

    getGridExport(context);
    m_timer = NvFlowCreateContextTimer(context);
}

Grid::~Grid() {
    NvFlowReleaseContextTimer(m_timer);
    SafeRelease(m_blockManager);
    SafeRelease(m_velocityPool);
    SafeRelease(m_pressurePool);
    SafeRelease(m_densityPool);
    SafeRelease(m_velocityMemoryPool);
    SafeRelease(m_pressureMemoryPool);
    SafeRelease(m_densityMemoryPool);
    SafeRelease(m_advect);
    SafeRelease(m_vorticityConfinement);
    SafeRelease(m_emitter);
    SafeRelease(m_pressureOp);
    SafeRelease(m_shapeSDFs);
}

void Grid::advectCombustGetPerLayerImpl(AdvectPerLayerParams *params, uint32_t layerIdx) {
    auto PerMaterialFromLayerIdx = getPerMaterialFromLayerIdx(layerIdx);
    auto &materialParams = PerMaterialFromLayerIdx->materialParams;

    params->ignitionTemp = materialParams.ignitionTemp;
    params->burnPerTemp = materialParams.burnPerTemp;
    params->fuelPerBurn = materialParams.fuelPerBurn;
    params->tempPerBurn = materialParams.tempPerBurn;
    params->smokePerBurn = materialParams.smokePerBurn;
    params->divergencePerBurn = materialParams.divergencePerBurn;
    params->buoyancyPerTemp = materialParams.buoyancyPerTemp;
    params->coolingRate = materialParams.coolingRate;
    params->materialIdx = getMaterialIdxFromLayerIdx(layerIdx);

    if (m_resetMode) {
        params->damping = make_float4(1.f);
    }
}

void Grid::advectDensityGetPerLayer(AdvectPerLayerParams *params, Grid *userdata,
                                    uint32_t layerIdx) {
    return userdata->advectDensityGetPerLayerImpl(params, layerIdx);
}

void Grid::advectDensityGetPerLayerImpl(AdvectPerLayerParams *params, uint32_t layerIdx) {
    auto PerMaterialFromLayerIdx = getPerMaterialFromLayerIdx(layerIdx);
    auto &materialParams = PerMaterialFromLayerIdx->materialParams;
    params->blendFactor.x = materialParams.temperature.macCormackBlendFactor;
    params->blendFactor.y = materialParams.fuel.macCormackBlendFactor;
    params->blendFactor.z = materialParams.smoke.macCormackBlendFactor;
    params->blendFactor.w = materialParams.smoke.macCormackBlendFactor;

    params->blendThreshold.x = materialParams.temperature.macCormackBlendThreshold;
    params->blendThreshold.y = materialParams.fuel.macCormackBlendThreshold;
    params->blendThreshold.z = infinity();
    params->blendThreshold.w = materialParams.smoke.macCormackBlendThreshold;

    params->damping.x = materialParams.temperature.damping;
    params->damping.y = materialParams.fuel.damping;
    params->damping.z = 0.f;
    params->damping.w = materialParams.smoke.damping;

    params->fade.x = materialParams.temperature.fade;
    params->fade.y = materialParams.fuel.fade;
    params->fade.z = 0.f;
    params->fade.w = materialParams.smoke.fade;

    advectCombustGetPerLayerImpl(params, layerIdx);
}

void Grid::advectVelocityGetPerLayer(AdvectPerLayerParams *params, Grid *userdata,
                                     uint32_t layerIdx) {
    userdata->advectVelocityPerLayerImpl(params, layerIdx);
}

void Grid::advectVelocityPerLayerImpl(AdvectPerLayerParams *params, uint32_t layerIdx) {
    auto PerMaterialFromLayerIdx = getPerMaterialFromLayerIdx(layerIdx);
    auto &materialParams = PerMaterialFromLayerIdx->materialParams;
    params->blendFactor = make_float4(materialParams.velocity.macCormackBlendFactor);

    params->blendThreshold.x = materialParams.velocity.macCormackBlendThreshold;
    params->blendThreshold.y = materialParams.velocity.macCormackBlendThreshold;
    params->blendThreshold.z = materialParams.velocity.macCormackBlendThreshold;
    params->blendThreshold.w = infinity();

    params->damping.x = materialParams.velocity.damping;
    params->damping.y = materialParams.velocity.damping;
    params->damping.z = materialParams.velocity.damping;
    params->damping.w = 0.f;

    params->fade.x = materialParams.velocity.fade;
    params->fade.y = materialParams.velocity.fade;
    params->fade.z = materialParams.velocity.fade;
    params->fade.w = 0.f;

    advectCombustGetPerLayerImpl(params, layerIdx);
}

void Grid::updateModelMatrix() {
    auto T =
        matrixTranslation(m_currentLocation.x, m_currentLocation.y, m_currentLocation.z);
    auto S = matrixScaling(m_currentHalfSize.x, m_currentHalfSize.y, m_currentHalfSize.z);

    m_currentModelMatrix = S * T;
}

void Grid::vorticityConfinementPerLayer(VorticityConfinementPerLayerParams *params,
                                        Grid *userdata, uint32_t layerIdx) {
    userdata->vorticityConfinementPerLayerImpl(params, layerIdx);
}

void Grid::vorticityConfinementPerLayerImpl(VorticityConfinementPerLayerParams *params,
                                            uint32_t layerIdx) {
    auto PerMaterialFromLayerIdx = getPerMaterialFromLayerIdx(layerIdx);
    const auto &materialParams = PerMaterialFromLayerIdx->materialParams;
    params->forceScale = materialParams.vorticityStrength;
    params->velocityMask = materialParams.vorticityVelocityMask;
    params->temperatureMask = materialParams.vorticityTemperatureMask;
    params->smokeMask = materialParams.vorticitySmokeMask;
    params->fuelMask = materialParams.vorticityFuelMask;
    params->constantMask = materialParams.vorticityConstantMask;
}

void Grid::blockManagerPerLayer(BlockManagerPerLayerParams *params, Grid *userdata,
                                uint32_t layerIdx) {
    userdata->blockManagerPerLayerImpl(params, layerIdx);
}

void Grid::blockManagerPerLayerImpl(BlockManagerPerLayerParams *params, uint32_t layerIdx) {
    auto perMaterial = getPerMaterialFromLayerIdx(layerIdx);
    auto &materialParams = perMaterial->materialParams;
    params->velocityWeight = materialParams.velocity.allocWeight;
    params->smokeWeight = materialParams.smoke.allocWeight;
    params->tempWeight = materialParams.temperature.allocWeight;
    params->fuelWeight = materialParams.fuel.allocWeight;
    params->velocityThreshold = materialParams.velocity.allocThreshold;
    params->smokeThreshold = materialParams.smoke.allocThreshold;
    params->tempThreshold = materialParams.temperature.allocThreshold;
    params->fuelThreshold = materialParams.fuel.allocThreshold;
}

void Grid::doUpdate(NvFlowContext *context, float dt) {
    bool resetRequested = m_resetRequested;
    m_resetMode = resetRequested;

    updateModelMatrix();

    NvFlowFloat4x4 invModelMatrix = inverse(m_currentModelMatrix);

    EmitterData emitterData = {};
    EmitterAllocParams emitterAllocParams = {};
    emitterData.shapes = m_shapes.data();
    emitterData.numShapes = m_shapes.size();
    emitterData.numShapeRefs = m_shapeRefs;
    emitterData.params = m_emitters.data();
    emitterData.numParams = m_emitters.size();
    emitterData.lookups.emitMaterials = m_emitMaterials.data();
    emitterData.lookups.numEmitMaterials = m_emitMaterials.size();
    emitterData.lookups.sdfs = m_shapeSDFs.data();
    emitterData.lookups.numSdfs = m_shapeSDFs.size();

    emitterAllocParams.gridToWorld = m_currentModelMatrix;
    emitterAllocParams.worldToGrid = invModelMatrix;
    emitterAllocParams.getPerLayer =
        (void (*)(EmitterPerLayerParams *, void *, uint32_t))emitterPerLayer;
    emitterAllocParams.userdata = this;
    m_emitter->allocate(context, m_blockManager, &emitterAllocParams, &emitterData);
    m_emitter->allocateShape(context, m_blockManager, &emitterAllocParams, &emitterData);

    emitCustomAlloc(context);

    NvFlowResource *velocityParamsResource = 0;
    NvFlowUint4 velocityParamsCount = make_uint4(0);

    EmitterVelocityParams emitterVelocityParams = {};
    emitterVelocityParams.gridToWorld = m_currentModelMatrix;
    emitterVelocityParams.worldToGrid = invModelMatrix;
    emitterVelocityParams.virtualDim = m_velocity.getDesc().virtualDim;
    emitterVelocityParams.getPerLayer =
        (void (*)(EmitterPerLayerParams *, void *, uint32_t))emitterPerLayer;
    emitterVelocityParams.userdata = this;

    m_emitter->emitVelocityParameters(context, &emitterVelocityParams, &emitterData,
                                      &velocityParamsResource, &velocityParamsCount);

    NvFlowResource *densityParamsResource = 0;
    NvFlowUint4 densityParamsCount = make_uint4(0);

    EmitterDensityParams emitterDensityParams = {};
    emitterDensityParams.gridToWorld = m_currentModelMatrix;
    emitterDensityParams.worldToGrid = invModelMatrix;
    emitterDensityParams.virtualDim = m_density.getDesc().virtualDim;
    emitterDensityParams.getPerLayer =
        (void (*)(EmitterPerLayerParams *, void *, uint32_t))emitterPerLayer;
    emitterDensityParams.userdata = this;

    m_emitter->emitDensityParameters(context, &emitterDensityParams, &emitterData,
                                     &densityParamsResource, &densityParamsCount);

    NvFlowFloat3 densityVirtualDim = make_float3(m_density.getDesc().virtualDim);

    AdvectParams densityAdvectParams = {};
    densityAdvectParams.deltaTime = dt;
    densityAdvectParams.valueCellSize = (2.f * m_currentHalfSize) / densityVirtualDim;
    densityAdvectParams.gravity = m_params.gravity;
    densityAdvectParams.emitterCount = densityParamsCount;
    densityAdvectParams.gridToWorld = m_currentModelMatrix;
    densityAdvectParams.gridHalfSize = m_currentHalfSize;
    densityAdvectParams.gridNewLocation = m_currentLocation;
    densityAdvectParams.gridOldLocation = m_oldLocation;
    densityAdvectParams.singlePassAdvection = m_params.singlePassAdvection;
    densityAdvectParams.getPerLayer =
        (void (*)(AdvectPerLayerParams *, void *, uint32_t))advectDensityGetPerLayer;
    densityAdvectParams.userdata = this;

    auto fadeField = m_blockManager->getFadeField();

    m_advect->advectCombustDensity(context, &m_density, &m_velocity, &m_densityCoarse,
                                   fadeField, densityParamsResource, &densityAdvectParams);

    AdvectParams velocityAdvectParams = {};
    velocityAdvectParams.deltaTime = dt;
    velocityAdvectParams.valueCellSize =
        (2.f * m_currentHalfSize) / make_float3(m_desc.virtualDim);
    velocityAdvectParams.gravity = m_params.gravity;
    velocityAdvectParams.emitterCount = velocityParamsCount;
    velocityAdvectParams.gridToWorld = m_currentModelMatrix;
    velocityAdvectParams.gridHalfSize = m_currentHalfSize;
    velocityAdvectParams.gridNewLocation = m_currentLocation;
    velocityAdvectParams.gridOldLocation = m_oldLocation;
    velocityAdvectParams.singlePassAdvection = m_params.singlePassAdvection;
    velocityAdvectParams.getPerLayer =
        (void (*)(AdvectPerLayerParams *, void *, uint32_t))advectVelocityGetPerLayer;
    velocityAdvectParams.userdata = this;

    fadeField = m_blockManager->getFadeField();

    m_advect->advectCombustVelocity(context, &m_velocity, &m_density, &m_densityCoarse,
                                    fadeField, velocityParamsResource,
                                    &velocityAdvectParams);

    if (!resetRequested) {
        ZeroMemory(&emitterVelocityParams, sizeof(emitterVelocityParams));
        emitterVelocityParams.worldToGrid = invModelMatrix;
        emitterVelocityParams.virtualDim = m_velocity.getDesc().virtualDim;
        emitterVelocityParams.getPerLayer =
            (void (*)(EmitterPerLayerParams *, void *, uint32_t))emitterPerLayer;
        emitterVelocityParams.userdata = this;
        m_emitter->emitVelocity(context, &m_velocity, &emitterVelocityParams, &emitterData);

        ZeroMemory(&emitterDensityParams, sizeof(emitterDensityParams));
        emitterDensityParams.worldToGrid = invModelMatrix;
        emitterDensityParams.virtualDim = m_density.getDesc().virtualDim;
        emitterDensityParams.getPerLayer =
            (void (*)(EmitterPerLayerParams *, void *, uint32_t))emitterPerLayer;
        emitterDensityParams.userdata = this;
        m_emitter->emitDensity(context, &m_density, &m_densityCoarse, &emitterDensityParams,
                               &emitterData);
    }

    emitDebugVis();
    m_emitters.clear();
    m_shapes.clear();
    m_shapeRefs = 0;

    if (!resetRequested)
        emitCustomEmit(context);

    VorticityConfinementParams vcParams;
    vcParams.deltaTime = dt;
    vcParams.getPerLayer = (void (*)(VorticityConfinementPerLayerParams *, void *,
                                     uint32_t))vorticityConfinementPerLayer;
    vcParams.userdata = this;
    m_vorticityConfinement->execute(context, &m_velocity, &m_densityCoarse, &vcParams);

    PressureParams pressureParams;
    pressureParams.iterations = 8;
    pressureParams.legacyMode = m_params.pressureLegacyMode;
    fadeField = m_blockManager->getFadeField();
    m_pressureOp->execute(context, &m_velocity, &m_pressure, fadeField, &pressureParams);

    getGridExport(context);

    m_density.front.readPointHandleRelease(context);
    m_velocity.front.readPointHandleRelease(context);
    m_densityCoarse.front.readPointHandleRelease(context);

    if (resetRequested) {
        m_resetRequested = 0;
        m_currentLocation = m_resetDesc.initialLocation;
        m_currentHalfSize = m_resetDesc.halfSize;
        m_targetLocation = m_currentLocation;
    }
}

void Grid::emitCustomAlloc(NvFlowContext *context) {
    if (m_emitCustomAlloc.func) {
        NvFlowContextProfileItemBegin(context, L"customAlloc");

        NvFlowGridEmitCustomAllocParams allocParams = {};
        auto maskLayered = m_blockManager->map(context);

        for (uint32_t layerIdx = 0; layerIdx < maskLayered.numLayers; ++layerIdx) {
            auto maskTex = maskLayered.mapAccumLayer(layerIdx);
            allocParams.maskResourceRW = NvFlowTexture3DGetResourceRW(maskTex.mask);
            allocParams.maskDim = m_blockManager->getDim();
            allocParams.gridLocation = m_currentLocation;
            allocParams.gridHalfSize = m_currentHalfSize;
            allocParams.material = getMaterialHandleFromLayerIdx(layerIdx);
            m_emitCustomAlloc.func(m_emitCustomAlloc.userdata, &allocParams);
            maskLayered.unmapAccumLayer(layerIdx);
        }

        m_blockManager->unmap(context);
        NvFlowContextProfileItemEnd(context);
    }
}

void Grid::emitCustomEmit(NvFlowContext *context) {
    uint32_t numCallbacks = 2;
    EmitCustomEmitCallback *callbacks[2];
    SparseTextureFront *fields[2];
    const wchar_t *labels[2];

    callbacks[0] = &m_emitCustomEmit[0];
    callbacks[1] = &m_emitCustomEmit[1];
    fields[0] = &m_velocity;
    fields[1] = &m_density;
    labels[0] = L"customEmitVelocity";
    labels[1] = L"customEmitDensity";

    for (int i = 0; i < 2; ++i) {
        auto customEmit = callbacks[i];
        auto field = fields[i];

        if (customEmit->func) {
            NvFlowContextProfileItemBegin(context, labels[i]);

            auto tempTexture = field->acquireTexture(context);

            EmitCustomParams emitCustomParams = {};
            emitCustomParams.frontWriteLayeredHandle =
                field->front.writePointHandle(context);
            emitCustomParams.tempWriteLayeredHandle = tempTexture.writePointHandle(context);
            emitCustomParams.layeredView =
                emitCustomParams.frontWriteLayeredHandle.layeredView();
            emitCustomParams.field = field;

            NvFlowGridEmitCustomEmitParams emitParams = {};
            emitParams.grid = this;
            emitParams.numLayers = emitCustomParams.frontWriteLayeredHandle.numLayers;
            emitParams.flowInternal = &emitCustomParams;

            uint32_t frontIdx = 0;
            customEmit->func(customEmit->userdata, &frontIdx, &emitParams);
            if (frontIdx == 1) {
                field->swap(tempTexture);
            } else {
                tempTexture.releaseTexture();
            }

            NvFlowContextProfileItemEnd(context);
        }
    }
}

void Grid::emitDebugVis() {
    if (m_params.debugVisFlags & eNvFlowGridDebugVisEmitBounds) {
        m_debugVis.m_bounds.resize(m_emitters.size());
        for (uint32_t i = 0; i < m_emitters.size(); ++i)
            m_debugVis.m_bounds[i] = m_emitters[i].bounds;
    } else {
        m_debugVis.m_bounds.clear();
    }

    if (m_params.debugVisFlags & eNvFlowGridDebugVisShapesSimple) {
        m_gridExport.m_debugVis.debugVisFlags = m_params.debugVisFlags;
        emitDebugVisSimpleShapes(m_debugVis.m_spheres, eNvFlowShapeTypeSphere);
        emitDebugVisSimpleShapes(m_debugVis.m_capsules, eNvFlowShapeTypeCapsule);
        emitDebugVisSimpleShapes(m_debugVis.m_boxes, eNvFlowShapeTypeBox);
    } else {
        m_gridExport.m_debugVis.debugVisFlags = m_params.debugVisFlags;
        m_debugVis.m_spheres.clear();
        m_debugVis.m_capsules.clear();
        m_debugVis.m_boxes.clear();
    }
}

void Grid::emitDebugVisSimpleShapes(VectorCached<NvFlowGridExportSimpleShape, 16> &arr,
                                    NvFlowShapeType shapeType) {
    arr.clear();
    for (uint32_t i = 0; i < m_emitters.size(); ++i) {
        auto &emitter = m_emitters[i];
        if (emitter.shapeType == shapeType) {
            uint32_t allocIdx = arr.allocateBack();
            auto &shape = arr[allocIdx];
            shape.localToWorld = emitter.localToWorld;
            shape.shapeDesc = m_shapes[emitter.shapeRangeOffset];
        }
    }
}

NvFlowGridMaterialHandle Grid::emitMaterialIndexToMaterial(uint32_t emitMaterialIndex) {
    NvFlowGridMaterialHandle result;
    if (emitMaterialIndex < m_emitMaterials.size())
        result = m_emitMaterials[emitMaterialIndex];
    else
        ZeroMemory(&result, sizeof(result));
    return result;
}

void Grid::emitterPerLayer(EmitterPerLayerParams *params, Grid *userdata,
                           uint32_t layerIdx) {
    userdata->emitPerLayerImpl(params, layerIdx);
}

void Grid::emitPerLayerImpl(EmitterPerLayerParams *params, uint32_t layerIdx) {
    auto materialParams = getPerMaterialFromLayerIdx(layerIdx);
    params->materialIdx = getMaterialIdxFromLayerIdx(layerIdx);
}

NvFlowGridMaterialHandle Grid::getMaterialHandleFromLayerIdx(uint32_t layerIdx) {
    NvFlowGridMaterialHandle result;
    result.grid = this;
    result.uid = getMaterialIdxFromLayerIdx(layerIdx);
    return result;
}

uint32_t Grid::getMaterialIdxFromLayerIdx(uint32_t layerIdx) {
    if (layerIdx < m_layers.size()) {
        return m_layers[layerIdx].materialIdx;
    } else
        return 0;
}

Grid::PerMaterial *Grid::getPerMaterialFromLayerIdx(uint32_t layerIdx) {
    if (layerIdx >= m_layers.size())
        return &m_materials[0];

    auto &layer = m_layers[layerIdx];
    if (!layer.materialValid)
        return &m_materials[0];

    return &m_materials[layer.materialIdx];
}

Grid::PerMaterial *Grid::handleToPerMaterial(const NvFlowGridMaterialHandle &handle) {
    if (handle.grid == this && handle.uid < m_materials.size())
        return &m_materials[handle.uid];
    if (handle.uid)
        return nullptr;
    return &m_materials[0];
}

void Grid::reportDensityLayerNumBlocks(Grid *userdata, uint32_t numBlocks,
                                       uint32_t layerIdx) {
    userdata->reportDensityLayerNumBlocksImpl(numBlocks, layerIdx);
}

void Grid::reportDensityLayerNumBlocksImpl(uint32_t numBlocks, uint32_t layerIdx) {
    if (layerIdx < m_layers.size()) {
        auto &layer = m_layers[layerIdx];
        layer.densityNumBlocksOld = layer.densityNumBlocks;
        layer.densityNumBlocks = numBlocks;
    }
}

void Grid::reportSummaryUpdate(Grid *userdata) {
    userdata->reportSummaryUpdateImpl();
}

void Grid::reportSummaryUpdateImpl() {
    ++m_summaryCount;
}

void Grid::reportVelocityLayerNumBlocks(Grid *userdata, uint32_t numBlocks,
                                        uint32_t layerIdx) {
    return userdata->reportVelocityLayerNumBlocksImpl(numBlocks, layerIdx);
}

void Grid::reportVelocityLayerNumBlocksImpl(uint32_t numBlocks, uint32_t layerIdx) {
    if (layerIdx < m_layers.size()) {
        auto &layer = m_layers[layerIdx];
        layer.velocityNumBlocksOld = layer.velocityNumBlocks;
        layer.velocityNumBlocks = numBlocks;
    }
}

void Grid::updateLayers() {
    if (m_summaryCount) {
        for (auto &material : m_materials) {
            material.emitterAllocRefCount = 0;
        }
        m_summaryCount = 0;
    }

    for (auto &emitter : m_emitters) {
        auto materialHandle = emitMaterialIndexToMaterial(emitter.emitMaterialIndex);
        auto material = handleToPerMaterial(materialHandle);
        if (material) {
            bool nonZeroAllocScale = emitter.allocationScale.x > 0.f &&
                                     emitter.allocationScale.y > 0.f &&
                                     emitter.allocationScale.z > 0.f;
            bool defaultAllocMode =
                (emitter.emitMode & eNvFlowGridEmitModeDisableAlloc) == 0;
            bool shapeAllocMode = (emitter.emitMode & eNvFlowGridEmitModeAllocShape) != 0;
            if (nonZeroAllocScale && defaultAllocMode || shapeAllocMode)
                ++material->emitterAllocRefCount;
        }
    }

    auto sparseMapping = m_blockManager->getSparseMapping();
    for (uint32_t idx = 0; idx < m_materials.size(); ++idx) {
        auto &material = m_materials[idx];

        uint32_t layerIdx;
        if (material.emitterAllocRefCount) {
            for (layerIdx = 0; layerIdx < m_layers.size(); ++layerIdx) {
                auto &layer = m_layers[layerIdx];
                if (layer.materialValid && layer.materialIdx == idx)
                    break;
            }

            if (layerIdx == m_layers.size()) {
                for (layerIdx = 0; layerIdx < m_layers.size(); ++layerIdx) {
                    if (!m_layers[layerIdx].materialValid)
                        break;
                }

                if (layerIdx == m_layers.size())
                    layerIdx = m_layers.allocateBack();

                auto &layer = m_layers[layerIdx];
                layer.materialValid = 1;
                layer.materialIdx = idx;
                layer.velocityNumBlocks = 1;
                layer.velocityNumBlocksOld = 1;
                layer.densityNumBlocks = 1;
                layer.densityNumBlocksOld = 1;

                while (sparseMapping->getNumLayers() < m_layers.size())
                    sparseMapping->addLayer();
            }

            sparseMapping->enableLayer(layerIdx);
        } else {
            for (layerIdx = 0; layerIdx < m_layers.size(); ++layerIdx) {
                auto &layer = m_layers[layerIdx];
                if (layer.materialValid && layer.materialIdx == idx)
                    break;
            }

            if (layerIdx < m_layers.size()) {
                auto &layer = m_layers[layerIdx];
                if (layer.materialValid && !layer.velocityNumBlocks &&
                    !layer.velocityNumBlocksOld && !layer.densityNumBlocks &&
                    !layer.densityNumBlocksOld) {
                    sparseMapping->disableLayer(layerIdx);
                    layer.materialIdx = 0;
                    layer.materialValid = 0;
                }
            }
        }
    }
}

void Grid::GPUMemUsage(uint64_t *numBytes) {
    uint64_t totalBytes = 0;
    if (m_velocity.pool)
        totalBytes += m_velocity.pool->getGPUBytesUsed();

    if (m_density.pool)
        totalBytes += m_density.pool->getGPUBytesUsed();

    if (m_pressureOp)
        totalBytes += m_pressure.pool->getGPUBytesUsed();

    if (numBytes)
        *numBytes = totalBytes;
}

void Grid::update(NvFlowContext *context, float dt) {
    NvFlowContextProfileGroupBegin(context, L"GridUpdate");
    NvFlowContextTimerBegin(context, m_timer);

    const uint32_t numFields = 3;
    SparseMappable *fields[3];
    fields[0] = m_velocity.front.pool;
    fields[1] = m_density.front.pool;
    fields[2] = m_pressure.front.pool;
    m_blockManager->commit(&m_velocity, &m_density, context, fields, numFields);

    NvFlowFloat3 oldGridLocation = make_float3(0.f);
    NvFlowFloat3 newGridLocation = make_float3(0.f);
    if (m_blockManager->updateLocation(&oldGridLocation, &newGridLocation)) {
        m_oldLocation = oldGridLocation;
        m_currentLocation = newGridLocation;
        updateModelMatrix();
    } else {
        m_oldLocation = m_currentLocation;
    }

    updateLayers();
    doUpdate(context, dt);

    BlockManagerParams managerParams;
    managerParams.getPerLayer =
        (void (*)(BlockManagerPerLayerParams *, void *, uint32_t))blockManagerPerLayer;
    managerParams.reportVelocityLayerNumBlocks =
        (void (*)(void *, uint32_t, uint32_t))reportVelocityLayerNumBlocks;
    managerParams.reportDensityLayerNumBlocks =
        (void (*)(void *, uint32_t, uint32_t))reportDensityLayerNumBlocks;
    managerParams.reportSummaryUpdate = (void (*)(void *))reportSummaryUpdate;
    managerParams.userdata = this;
    managerParams.bigEffectMode = m_params.bigEffectMode;
    managerParams.bigEffectPredictTime = m_params.bigEffectPredictTime;
    managerParams.gridHalfSize = m_currentHalfSize;
    managerParams.gridLocation = m_currentLocation;
    managerParams.gridTargetLocation = m_targetLocation;
    m_blockManager->update(context, &m_velocity, &m_density, &m_densityCoarse, fields,
                           numFields, &managerParams);

    NvFlowContextTimerEnd(context, m_timer);

    NvFlowContextTimerGetResult(context, m_timer, &m_simTimeGPU, &m_simTimeCPU);

    NvFlowContextProfileGroupEnd(context);
}

void Grid::setTargetLocation(NvFlowFloat3 newLoc) {
    m_targetLocation = newLoc;
}

void Grid::setParams(const NvFlowGridParams *params) {
    m_params = *params;
}

NvFlowGridMaterialHandle Grid::getDefaultMaterial() {
    NvFlowGridMaterialHandle result;
    result.grid = this;
    result.uid = 0;
    return result;
}

NvFlowGridMaterialHandle Grid::createMaterial(
    const NvFlowGridMaterialParams *materialParams) {
    uint32_t allocIdx;
    for (allocIdx = 0; allocIdx < m_materials.size() && m_materials[allocIdx].valid;
         ++allocIdx)
        ;

    if (allocIdx == m_materials.size())
        allocIdx = m_materials.allocateBack();

    auto &material = m_materials[allocIdx];
    material = PerMaterial();
    material.materialParams = *materialParams;
    material.valid = 1;

    NvFlowGridMaterialHandle result;
    result.grid = this;
    result.uid = allocIdx;
    return result;
}

void Grid::releaseMaterial(NvFlowGridMaterialHandle handle) {
    auto material = handleToPerMaterial(handle);
    if (material)
        material->valid = 0;
}

void Grid::setMaterialParams(NvFlowGridMaterialHandle handle,
                             const NvFlowGridMaterialParams *materialParams) {
    auto material = handleToPerMaterial(handle);
    if (material)
        material->materialParams = *materialParams;
}

void Grid::emit(const NvFlowShapeDesc *shapes, uint32_t numShapes,
                const NvFlowGridEmitParams *params, uint32_t numParams) {
    uint32_t shapeIdx = m_shapes.size();
    for (uint32_t i = 0; i < numShapes; ++i) {
        uint32_t dstIdx = m_shapes.allocateBack();
        auto &shapeDesc = m_shapes[dstIdx];
        shapeDesc = shapes[i];
    }

    for (uint32_t k = 0; k < numParams; ++k) {
        uint32_t allocIdx = m_emitters.allocateBack();
        auto &emitterData = m_emitters[allocIdx];
        emitterData = params[k];
        emitterData.shapeRangeOffset += shapeIdx;
        m_shapeRefs += emitterData.shapeRangeSize;
    }
}

void Grid::updateEmitMaterials(NvFlowGridMaterialHandle *materials, uint32_t numMaterials) {
    m_emitMaterials.clear();
    m_emitMaterials.resize(numMaterials);
    for (uint32_t idx = 0; idx < numMaterials; ++idx) {
        m_emitMaterials[idx] = materials[idx];
    }
}

void Grid::updateEmitSDFs(NvFlowShapeSDF *const *sdfs, uint32_t numSdfs) {
    SafeRelease(m_shapeSDFs);
    m_shapeSDFs.clear();
    if (sdfs) {
        m_shapeSDFs.resize(numSdfs);
        for (uint32_t j = 0; j < numSdfs; ++j) {
            auto shapeSDF = sdfs[j];
            m_shapeSDFs[j] = shapeSDF;
            if (shapeSDF)
                shapeSDF->addRef();
        }
    }
}

void Grid::emitCustomRegisterAllocFunc(NvFlowGridEmitCustomAllocFunc func, void *userdata) {
    m_emitCustomAlloc.func = func;
    m_emitCustomAlloc.userdata = userdata;
}

void Grid::emitCustomRegisterEmitFunc(NvFlowGridTextureChannel channel,
                                      NvFlowGridEmitCustomEmitFunc func, void *userdata) {
    m_emitCustomEmit[channel].func = func;
    m_emitCustomEmit[channel].userdata = userdata;
}

void Grid::emitCustomGetLayerParams(const NvFlowGridEmitCustomEmitParams *emitParams,
                                    uint32_t layerIdx,
                                    NvFlowGridEmitCustomEmitLayerParams *emitLayerParams) {
    if (layerIdx < emitParams->numLayers) {
        auto emitCustomParams = (EmitCustomParams *)emitParams->flowInternal;
        auto frontWriteHandle =
            emitCustomParams->frontWriteLayeredHandle.layerView(layerIdx);
        auto tempWriteHandle = emitCustomParams->tempWriteLayeredHandle.layerView(layerIdx);
        uint32_t maxBlocks = emitCustomParams->field->getConfig().maxBlocks;
        if (emitLayerParams) {
            emitLayerParams->dataRW[0] = frontWriteHandle.data;
            emitLayerParams->dataRW[1] = tempWriteHandle.data;
            emitLayerParams->blockTable = frontWriteHandle.mapping.blockTable;
            emitLayerParams->blockList = frontWriteHandle.mapping.blockList;
            emitLayerParams->shaderParams = emitCustomParams->layeredView.params;
            emitLayerParams->numBlocks = frontWriteHandle.mapping.numBlocks;
            emitLayerParams->maxBlocks = maxBlocks;
            emitLayerParams->gridLocation = m_resetDesc.initialLocation;
            emitLayerParams->gridHalfSize = m_resetDesc.halfSize;
            emitLayerParams->material = getMaterialHandleFromLayerIdx(layerIdx);
        }
    }
}

NvFlowGridExport *Grid::getGridExport(NvFlowContext *context) {
    m_gridExport.getHandle(context, eNvFlowGridTextureChannelVelocity);
    m_gridExport.getHandle(context, eNvFlowGridTextureChannelDensity);
    m_gridExport.getHandle(context, eNvFlowGridTextureChannelDensityCoarse);
    return &m_gridExport;
}

}  // namespace NvFlow
