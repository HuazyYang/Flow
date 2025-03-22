#include "Emitter.h"
#include <NvFlowShader.h>
#include "Object.h"
#include "ClientHelper.h"
#include "SparseMapping.h"
#include "BlockManager.h"
#include "ShapeSDF.h"

namespace NvFlow {

struct EmitterVelocityShaderParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowShaderPointParams valueParams;
    NvFlowFloat4 vDimInv;
    uint32_t materialIdx;
    uint32_t matPad0;
    uint32_t matPad1;
    uint32_t matPad2;
    NvFlowFloat4 sdata[20];
    NvFlowFloat4 sshape[256];
};

struct EmitterAllocShaderParams {
    NvFlowUint4 emitterCount;
    NvFlowUint4 gridDim;
    uint32_t materialIdx;
    uint32_t matPad0;
    uint32_t matPad1;
    uint32_t matPad2;
};

struct EmitterDensityShaderParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowShaderPointParams valueParams;
    NvFlowShaderLinearParams coarseParams;
    NvFlowFloat4 vDimInv;
    uint32_t materialIdx;
    uint32_t matPad0;
    uint32_t matPad1;
    uint32_t matPad2;
    NvFlowFloat4 sdata[20];
    NvFlowFloat4 sshape[256];
};

struct EmitAllocShapeShaderParams {
    NvFlowFloat4 gridDimInv;
    NvFlowUint4 numShapes;
    NvFlowUint4 blockIdxOffset;
    uint32_t materialIdx;
    uint32_t matPad0;
    uint32_t matPad1;
    uint32_t matPad2;
};

struct EmitterImpl : Object, Emitter {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void allocate(NvFlowContext *context, BlockManager *blockManager,
                  const EmitterAllocParams *opParams, const EmitterData *emitData) override;
    void allocateShape(NvFlowContext *context, BlockManager *blockManager,
                       const EmitterAllocParams *opParams,
                       const EmitterData *emitData) override;
    void emitVelocityParameters(NvFlowContext *context,
                                const EmitterVelocityParams *opParams,
                                const EmitterData *emitData,
                                NvFlowResource **parameterResources,
                                NvFlowUint4 *parameterCount) override;

    void emitDensityParameters(NvFlowContext *context, const EmitterDensityParams *opParams,
                               const EmitterData *emitData,
                               NvFlowResource **parameterResources,
                               NvFlowUint4 *parameterCount) override;

    void emitVelocity(NvFlowContext *context, SparseTextureFront *velocity,
                      const EmitterVelocityParams *opParams,
                      const EmitterData *emitdata) override;
    void emitDensity(NvFlowContext *context, SparseTextureFront *density,
                     SparseTextureFront *coarseDensity,
                     const EmitterDensityParams *opParams,
                     const EmitterData *emitData) override;

    // Details
    EmitterImpl(NvFlowContext *context, const EmitterDesc *desc);
    ~EmitterImpl();

    void allocShapeFlush(NvFlowContext *context, SparseMappingLayerHandle *layerView,
                         uint32_t layerIdx, uint32_t materialIdx,
                         NvFlowResource *resourceSDF0, EmitAllocShapeShaderParams *mappedCB,
                         NvFlowFloat4 *mappedUB, const NvFlowFloat4 &gridDimInv,
                         const uint32_t cbShapesIdx, uint32_t cbShapeDataIdx,
                         uint32_t cbSDFCount, const NvFlowUint3 &batchBoundMini,
                         const NvFlowUint3 &batchBoundMaxi);

    NvFlowGridMaterialHandle emitMaterialIndexToMaterial(uint32_t emitMaterialIndex,
                                                         const EmitterLookups *lookups);

    void allocateShape(NvFlowContext *context, SparseMappingLayerHandle *layerView,
                       uint32_t layerIdx, uint32_t materialIdx,
                       const EmitterAllocParams *opParams, const EmitterData *emitData);

    void compute_bounds(NvFlowFloat4 *boundMinf, NvFlowFloat4 *boundMaxf,
                        const NvFlowFloat4 &vdimf,
                        const NvFlowFloat4x4 &emitterToGridBounds);

    NvFlowFloat4 genShapeData(NvFlowShapeType shapeType, const NvFlowShapeDesc *desc);

    void emitDensity_sdata(bool isStandalone, NvFlowFloat4x4 *sdata, NvFlowFloat4 *sshape,
                           NvFlowDim *boundMini, NvFlowDim *boundMaxi,
                           NvFlowFloat4 *vDimInv, const EmitterDensityParams *opParams,
                           const NvFlowShapeDesc *shapes,
                           const NvFlowGridEmitParams *params,
                           const EmitterLookups *lookups);

    void emitVelocity_sdata(bool isStandalone, NvFlowFloat4x4 *sdata, NvFlowFloat4 *sshape,
                            NvFlowDim *boundMini, NvFlowDim *boundMaxi,
                            NvFlowFloat4 *vDimInv, const EmitterVelocityParams *opParams,
                            const NvFlowShapeDesc *shapes,
                            const NvFlowGridEmitParams *params,
                            const EmitterLookups *lookups);

    EmitterDesc m_desc;
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowBuffer *m_uploadBuffer;
    NvFlowComputeShader *m_emitterDensityCS;
    NvFlowComputeShader *m_emitterVelocityCS;
    NvFlowComputeShader *m_emitterAllocCS;
    NvFlowComputeShader *m_emitterAllocShapeCS;
    NvFlowComputeShader *m_emitterAllocShapeCS_noSDF;
    NvFlowDim m_gridDim;
    NvFlowDim m_velocity_vdim;
    NvFlowDim m_density_vdim;
    NvFlowBuffer *m_allocBuffer;
    uint32_t m_allocBufferSize;
    NvFlowFloat4 *m_allocBufferData;
    uint32_t m_allocEmitterIndex;
    uint32_t m_allocEmitterMax;
    NvFlowBuffer *m_velocityBuffer;
    uint32_t m_velocityBufferSize;
    NvFlowFloat4 *m_velocityBufferData;
    uint32_t m_velocityEmitterIndex;
    uint32_t m_velocityEmitterMax;
    uint32_t m_velocityEmitterSeek;
    NvFlowBuffer *m_densityBuffer;
    uint32_t m_densityBufferSize;
    NvFlowFloat4 *m_densityBufferData;
    uint32_t m_densityEmitterIndex;
    uint32_t m_densityEmitterMax;
    uint32_t m_densityEmitterSeek;
};

#include "emitterDensityCS.hlsl.h"
#include "emitterVelocityCS.hlsl.h"
#include "emitterAllocCS.hlsl.h"
#include "emitterAllocShapeCS.hlsl.h"
#include "emitterAllocShapeCS_noSDF.hlsl.h"

void EmitterImpl::emitDensityParameters(NvFlowContext *context,
                                        const EmitterDensityParams *opParams,
                                        const EmitterData *emitData,
                                        NvFlowResource **parameterResources,
                                        NvFlowUint4 *parameterCount) {
    uint32_t neededBufferSize = emitData->numShapeRefs + 22 * emitData->numParams;
    uint32_t complexEmitterCount = 0;
    NvFlowBufferDesc bufDesc = {};

    m_density_vdim = opParams->virtualDim;
    m_densityEmitterMax = emitData->numParams;
    if (!m_densityBuffer) {
        bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
        bufDesc.dim = m_densityBufferSize;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_densityBuffer = NvFlowCreateBuffer(context, &bufDesc);
    }

    if (neededBufferSize > m_densityBufferSize) {
        while (neededBufferSize > m_densityBufferSize)
            m_densityBufferSize *= 2;

        NvFlowReleaseBuffer(m_densityBuffer);
        bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
        bufDesc.dim = m_densityBufferSize;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_densityBuffer = NvFlowCreateBuffer(context, &bufDesc);
    }

    m_densityBufferData = (NvFlowFloat4 *)NvFlowBufferMap(context, m_densityBuffer);
    m_densityEmitterIndex = 0;
    m_densityEmitterSeek = 2 * m_densityEmitterMax;

    for (uint32_t paramIndex = 0; paramIndex < emitData->numParams; ++paramIndex) {
        auto &params = emitData->params[paramIndex];
        if (params.shapeType &&
            (params.emitMode & eNvFlowGridEmitModeDisableDensity) == 0) {
            if (params.shapeType == eNvFlowShapeTypePlane)
                ++complexEmitterCount;

            NvFlowDim boundMini = make_dim(0);
            NvFlowDim boundMaxi = make_dim(0);
            NvFlowFloat4 vDimInv = make_float4(0);

            NvFlowUint4 headerData2;
            headerData2.x = m_densityEmitterSeek;
            headerData2.y = headerData2.x + 20;
            headerData2.z = headerData2.x + 20;
            headerData2.w = params.shapeRangeSize + headerData2.x + 20;
            NvFlowFloat4x4 *sdata = (NvFlowFloat4x4 *)(m_densityBufferData + headerData2.x);
            NvFlowFloat4 *sgeom = m_densityBufferData + headerData2.x + 20;
            NvFlowFloat4 *sshape = m_densityBufferData + headerData2.x + 20;
            if (sdata) {
                emitDensity_sdata(0, sdata, sshape, &boundMini, &boundMaxi, &vDimInv,
                                  opParams, emitData->shapes, &params, &emitData->lookups);
            }

            NvFlowUint4 headerData1;
            headerData1.x = (boundMaxi.x << 16) | boundMini.x;
            headerData1.y = (boundMaxi.y << 16) | boundMini.y;
            headerData1.z = (boundMaxi.z << 16) | boundMini.z;
            headerData1.w = m_densityEmitterIndex;

            memcpy(m_densityBufferData + m_densityEmitterIndex, &headerData1,
                   sizeof(headerData1));
            memcpy(m_densityBufferData + m_densityEmitterMax + m_densityEmitterIndex++,
                   &headerData2, sizeof(headerData2));
            m_densityEmitterSeek = headerData2.w;
        }
    }

    uint32_t copyBytes = sizeof(NvFlowFloat4) * neededBufferSize;
    NvFlowBufferUnmapRange(context, m_densityBuffer, 0, copyBytes);
    *parameterResources = NvFlowBufferGetResource(m_densityBuffer);

    parameterCount->x = m_densityEmitterIndex;
    parameterCount->y = m_densityEmitterMax;
    parameterCount->z = (m_densityEmitterIndex + 511) / 0x200;
    parameterCount->w = complexEmitterCount;
}

void EmitterImpl::emitVelocity(NvFlowContext *context, SparseTextureFront *velocity,
                               const EmitterVelocityParams *opParams,
                               const EmitterData *emitData) {
    for (uint32_t j = 0; j < emitData->numParams; ++j) {
        auto &params = emitData->params[j];
        if (params.shapeType == eNvFlowShapeTypeSDF &&
            (params.emitMode & eNvFlowGridEmitModeDisableVelocity) == 0) {
            const NvFlowShapeDesc *shapeFirst = &emitData->shapes[params.shapeRangeOffset];
            Shape *shapeSDF = 0;
            if (params.shapeType == eNvFlowShapeTypeSDF) {
                uint32_t sdfOffset = shapeFirst->sdf.sdfOffset;
                if (sdfOffset >= emitData->lookups.numSdfs) {
                    shapeSDF = nullptr;
                } else {
                    shapeSDF = implCast<Shape>(emitData->lookups.sdfs[sdfOffset]);
                }
            }

            auto tempTexture = velocity->acquireTexture(context);

            for (int passID = 0; passID < 2; ++passID) {
                SparseTextureHandle valueWriteTex = passID ? velocity->front : tempTexture;
                SparseTextureHandle valueReadTex = passID ? tempTexture : velocity->front;

                SparseWritePointHandle valueWriteHandle =
                    valueWriteTex.writePointHandle(context);
                SparseReadPointHandle valueReadHandle =
                    valueReadTex.readPointHandle(context);

                SparseWritePointLayeredView valueWriteLayeredView =
                    valueWriteHandle.layeredView();
                SparseReadPointLayeredView valueReadLayeredView =
                    valueReadHandle.layeredView();

                for (uint32_t layerIdx = 0; layerIdx < valueWriteHandle.numLayers;
                     ++layerIdx) {
                    EmitterPerLayerParams perLayer = {};
                    opParams->getPerLayer(&perLayer, opParams->userdata, layerIdx);

                    auto valueWriteLayerView = valueWriteHandle.layerView(layerIdx);
                    auto valueReadLayerView = valueReadHandle.layerView(layerIdx);

                    m_velocity_vdim = velocity->getDesc().virtualDim;

                    NvFlowDim boundMini = make_dim(0);
                    NvFlowDim boundMaxi = make_dim(0);
                    NvFlowFloat4 vDimInv = make_float4(0.f);

                    auto emitParams =
                        (EmitterVelocityShaderParams *)NvFlowConstantBufferMap(
                            context, m_constantBuffer);
                    if (emitParams) {
                        emitParams->gridToWorld = transpose(opParams->gridToWorld);
                        emitVelocity_sdata(1, (NvFlowFloat4x4 *)emitParams->sdata,
                                           emitParams->sshape, &boundMini, &boundMaxi,
                                           &vDimInv, opParams, emitData->shapes, &params,
                                           &emitData->lookups);
                        emitParams->valueParams = valueReadLayeredView.pointParams;
                        emitParams->vDimInv = vDimInv;
                        emitParams->materialIdx = perLayer.materialIdx;
                        NvFlowConstantBufferUnmap(context, m_constantBuffer);
                    }

                    NvFlowDim gridDim = (boundMaxi - boundMini + 7) >> 3;
                    if (valueWriteLayerView.mapping.numBlocks) {
                        NvFlowDispatchParams dispatchParams = {};
                        dispatchParams.shader = m_emitterVelocityCS;
                        dispatchParams.gridDim = gridDim;
                        dispatchParams.rootConstantBuffer = m_constantBuffer;
                        dispatchParams.readOnly[0] = valueWriteLayerView.mapping.blockTable;
                        dispatchParams.readOnly[1] = valueReadLayerView.data;
                        dispatchParams.readOnly[2] =
                            shapeSDF ? NvFlowTexture3DGetResource(shapeSDF->m_sdf)
                                     : nullptr;
                        dispatchParams.readWrite[0] = valueWriteLayerView.data;
                        NvFlowContextDispatch(context, &dispatchParams);
                    }
                }
            }
            tempTexture.releaseTexture();
        }
    }
}

void EmitterImpl::emitDensity(NvFlowContext *context, SparseTextureFront *density,
                              SparseTextureFront *coarseDensity,
                              const EmitterDensityParams *opParams,
                              const EmitterData *emitData) {
    for (uint32_t j = 0; j < emitData->numParams; ++j) {
        auto &params = emitData->params[j];
        if (params.shapeType == eNvFlowShapeTypeSDF &&
            (params.emitMode & eNvFlowGridEmitModeDisableDensity) == 0) {
            const NvFlowShapeDesc *shapeDesc = &emitData->shapes[params.shapeRangeOffset];
            Shape *shape = nullptr;
            if (params.shapeType == eNvFlowShapeTypeSDF) {
                uint32_t sdfOffset = shapeDesc->sdf.sdfOffset;
                if (sdfOffset >= emitData->lookups.numSdfs) {
                    shape = nullptr;
                } else {
                    auto shapeSDF = emitData->lookups.sdfs[sdfOffset];
                    if (shapeSDF)
                        shape = implCast<Shape>(shapeSDF);
                    else
                        shape = nullptr;
                }
            }

            NvFlowDim boundMini = make_dim(0);
            NvFlowDim boundMaxi = make_dim(0);
            NvFlowFloat4 vDimInv = make_float4(0.f);
            auto densityHandle = density->acquireTexture(context);

            m_density_vdim = density->getDesc().virtualDim;
            for (int k = 0; k < 2; ++k) {
                SparseTextureHandle *pWriteHandle, *pReadHandle;
                pWriteHandle = k ? &density->front : &densityHandle;
                pReadHandle = k ? &densityHandle : &density->front;

                auto writeHandle = pWriteHandle->writePointHandle(context);
                auto readHandle = pReadHandle->readPointHandle(context);

                SparseReadLinearHandle readCoarseHandle;

                if (coarseDensity) {
                    readCoarseHandle = coarseDensity->front.readLinearHandle(context);
                } else
                    ZeroMemory(&readCoarseHandle, sizeof(readCoarseHandle));

                SparseWritePointLayeredView writeLayered = writeHandle.layeredView();
                SparseReadPointLayeredView readLayered = readHandle.layeredView();

                SparseReadLinearLayeredView readCoarseLayered;

                if (coarseDensity)
                    readCoarseLayered = readCoarseHandle.layeredView();
                else
                    ZeroMemory(&readCoarseLayered, sizeof(readCoarseLayered));

                for (uint32_t layerIdx = 0; layerIdx < writeHandle.numLayers; ++layerIdx) {
                    EmitterPerLayerParams perLayer;
                    opParams->getPerLayer(&perLayer, opParams->userdata, layerIdx);
                    auto writeLayer = writeHandle.layerView(layerIdx);
                    auto readLayer = readHandle.layerView(layerIdx);

                    SparseReadLinearLayerView readCoarseLayer;
                    if (coarseDensity) {
                        readCoarseLayer = readCoarseHandle.layerView(layerIdx);
                    } else
                        ZeroMemory(&readCoarseLayer, sizeof(readCoarseLayer));

                    auto mappedCB = (EmitterDensityShaderParams *)NvFlowConstantBufferMap(
                        context, m_constantBuffer);
                    if (mappedCB) {
                        mappedCB->gridToWorld = transpose(opParams->gridToWorld);
                        emitDensity_sdata(1, (NvFlowFloat4x4 *)mappedCB->sdata,
                                          mappedCB->sshape, &boundMini, &boundMaxi,
                                          &vDimInv, opParams, emitData->shapes, &params,
                                          &emitData->lookups);
                        mappedCB->valueParams = readLayered.pointParams;
                        mappedCB->coarseParams = readCoarseLayered.params;
                        mappedCB->vDimInv = vDimInv;
                        mappedCB->materialIdx = perLayer.materialIdx;
                        NvFlowConstantBufferUnmap(context, m_constantBuffer);
                    }

                    NvFlowDim gridDim = (boundMaxi - boundMini + 7) >> 3;
                    if (writeLayer.mapping.numBlocks) {
                        NvFlowDispatchParams dparams = {};
                        dparams.shader = m_emitterDensityCS;
                        dparams.gridDim = gridDim;
                        dparams.rootConstantBuffer = m_constantBuffer;
                        dparams.readOnly[0] = writeLayer.mapping.blockTable;
                        dparams.readOnly[1] = readLayer.data;
                        dparams.readOnly[2] =
                            shape ? NvFlowTexture3DGetResource(shape->m_sdf) : nullptr;
                        dparams.readOnly[3] = readCoarseLayer.data;
                        dparams.readOnly[4] = readCoarseLayer.mapping.blockTable;
                        dparams.readWrite[0] = writeLayer.data;
                        NvFlowContextDispatch(context, &dparams);
                    }
                }
            }
            densityHandle.releaseTexture();
        }
    }
}

EmitterImpl::EmitterImpl(NvFlowContext *context, const EmitterDesc *desc)
    : m_constantBuffer(0),
      m_uploadBuffer(0),
      m_emitterDensityCS(0),
      m_emitterVelocityCS(0),
      m_emitterAllocCS(0),
      m_emitterAllocShapeCS(0),
      m_emitterAllocShapeCS_noSDF(0),
      m_gridDim{0, 0, 0},
      m_velocity_vdim{0, 0, 0},
      m_density_vdim{0, 0, 0},
      m_allocBuffer(0),
      m_allocBufferSize(13312),
      m_allocBufferData(0),
      m_allocEmitterIndex(0),
      m_allocEmitterMax(0),
      m_velocityBuffer(0),
      m_velocityBufferSize(21504),
      m_velocityBufferData(0),
      m_velocityEmitterIndex(0),
      m_velocityEmitterMax(0),
      m_velocityEmitterSeek(0),
      m_densityBuffer(0),
      m_densityBufferSize{21504},
      m_densityBufferData(0),
      m_densityEmitterIndex(0),
      m_densityEmitterMax(0),
      m_densityEmitterSeek(0) {
    m_desc = *desc;

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_emitterDensityCS = createShader(NVFLOW_CREATE_SHADER_ARGS(emitterDensityCS));
    m_emitterVelocityCS = createShader(NVFLOW_CREATE_SHADER_ARGS(emitterVelocityCS));
    m_emitterAllocCS = createShader(NVFLOW_CREATE_SHADER_ARGS(emitterAllocCS));
    m_emitterAllocShapeCS = createShader(NVFLOW_CREATE_SHADER_ARGS(emitterAllocShapeCS));
    m_emitterAllocShapeCS_noSDF =
        createShader(NVFLOW_CREATE_SHADER_ARGS(emitterAllocShapeCS_noSDF));

    uint32_t cbMaxSize = 0x10000;
    NvFlowConstantBufferDesc cbDesc;
    cbDesc.sizeInBytes = cbMaxSize;
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);

    NvFlowBufferDesc bufDesc = {};
    bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
    bufDesc.dim = cbMaxSize / sizeof(NvFlowFloat4);
    bufDesc.uploadAccess = 1;
    bufDesc.downloadAccess = 0;
    m_uploadBuffer = NvFlowCreateBuffer(context, &bufDesc);
}

EmitterImpl::~EmitterImpl() {
    SafeRelease(m_constantBuffer);
    SafeRelease(m_uploadBuffer);
    SafeRelease(m_emitterDensityCS);
    SafeRelease(m_emitterVelocityCS);
    SafeRelease(m_emitterAllocCS);
    SafeRelease(m_emitterAllocShapeCS);
    SafeRelease(m_emitterAllocShapeCS_noSDF);
    SafeRelease(m_allocBuffer);
    SafeRelease(m_velocityBuffer);
    SafeRelease(m_densityBuffer);
}

void EmitterImpl::allocShapeFlush(NvFlowContext *context,
                                  SparseMappingLayerHandle *layerView, uint32_t layerIdx,
                                  uint32_t materialIdx, NvFlowResource *resourceSDF0,
                                  EmitAllocShapeShaderParams *mappedCB,
                                  NvFlowFloat4 *mappedUB, const NvFlowFloat4 &gridDimInv,
                                  const uint32_t cbShapesIdx, uint32_t cbShapeDataIdx,
                                  uint32_t cbSDFCount, const NvFlowUint3 &batchBoundMini,
                                  const NvFlowUint3 &batchBoundMaxi) {
    mappedCB->gridDimInv = gridDimInv;
    mappedCB->numShapes = make_uint4(cbShapesIdx, cbShapeDataIdx, 0, 0);
    mappedCB->blockIdxOffset = make_uint4(batchBoundMini, 1);
    mappedCB->materialIdx = materialIdx;
    NvFlowConstantBufferUnmap(context, m_constantBuffer);
    NvFlowBufferUnmap(context, m_uploadBuffer);

    if (cbShapesIdx) {
        NvFlowDim gridDim = make_dim((batchBoundMaxi - batchBoundMini + 7) >> 3);
        NvFlowDispatchParams dispatchParams = {};
        dispatchParams.shader =
            (cbSDFCount) ? m_emitterAllocShapeCS : m_emitterAllocShapeCS_noSDF;
        dispatchParams.gridDim = gridDim;
        dispatchParams.rootConstantBuffer = m_constantBuffer;
        dispatchParams.readOnly[0] = NvFlowBufferGetResource(m_uploadBuffer);
        dispatchParams.readOnly[1] = resourceSDF0;
        dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(layerView->mask);
        NvFlowContextDispatch(context, &dispatchParams);
    }
}

NvFlowGridMaterialHandle EmitterImpl::emitMaterialIndexToMaterial(
    uint32_t emitMaterialIndex, const EmitterLookups *lookups) {
    NvFlowGridMaterialHandle result;
    if (emitMaterialIndex >= lookups->numEmitMaterials) {
        ZeroMemory(&result, sizeof(result));
    } else
        result = lookups->emitMaterials[emitMaterialIndex];

    return result;
}

void EmitterImpl::allocateShape(NvFlowContext *context, SparseMappingLayerHandle *layerView,
                                uint32_t layerIdx, uint32_t materialIdx,
                                const EmitterAllocParams *opParams,
                                const EmitterData *emitData) {
    NvFlowFloat4x4 worldToGrid = opParams->worldToGrid;
    NvFlowFloat4 gridDimInv = 1.f / make_float4(m_gridDim, 1.f);
    uint32_t cbShapeIdx = 0;
    uint32_t cbShapeDataIdx = 0;
    uint32_t cbSDFCount = 0;
    NvFlowResource *resourceSDF0 = 0;

    NvFlowUint3 batchBoundMini = make_uint3(m_gridDim);
    NvFlowUint3 batchBoundMaxi = make_uint3(0);
    uint32_t shapeDataOffset = 256;

    auto mappedCB =
        (EmitAllocShapeShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    auto mappedUB = (NvFlowFloat4 *)NvFlowBufferMap(context, m_uploadBuffer);

    for (uint32_t paramIdx = 0; paramIdx < emitData->numParams; ++paramIdx) {
        auto &params = emitData->params[paramIdx];
        if ((params.emitMode & eNvFlowGridEmitModeAllocShape) != 0) {
            if (cbShapeIdx + 1 > 32 || params.shapeRangeSize + cbShapeDataIdx > 512 ||
                cbSDFCount + 1 > 1 || params.shapeType == eNvFlowShapeTypeSDF) {
                allocShapeFlush(context, layerView, layerIdx, materialIdx, resourceSDF0,
                                mappedCB, mappedUB, gridDimInv, cbShapeIdx, cbShapeDataIdx,
                                cbSDFCount, batchBoundMini, batchBoundMaxi);

                cbShapeIdx = 0;
                cbShapeDataIdx = 0;
                cbSDFCount = 0;
                resourceSDF0 = 0;
                batchBoundMini = make_uint3(m_gridDim);
                batchBoundMaxi = make_uint3(0);
                mappedCB = (EmitAllocShapeShaderParams *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                mappedUB = (NvFlowFloat4 *)NvFlowBufferMap(context, m_uploadBuffer);
            }

            if (params.shapeType == eNvFlowShapeTypeSDF) {
                ++cbSDFCount;
                uint32_t sdfOffset =
                    emitData->shapes[params.shapeRangeOffset].sdf.sdfOffset;

                Shape *sdf;
                if (sdfOffset >= emitData->lookups.numSdfs) {
                    sdf = 0;
                } else {
                    NvFlowShapeSDF *shapeSDF = emitData->lookups.sdfs[sdfOffset];
                    if (shapeSDF)
                        sdf = implCast<Shape>(shapeSDF);
                    else
                        sdf = 0;
                }
                resourceSDF0 = sdf ? NvFlowTexture3DGetResource(sdf->m_sdf) : nullptr;
            }

            NvFlowFloat4x4 bounds = params.bounds;
            NvFlowFloat4x4 emitterToGridBounds = bounds * worldToGrid;
            NvFlowFloat4x4 gridToEmitterBounds = inverse(emitterToGridBounds);
            NvFlowFloat4x4 emitterToGrid = params.localToWorld * worldToGrid;
            NvFlowFloat4x4 gridToEmitter = inverse(emitterToGrid);

            NvFlowDim vdim = m_gridDim;
            NvFlowFloat4 vdimf = make_float4(m_gridDim, 1);
            NvFlowFloat4 boundMinf, boundMaxf;
            compute_bounds(&boundMinf, &boundMaxf, vdimf, emitterToGridBounds);

            NvFlowUint4 boundMini, boundMaxi;
            boundMini = make_uint4(boundMinf);
            boundMaxi = make_uint4(boundMaxf);
            if (boundMinf.x < batchBoundMini.x)
                batchBoundMini.x = boundMini.x;
            if (boundMinf.y < batchBoundMini.y)
                batchBoundMini.y = boundMini.y;
            if (boundMinf.z < batchBoundMini.z)
                batchBoundMini.z = boundMini.z;

            if (boundMaxf.x > batchBoundMaxi.x)
                batchBoundMaxi.x = boundMaxi.x;
            if (boundMaxf.x > batchBoundMaxi.y)
                batchBoundMaxi.y = boundMaxi.y;
            if (boundMaxf.z > batchBoundMaxi.z)
                batchBoundMaxi.z = boundMaxi.z;

            auto ptrUB = &mappedUB[8 * cbShapeIdx];
            memcpy(ptrUB, &gridToEmitter, sizeof(gridToEmitter));
            memcpy(&ptrUB[4], &boundMini, sizeof(NvFlowUint3));
            (uint32_t &)ptrUB[4].w = params.shapeType;
            memcpy(&ptrUB[5], &boundMaxi, sizeof(NvFlowUint3));
            ptrUB[5].w = params.shapeDistScale;
            (uint32_t &)ptrUB[6].x = cbShapeDataIdx;
            (uint32_t &)ptrUB[6].y = params.shapeRangeSize;
            ptrUB[6].z = params.minActiveDist;
            ptrUB[6].w = params.maxActiveDist;
            (uint32_t &)ptrUB[7].x =
                emitMaterialIndexToMaterial(params.emitMaterialIndex, &emitData->lookups)
                    .uid;

            ++cbShapeIdx;
            auto shapePtrUB = &mappedUB[shapeDataOffset];
            for (uint32_t shapeIdx = 0; shapeIdx < params.shapeRangeSize; ++shapeIdx) {
                auto shape = &emitData->shapes[shapeIdx + params.shapeRangeOffset];
                shapePtrUB[cbShapeDataIdx++] = genShapeData(params.shapeType, shape);
            }
        }
    }

    allocShapeFlush(context, layerView, layerIdx, materialIdx, resourceSDF0, mappedCB,
                    mappedUB, gridDimInv, cbShapeIdx, cbShapeDataIdx, cbSDFCount,
                    batchBoundMini, batchBoundMaxi);
}

void EmitterImpl::compute_bounds(NvFlowFloat4 *boundMinf, NvFlowFloat4 *boundMaxf,
                                 const NvFlowFloat4 &vdimf,
                                 const NvFlowFloat4x4 &emitterToGridBounds) {
    NvFlowFloat4 boundMin = make_float4(1.f);
    NvFlowFloat4 boundMax = make_float4(-1.f);

    for (int k = -1; k <= 1; k += 2)
        for (int j = -1; j <= 1; j += 2)
            for (int i = -1; i <= 1; i += 2) {
                auto valIn = make_float4(i, j, k, 1);
                auto val = transform4(valIn, emitterToGridBounds);
                boundMin = min(boundMin, val);
                boundMax = max(boundMax, val);
            }

    boundMin = 0.5f * boundMin + 0.5f;
    boundMax = 0.5f * boundMax + 0.5f;

    boundMin = boundMin * vdimf;
    boundMax = boundMax * vdimf;

    boundMin = floor(boundMin);
    boundMax = ceil(boundMax);

    boundMin = max(boundMin, make_float4(0.f));
    boundMax = min(boundMax, vdimf);

    *boundMinf = boundMin;
    *boundMaxf = boundMax;
}

NvFlowFloat4 EmitterImpl::genShapeData(NvFlowShapeType shapeType,
                                       const NvFlowShapeDesc *shape) {
    NvFlowFloat4 ret = make_float4(0.f);

    if (shapeType == eNvFlowShapeTypeSphere) {
        ret.x = shape->sphere.radius;
        ret.y = shape->sphere.radius;
        ret.z = shape->sphere.radius;
    } else {
        switch (shapeType) {
            case eNvFlowShapeTypeBox:
                ret.x = shape->box.halfSize.x;
                ret.y = shape->box.halfSize.y;
                ret.z = shape->box.halfSize.z;
                ret.w = 1.f;
                break;
            case eNvFlowShapeTypeCapsule:
                ret.x = shape->capsule.radius;
                ret.y = shape->capsule.length;
                ret.z = 0.f;
                ret.w = 1.f;
                break;
            case eNvFlowShapeTypePlane:
                memcpy(&ret, &shape->plane, sizeof(shape->plane));
                break;
        }
    }
    return ret;
}

void EmitterImpl::emitDensity_sdata(bool isStandalone, NvFlowFloat4x4 *sdata,
                                    NvFlowFloat4 *sshape, NvFlowDim *boundMini,
                                    NvFlowDim *boundMaxi, NvFlowFloat4 *vDimInv,
                                    const EmitterDensityParams *opParams,
                                    const NvFlowShapeDesc *shapes,
                                    const NvFlowGridEmitParams *params,
                                    const EmitterLookups *lookups) {
    auto &shapeArrayLocal = shapes[params->shapeRangeOffset];
    Shape *shapeSDF = 0;
    if (params->shapeType == eNvFlowShapeTypeSDF) {
        uint32_t sdfOffset = shapeArrayLocal.sdf.sdfOffset;
        if (sdfOffset >= lookups->numSdfs)
            shapeSDF = 0;
        else {
            shapeSDF = implCast<Shape>(lookups->sdfs[sdfOffset]);
        }
    }

    NvFlowFloat4x4 bounds = params->bounds;
    NvFlowFloat4x4 worldToGrid = opParams->worldToGrid;
    NvFlowFloat4x4 emitterToGridBounds = bounds * worldToGrid;
    NvFlowFloat4x4 gridToEmitterBounds = inverse(emitterToGridBounds);
    NvFlowDim vdim = m_density_vdim;
    NvFlowFloat4 vdimf = make_float4(vdim, 1);

    NvFlowFloat4 boundMinf, boundMaxf;
    compute_bounds(&boundMinf, &boundMaxf, vdimf, emitterToGridBounds);
    *boundMini = make_dim(boundMinf.x, boundMinf.y, boundMinf.z);
    *boundMaxi = make_dim(boundMaxf.x, boundMaxf.y, boundMaxf.z);
    *vDimInv = 1.f / vdimf;

    NvFlowDim gridDim = (*boundMaxi - *boundMini + 7) >> 3;

    float dt = 0.5f * params->deltaTime;
    if (!isStandalone)
        dt *= 2.f;

    NvFlowFloat4x4 emitterToGrid = params->localToWorld * worldToGrid;
    NvFlowFloat4x4 gridToEmitter = inverse(emitterToGrid);
    memcpy(sdata, &gridToEmitter, sizeof(gridToEmitter));

    sdata[1] = identity();
    sdata[2].x.x = 0.f;
    sdata[2].x.y = asfloat(params->shapeRangeSize);
    sdata[2].x.z = asfloat(
        uint32_t(emitMaterialIndexToMaterial(params->emitMaterialIndex, lookups).uid));
    sdata[2].y.x = params->minActiveDist;
    sdata[2].y.y = params->maxActiveDist;
    sdata[2].y.z = 1.f / abs(params->minEdgeDist);
    sdata[2].y.w = 1.f / abs(params->maxEdgeDist);
    memcpy(&sdata[2].z, boundMini, sizeof(NvFlowDim));
    sdata[2].z.w = asfloat(1);
    memcpy(&sdata[2].w, boundMaxi, sizeof(NvFlowDim));
    sdata[2].w.w = asfloat(int(params->shapeType));

    sdata[3].x = make_float4(0.f, 0.f, 0.f, params->shapeDistScale);
    sdata[3].y = make_float4(params->temperature, params->fuel, 0.f, params->smoke);
    sdata[3].z = make_float4(0.f);
    sdata[3].w = make_float4(0.f);

    sdata[4].x = make_float4(params->temperatureCoupleRate, params->fuelCoupleRate, 0.f,
                             params->smokeCoupleRate);
    sdata[4].y = make_float4(dt);
    sdata[4].z = make_float4(params->slipThickness, params->slipFactor,
                             params->fuelReleaseTemp, params->fuelRelease);

    for (uint32_t shapeIdx = 0; shapeIdx < params->shapeRangeSize; ++shapeIdx) {
        auto shape = &shapes[shapeIdx + params->shapeRangeOffset];
        sshape[shapeIdx] = genShapeData(params->shapeType, shape);
    }
}

void EmitterImpl::emitVelocity_sdata(bool isStandalone, NvFlowFloat4x4 *sdata,
                                     NvFlowFloat4 *sshape, NvFlowDim *boundMini,
                                     NvFlowDim *boundMaxi, NvFlowFloat4 *vDimInv,
                                     const EmitterVelocityParams *opParams,
                                     const NvFlowShapeDesc *shapes,
                                     const NvFlowGridEmitParams *params,
                                     const EmitterLookups *lookups) {
    auto &shapeArrayLocal = shapes[params->shapeRangeOffset];
    Shape *shapeSDF = 0;
    if (params->shapeType == eNvFlowShapeTypeSDF) {
        uint32_t sdfOffset = shapeArrayLocal.sdf.sdfOffset;
        if (sdfOffset >= lookups->numSdfs)
            shapeSDF = 0;
        else {
            shapeSDF = implCast<Shape>(lookups->sdfs[sdfOffset]);
        }
    }

    NvFlowFloat4x4 bounds = params->bounds;
    NvFlowFloat4x4 worldToGrid = opParams->worldToGrid;
    NvFlowFloat4x4 emitterToGridBounds = bounds * worldToGrid;
    NvFlowFloat4x4 gridToEmitterBounds = inverse(emitterToGridBounds);
    NvFlowDim vdim = m_velocity_vdim;
    NvFlowFloat4 vdimf = make_float4(vdim, 1);

    NvFlowFloat4 sdfHalfDimInv = make_float4(0.015625f, 0.015625f, 0.015625, 1.f);
    if (shapeSDF) {
        (NvFlowFloat3 &)sdfHalfDimInv = 2.f / make_float3(shapeSDF->m_desc.resolution);
    }

    NvFlowFloat4 boundMinf, boundMaxf;
    compute_bounds(&boundMinf, &boundMaxf, vdimf, emitterToGridBounds);
    *boundMini = make_dim(boundMinf.x, boundMinf.y, boundMinf.z);
    *boundMaxi = make_dim(boundMaxf.x, boundMaxf.y, boundMaxf.z);

    auto A = bounds;
    auto x = transform4(make_float4(params->centerOfMass, 1.f), A);
    auto ANormalized = matrixNormalize(A);
    auto velocityLinearGridSpace =
        transform4(make_float4(params->velocityLinear, 0.f), ANormalized);
    auto velocityAngularGridspace =
        transform4(make_float4(params->velocityAngular, 0.f), ANormalized);

    *vDimInv = 1.f / vdimf;

    float dt = 0.5f * params->deltaTime;
    if (!isStandalone)
        dt *= 2.f;

    NvFlowDim gridDim = (*boundMaxi - *boundMini + 7) >> 3;

    NvFlowFloat4x4 emitterToGrid = params->localToWorld * worldToGrid;
    NvFlowFloat4x4 gridToEmitter = inverse(emitterToGrid);
    memcpy(sdata, &gridToEmitter, sizeof(gridToEmitter));
    sdata[1] = ANormalized;

    sdata[2].x.x = 0.f;
    sdata[2].x.y = asfloat(params->shapeRangeSize);
    sdata[2].x.z = asfloat(
        uint32_t(emitMaterialIndexToMaterial(params->emitMaterialIndex, lookups).uid));
    sdata[2].y.x = params->minActiveDist;
    sdata[2].y.y = params->maxActiveDist;
    sdata[2].y.z = 1.f / abs(params->minEdgeDist);
    sdata[2].y.w = 1.f / abs(params->maxEdgeDist);
    memcpy(&sdata[2].z, boundMini, sizeof(NvFlowDim));
    sdata[2].z.w = asfloat(1);
    memcpy(&sdata[2].w, boundMaxi, sizeof(NvFlowDim));
    sdata[2].w.w = asfloat(int(params->shapeType));

    memcpy(&sdata[3].x, &sdfHalfDimInv, sizeof(NvFlowFloat3));
    sdata[3].x.w = params->shapeDistScale;
    sdata[3].y = velocityLinearGridSpace;
    sdata[3].z = velocityAngularGridspace;
    sdata[3].w = x;

    sdata[4].x = make_float4(params->velocityCoupleRate, 0.f);
    sdata[4].y = make_float4(dt);
    sdata[4].z = make_float4(params->slipThickness, params->slipFactor,
                             params->fuelReleaseTemp, params->fuelRelease);

    for (uint32_t j = 0; j < params->shapeRangeSize; ++j) {
        auto shape = &shapes[j + params->shapeRangeOffset];
        sshape[j] = genShapeData(params->shapeType, shape);
    }
}

uint64_t EmitterImpl::getGPUBytesUsed() {
    return 0;
}

void EmitterImpl::allocate(NvFlowContext *context, BlockManager *blockManager,
                           const EmitterAllocParams *opParams,
                           const EmitterData *emitData) {
    m_gridDim = blockManager->getDim();
    m_allocEmitterMax = emitData->numParams;
    uint32_t allocEmitterCapacity = 13 * m_allocEmitterMax;

    NvFlowBufferDesc bufDesc = {};
    if (!m_allocBuffer) {
        bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
        bufDesc.dim = m_allocBufferSize;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_allocBuffer = NvFlowCreateBuffer(context, &bufDesc);
    }

    if (allocEmitterCapacity > m_allocBufferSize) {
        while (allocEmitterCapacity > m_allocBufferSize)
            m_allocBufferSize *= 2;
        bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
        bufDesc.dim = m_allocBufferSize;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_allocBuffer = NvFlowCreateBuffer(context, &bufDesc);
    }

    m_allocBufferData = (NvFlowFloat4 *)NvFlowBufferMap(context, m_allocBuffer);
    m_allocEmitterIndex = 0;

    for (uint32_t j = 0; j < emitData->numParams; ++j) {
        auto &param = emitData->params[j];
        if ((param.emitMode & eNvFlowGridEmitModeDisableAlloc) == 0) {
            NvFlowFloat3 allocationScale = param.allocationScale;
            NvFlowFloat4x4 bounds;
            bounds.x = param.bounds.x * allocationScale.x;
            bounds.y = param.bounds.y * allocationScale.y;
            bounds.z = param.bounds.z * allocationScale.z;
            bounds.w = param.bounds.w;

            NvFlowFloat4 gridDim = make_float4(m_gridDim, 1);
            NvFlowFloat4 invGridDim = 1.f / make_float4(m_gridDim, 1);
            NvFlowFloat3 gridSpacing;
            gridSpacing.x = invGridDim.x * vector3Length(opParams->gridToWorld.x);
            gridSpacing.y = invGridDim.y * vector3Length(opParams->gridToWorld.y);
            gridSpacing.z = invGridDim.z * vector3Length(opParams->gridToWorld.z);
            float maxGridSpacing = 1.5f * max3(gridSpacing.x, gridSpacing.y, gridSpacing.z);

            NvFlowFloat3 emitterToWorldScale;
            emitterToWorldScale.x = vector3Length(bounds.x);
            emitterToWorldScale.y = vector3Length(bounds.y);
            emitterToWorldScale.z = vector3Length(bounds.z);
            NvFlowFloat3 worldToGridScale = maxGridSpacing / emitterToWorldScale;
            worldToGridScale = max(worldToGridScale, make_float3(1.f));

            NvFlowFloat4x4 emitterToWorld;
            emitterToWorld.x = bounds.x * worldToGridScale.x;
            emitterToWorld.y = bounds.y * worldToGridScale.y;
            emitterToWorld.z = bounds.z * worldToGridScale.z;
            emitterToWorld.w = bounds.w;

            NvFlowFloat4x4 ndcToWorld = emitterToWorld;
            if (param.allocationPredict > 0.f) {
                float predictVelocityWeight = param.predictVelocityWeight;
                NvFlowFloat3 velocity = lerp(param.velocityLinear, param.predictVelocity,
                                             predictVelocityWeight);
                float velocityAbsSum = abs(velocity.x) + abs(velocity.y) + abs(velocity.z);
                if (velocityAbsSum > 0.f) {
                    NvFlowFloat4x4 A = matrixNormalize(emitterToWorld);
                    auto x = make_float3(transform4(make_float4(velocity, 0.f), A));
                    x = x * param.allocationPredict;
                    auto xAxis = normalize(x);
                    auto xAbs = abs(x);

                    NvFlowFloat3 yAxis, zAxis;
                    if (xAbs.x >= xAbs.y) {
                        if (xAbs.x >= xAbs.z) {
                            yAxis = make_float3(0.f, 1.f, 0.f);
                            zAxis = make_float3(0.f, 0.f, 1.f);
                        }
                    }
                    if (xAbs.y >= xAbs.x) {
                        if (xAbs.y >= xAbs.z) {
                            yAxis = make_float3(1.f, 0.f, 0.f);
                            zAxis = make_float3(0.f, 0.f, 1.f);
                        }
                    }
                    if (xAbs.z >= xAbs.x) {
                        if (xAbs.z >= xAbs.y) {
                            yAxis = make_float3(1.f, 0.f, 0.f);
                            zAxis = make_float3(0.f, 1.f, 0.f);
                        }
                    }

                    yAxis = normalize(yAxis - dot(yAxis, xAxis) * xAxis);
                    zAxis = normalize(zAxis - dot(zAxis, yAxis) * yAxis);
                    zAxis = normalize(zAxis - dot(zAxis, xAxis) * xAxis);

                    NvFlowFloat4x4 ndcToAllocation2;
                    ndcToAllocation2.x = make_float4(x, 0.f);
                    ndcToAllocation2.y = make_float4(yAxis, 0.f);
                    ndcToAllocation2.z = make_float4(zAxis, 0.f);
                    ndcToAllocation2.w = make_float4(0.f, 0.f, 0.f, 1.f);
                    NvFlowFloat4x4 allocationToNdc2 = inverse(ndcToAllocation2);
                    NvFlowFloat4 origin =
                        transform4(make_float4(0.f, 0.f, 0.f, 1.f), emitterToWorld);
                    NvFlowFloat4 ndcMin = transform4(origin, allocationToNdc2);
                    NvFlowFloat4 ndcMax = ndcMin;

                    for (int k = -1; k <= 1; k += 2)
                        for (int m = -1; m <= 1; m += 2)
                            for (int n = -1; n <= 1; n += 2) {
                                auto pt1 =
                                    transform4(make_float4(n, m, k, 1), emitterToWorld);
                                auto pt2 = pt1 + make_float4(x, 0.f);
                                auto x1 = transform4(pt1, allocationToNdc2);
                                auto x2 = transform4(pt2, allocationToNdc2);
                                ndcMin = min(ndcMin, x1);
                                ndcMax = max(ndcMax, x1);
                                ndcMin = min(ndcMin, x2);
                                ndcMax = max(ndcMax, x2);
                            }

                    NvFlowFloat4 ndcRange = ndcMax - ndcMin;
                    NvFlowFloat4 ndcHalfRange = 0.5f * ndcRange;
                    NvFlowFloat4 ndcCenter = 0.5f * (ndcMin + ndcMax);
                    NvFlowFloat4x4 ndcNormalMat;
                    ndcNormalMat.x = make_float4(ndcHalfRange.x, 0.f, 0.f, 0.f);
                    ndcNormalMat.y = make_float4(0.f, ndcHalfRange.y, 0.f, 0.f);
                    ndcNormalMat.z = make_float4(0.f, 0.f, ndcHalfRange.z, 0.f);
                    ndcNormalMat.w =
                        make_float4(ndcCenter.x, ndcCenter.y, ndcCenter.z, 1.f);

                    ndcToWorld = ndcNormalMat * ndcToAllocation2;
                }
            }

            NvFlowFloat4x4 emitterToGrid = emitterToWorld * opParams->worldToGrid;
            NvFlowFloat4x4 ndcToGrid = ndcToWorld * opParams->worldToGrid;
            NvFlowFloat4x4 gridToEmitter = inverse(emitterToGrid);
            NvFlowFloat4x4 gridToNdc = inverse(ndcToGrid);

            NvFlowFloat4 worldMin = make_float4(1.f);
            NvFlowFloat4 worldMax = make_float4(-1.f);
            for (int k = -1; k <= 1; k += 2)
                for (int m = -1; m <= 1; m += 2)
                    for (int n = -1; n <= 1; n += 2) {
                        NvFlowFloat4 corner = make_float4(n, m, k, 1);
                        NvFlowFloat4 cornerWorld = transform4(corner, emitterToGrid);
                        worldMin = min(worldMin, cornerWorld);
                        worldMax = max(worldMax, cornerWorld);
                        cornerWorld = transform4(corner, ndcToGrid);
                        worldMin = min(worldMin, cornerWorld);
                        worldMax = max(worldMax, cornerWorld);
                    }

            worldMin = worldMin * 0.5f + 0.5f;
            worldMax = worldMax * 0.5f + 0.5f;
            worldMin = worldMin * gridDim;
            worldMax = worldMax * gridDim;
            worldMin = floor(worldMin);
            worldMax = ceil(worldMax);
            worldMin = max(worldMin, make_float4(0.f));
            worldMax = min(worldMax, gridDim);

            NvFlowInt4 worldMini = make_int4(worldMin.x, worldMin.y, worldMin.z, 0);
            NvFlowInt4 worldMaxi = make_int4(worldMax.x, worldMax.y, worldMax.z, 0);

            if (allocationScale.x > 0.f && allocationScale.y > 0.f &&
                allocationScale.z > 0.f) {
                NvFlowUint4 packedData;
                packedData.x = (worldMaxi.x << 16) | worldMini.x;
                packedData.y = (worldMaxi.y << 16) | worldMini.y;
                packedData.z = (worldMaxi.z << 16) | worldMini.z;
                packedData.w = m_allocEmitterIndex;
                memcpy(&m_allocBufferData[m_allocEmitterIndex], &packedData,
                       sizeof(packedData));

                auto data =
                    m_allocBufferData + 12 * m_allocEmitterIndex + m_allocEmitterMax;
                if (data) {
                    memcpy(data, &gridToEmitter, sizeof(gridToEmitter));
                    memcpy(data + 4, &gridToNdc, sizeof(gridToNdc));
                    memcpy(data + 8, &worldMini, sizeof(worldMini));
                    memcpy(data + 9, &worldMaxi, sizeof(worldMaxi));
                    memcpy(data + 10, &invGridDim, sizeof(invGridDim));

                    NvFlowUint4 emitMaterialIndex;
                    emitMaterialIndex.x = emitMaterialIndexToMaterial(
                                              param.emitMaterialIndex, &emitData->lookups)
                                              .uid;
                    emitMaterialIndex.y = 0;
                    emitMaterialIndex.z = 0;
                    emitMaterialIndex.w = 0;
                    memcpy(data + 11, &emitMaterialIndex, sizeof(emitMaterialIndex));
                }
                ++m_allocEmitterIndex;
            }
        }
    }

    uint32_t numBytes =
        sizeof(NvFlowFloat4) * (12 * m_allocEmitterIndex + m_allocEmitterMax);
    NvFlowBufferUnmapRange(context, m_allocBuffer, 0, numBytes);

    auto blockMapped = blockManager->map(context);
    for (uint32_t layerIdx = 0; layerIdx < blockMapped.numLayers; ++layerIdx) {
        EmitterPerLayerParams emitLayerParams;
        opParams->getPerLayer(&emitLayerParams, opParams->userdata, layerIdx);
        auto mappedLayer = blockMapped.mapAccumLayer(layerIdx);
        if (mappedLayer.enable) {
            EmitterAllocShaderParams *emitParams =
                (EmitterAllocShaderParams *)NvFlowConstantBufferMap(context,
                                                                    m_constantBuffer);
            if (emitParams) {
                emitParams->emitterCount.x = m_allocEmitterIndex;
                emitParams->emitterCount.y = m_allocEmitterMax;
                emitParams->emitterCount.z = (m_allocEmitterIndex + 511) / 0x200;
                emitParams->emitterCount.w = 0;
                emitParams->gridDim = make_uint4(m_gridDim, 0);
                emitParams->materialIdx = emitLayerParams.materialIdx;
                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }
            NvFlowDim gridDim = (m_gridDim + 7) >> 3;
            NvFlowDispatchParams params = {};
            params.shader = m_emitterAllocCS;
            params.gridDim = gridDim;
            params.rootConstantBuffer = m_constantBuffer;
            params.readOnly[0] = NvFlowBufferGetResource(m_allocBuffer);
            params.readWrite[0] = NvFlowTexture3DGetResourceRW(mappedLayer.mask);
            NvFlowContextDispatch(context, &params);
        }
        blockMapped.unmapAccumLayer(layerIdx);
    }
    blockManager->unmap(context);
}

void EmitterImpl::allocateShape(NvFlowContext *context, BlockManager *blockManager,
                                const EmitterAllocParams *opParams,
                                const EmitterData *emitData) {
    m_gridDim = blockManager->getDim();
    auto maskLayered = blockManager->map(context);
    for (uint32_t layerIdx = 0; layerIdx < maskLayered.numLayers; ++layerIdx) {
        EmitterPerLayerParams perLayer = {};
        opParams->getPerLayer(&perLayer, opParams->userdata, layerIdx);
        auto mask = maskLayered.mapAccumLayer(layerIdx);
        if (mask.enable) {
            allocateShape(context, &mask, layerIdx, perLayer.materialIdx, opParams,
                          emitData);
        }
        maskLayered.unmapAccumLayer(layerIdx);
    }
    blockManager->unmap(context);
}

void EmitterImpl::emitVelocityParameters(NvFlowContext *context,
                                         const EmitterVelocityParams *opParams,
                                         const EmitterData *emitData,
                                         NvFlowResource **parameterResources,
                                         NvFlowUint4 *parameterCount) {
    uint32_t neededBufferSize = emitData->numShapeRefs + 22 * emitData->numParams;
    uint32_t complexEmitterCount = 0;
    NvFlowBufferDesc bufDesc = {};

    m_velocity_vdim = opParams->virtualDim;
    m_velocityEmitterMax = emitData->numParams;
    if (!m_velocityBuffer) {
        bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
        bufDesc.dim = m_velocityBufferSize;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_velocityBuffer = NvFlowCreateBuffer(context, &bufDesc);
    }

    if (neededBufferSize > m_velocityBufferSize) {
        while (neededBufferSize > m_velocityBufferSize)
            m_velocityBufferSize *= 2;

        NvFlowReleaseBuffer(m_velocityBuffer);
        bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
        bufDesc.dim = m_velocityBufferSize;
        bufDesc.uploadAccess = 1;
        bufDesc.downloadAccess = 0;
        m_velocityBuffer = NvFlowCreateBuffer(context, &bufDesc);
    }

    m_velocityBufferData = (NvFlowFloat4 *)NvFlowBufferMap(context, m_velocityBuffer);
    m_velocityEmitterIndex = 0;
    m_velocityEmitterSeek = 2 * m_velocityEmitterMax;

    for (uint32_t paramIndex = 0; paramIndex < emitData->numParams; ++paramIndex) {
        auto &params = emitData->params[paramIndex];
        if (params.shapeType &&
            (params.emitMode & eNvFlowGridEmitModeDisableVelocity) == 0) {
            if (params.shapeType == eNvFlowShapeTypePlane)
                ++complexEmitterCount;

            NvFlowDim boundMini = make_dim(0);
            NvFlowDim boundMaxi = make_dim(0);
            NvFlowFloat4 vDimInv = make_float4(0);

            NvFlowUint4 headerData2;
            headerData2.x = m_velocityEmitterSeek;
            headerData2.y = headerData2.x + 20;
            headerData2.z = headerData2.x + 20;
            headerData2.w = params.shapeRangeSize + headerData2.x + 20;
            NvFlowFloat4x4 *sdata =
                (NvFlowFloat4x4 *)(m_velocityBufferData + headerData2.x);
            NvFlowFloat4 *sgeom = m_velocityBufferData + headerData2.x + 20;
            NvFlowFloat4 *sshape = m_velocityBufferData + headerData2.x + 20;
            if (sdata) {
                emitVelocity_sdata(0, sdata, sshape, &boundMini, &boundMaxi, &vDimInv,
                                   opParams, emitData->shapes, &params, &emitData->lookups);
            }

            NvFlowUint4 headerData1;
            headerData1.x = (boundMaxi.x << 16) | boundMini.x;
            headerData1.y = (boundMaxi.y << 16) | boundMini.y;
            headerData1.z = (boundMaxi.z << 16) | boundMini.z;
            headerData1.w = m_velocityEmitterIndex;

            memcpy(m_velocityBufferData + m_velocityEmitterIndex, &headerData1,
                   sizeof(headerData1));
            memcpy(m_velocityBufferData + m_velocityEmitterMax + m_velocityEmitterIndex++,
                   &headerData2, sizeof(headerData2));
            m_velocityEmitterSeek = headerData2.w;
        }
    }

    uint32_t copyBytes = sizeof(NvFlowFloat4) * neededBufferSize;
    NvFlowBufferUnmapRange(context, m_velocityBuffer, 0, copyBytes);
    *parameterResources = NvFlowBufferGetResource(m_velocityBuffer);

    parameterCount->x = m_velocityEmitterIndex;
    parameterCount->y = m_velocityEmitterMax;
    parameterCount->z = (m_velocityEmitterIndex + 511) / 0x200;
    parameterCount->w = complexEmitterCount;
}

Emitter *createEmitter(NvFlowContext *context, const EmitterDesc *desc) {
    return new EmitterImpl(context, desc);
}

}  // namespace NvFlow