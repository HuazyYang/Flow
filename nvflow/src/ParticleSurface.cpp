#include "ParticleSurface.h"
#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ParticleSurfaceExport.h"
#include "ClientHelper.h"

namespace NvFlow {

struct ParticleSurfaceAllocParams {
    NvFlowFloat4x4 positionToMask;
    NvFlowUint4 maskDim;
    NvFlowUint4 particleCount;
};

struct ParticleSurfaceSmoothParams {
    NvFlowShaderLinearParams linearParams;
    unsigned int dispatchNumBlocks;
    unsigned int padd1;
    unsigned int padd2;
    unsigned int padd3;
    NvFlowFloat4 kernel[5];
};

struct ParticleSurfaceUpdateLinearParams {
    NvFlowShaderLinearParams params;
    unsigned int dispatchNumBlocks;
    float surfaceThreshold;
    unsigned int padd2;
    unsigned int padd3;
};

struct ParticleSurfaceEmitEmitParams {
    NvFlowShaderPointParams customEmitParams;
    NvFlowShaderLinearParams surfaceParams;
    NvFlowFloat4x4 gridToField;
    NvFlowFloat4 vdimInv;
    NvFlowFloat4 deltaTime;
    NvFlowFloat4 coupleRate;
    NvFlowFloat4 emitValue;
    float surfaceThreshold;
    float padd1;
    float padd2;
    float padd3;
};

struct ParticleSurfaceBlockTableAllocParams {
    NvFlowUint4 gridDim;
    NvFlowUint4 poolGridDim;
};

struct ParticleSurfaceSplatParams {
    NvFlowFloat4x4 positionToField;
    NvFlowShaderLinearParams linearParams;
    NvFlowUint4 fieldDim;
    NvFlowUint4 particleCount;
};

uint64_t ParticleSurface::getGPUBytesUsed() {
    return 0;
}

void ParticleSurface::updateParticles(NvFlowContext *context,
                                      const NvFlowParticleSurfaceData *data) {
    auto mapped = (NvFlowFloat4 *)NvFlowBufferMap(context, m_uploadBuffer);
    if (mapped) {
        uint32_t stride = data->positionStride / sizeof(float);
        for (uint32_t idx = 0; idx < data->numParticles; ++idx) {
            mapped[idx] = make_float4(data->positions[stride * idx],
                                      data->positions[stride * idx + 1],
                                      data->positions[stride * idx + 2], 1.f);
        }
        m_particleCount = data->numParticles;
        NvFlowBufferUnmap(context, m_uploadBuffer);
    }
}

void ParticleSurface::updateSurface(NvFlowContext *context,
                                    const NvFlowParticleSurfaceParams *params) {
    NvFlowContextProfileGroupBegin(context, L"UpdateSurface");

    auto atomicData = (NvFlowUint *)NvFlowBufferMap(context, m_atomicBuf);
    if (atomicData) {
        NvFlowBufferDesc desc;
        NvFlowBufferGetDesc(m_atomicBuf, &desc);
        memset(atomicData, 0, sizeof(desc.dim) * sizeof(NvFlowUint));
        NvFlowBufferUnmap(context, m_atomicBuf);
    }

    // Clear block table
    {
        NvFlowTexture3DDesc blockTableDesc;
        NvFlowTexture3DGetDesc(m_blockTable, &blockTableDesc);

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceBlockTableClearCS;
        dparams.gridDim = (blockTableDesc.dim + 7) / 8;
        dparams.rootConstantBuffer = 0;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_blockTable);
        NvFlowContextDispatch(context, &dparams);
    }

    // Clear block list
    {
        NvFlowBufferDesc blockListDesc;
        NvFlowBufferGetDesc(m_blockList, &blockListDesc);

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceBlockListClearCS;
        dparams.gridDim.x = (blockListDesc.dim + 127) / 128;
        dparams.gridDim.y = 1;
        dparams.gridDim.z = 1;
        dparams.rootConstantBuffer = 0;
        dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_blockList);
        NvFlowContextDispatch(context, &dparams);
    }

    // Block list allocate
    {
        NvFlowTexture3DDesc blockTableDesc;
        NvFlowTexture3DGetDesc(m_blockTable, &blockTableDesc);

        auto mapped = (ParticleSurfaceAllocParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->positionToMask = m_particleToFieldT;
            mapped->maskDim = make_uint4(blockTableDesc.dim, 0);
            mapped->particleCount = make_uint4(m_particleCount, 0, 0, 0);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceAllocCS;
        dparams.gridDim.x = (m_particleCount + 127) / 128;
        dparams.gridDim.y = 1;
        dparams.gridDim.z = 1;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = NvFlowBufferGetResource(m_uploadBuffer);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_blockTable);
        NvFlowContextDispatch(context, &dparams);
    }

    // Block table allocate
    {
        NvFlowTexture3DDesc blockTableDesc;
        NvFlowTexture3DGetDesc(m_blockTable, &blockTableDesc);

        auto mapped = (ParticleSurfaceBlockTableAllocParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->gridDim = make_uint4(m_blockConfig.gridDim, 0);
            mapped->poolGridDim = make_uint4(m_blockConfig.poolGridDim, 0);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceBlockTableAllocCS;
        dparams.gridDim = (blockTableDesc.dim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_blockTable);
        dparams.readWrite[1] = NvFlowBufferGetResourceRW(m_blockList);
        dparams.readWrite[2] = NvFlowBufferGetResourceRW(m_atomicBuf);
        NvFlowContextDispatch(context, &dparams);

        NvFlowContextCopyConstantBuffer(context, m_atomicConst, m_atomicBuf);
    }

    // Particle surface clear
    {
        NvFlowTexture3DDesc backTexDesc;
        NvFlowTexture3DGetDesc(m_fieldBack, &backTexDesc);
        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceClearCS;
        dparams.gridDim = (backTexDesc.dim + 7) >> 3;
        dparams.rootConstantBuffer = 0;
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
        NvFlowContextDispatch(context, &dparams);
    }

    // Splat
    {
        NvFlowDim gridDim;
        gridDim.x = (m_particleCount + 127) / 128;
        gridDim.y = 1;
        gridDim.z = 1;
        auto mapped = (ParticleSurfaceSplatParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->positionToField = m_particleToFieldT;
            mapped->linearParams = m_linearParams;
            mapped->fieldDim = make_uint4(m_desc.virtualDim, 0);
            mapped->particleCount = make_uint4(m_particleCount, 0, 0, 0);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceSplatCS;
        dparams.gridDim = gridDim;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = NvFlowBufferGetResource(m_uploadBuffer);
        dparams.readOnly[1] = NvFlowTexture3DGetResource(m_blockTable);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
        NvFlowContextDispatch(context, &dparams);
    }

    swap(m_fieldFront, m_fieldBack);

    if (!m_atomicState) {
        NvFlowBufferDownload(context, m_atomicBuf);
        m_atomicState = 1;
    }

    if (m_atomicState == 1) {
        auto atomicData = (NvFlowUint *)NvFlowBufferMapDownload(context, m_atomicBuf);
        if (atomicData) {
            m_numBlocks = atomicData[0];
            NvFlowBufferUnmapDownload(context, m_atomicBuf);
            m_atomicState = 0;
        }
    }

    m_dispatchBlocks = (3 * m_numBlocks + 1) / 4;

    float smoothRadius = params->smoothRadius;
    if (smoothRadius < 0.f)
        smoothRadius = 0.f;
    if (smoothRadius > 16.f)
        smoothRadius = 16.f;

    m_surfaceThreshold = params->surfaceThreshold;

    if (params->separableSmoothing) {
        float h = smoothRadius / 8.f;

        constexpr int KERNEL_HALF_DIM = 8;
        constexpr int KERNEL_DIM = 2 * KERNEL_HALF_DIM + 1;

        float kernelWeights[(KERNEL_DIM + 3) & ~3];
        float kernelWeightSum = 0.f;
        for (int i = 0; i < KERNEL_DIM; ++i) {
            float w = exp(-(i - 8) * (i - 8) / (2.f * h * h));
            kernelWeightSum += w;
            kernelWeights[i] = w;
        }

        float kernelWeightSumInv = 1.f / kernelWeightSum;
        for (int i = 0; i < KERNEL_DIM; ++i)
            kernelWeights[i] *= kernelWeightSumInv;

        auto mapped = (ParticleSurfaceSmoothParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->linearParams = m_linearParams;
            mapped->dispatchNumBlocks = m_dispatchBlocks;
            mapped->padd1 = 0;
            mapped->padd2 = 0;
            mapped->padd3 = 0;
            CopyMemory(mapped->kernel, kernelWeights, sizeof(kernelWeights));
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDim smoothXGridDim;
        smoothXGridDim.x = (m_blockConfig.blockDim.x * m_dispatchBlocks + 7) / 8;
        smoothXGridDim.y = (m_blockConfig.blockDim.y + 3) / 4;
        smoothXGridDim.z = (m_blockConfig.blockDim.z + 3) / 4;
        NvFlowDim smoothYDim;
        smoothYDim.x = (m_blockConfig.blockDim.x * m_dispatchBlocks + 3) / 4;
        smoothYDim.y = (m_blockConfig.blockDim.y + 7) / 8;
        smoothYDim.z = (m_blockConfig.blockDim.z + 3) / 4;
        NvFlowDim smoothZDim;
        smoothZDim.x = (m_blockConfig.blockDim.x * m_dispatchBlocks + 3) / 4;
        smoothZDim.y = (m_blockConfig.blockDim.y + 4) / 4;
        smoothZDim.z = (m_blockConfig.blockDim.z + 7) / 8;

        NvFlowDispatchParams dparams = {};
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.secondConstantBuffer = m_atomicConst;
        dparams.readOnly[1] = NvFlowTexture3DGetResource(m_blockTable);
        dparams.readOnly[2] = NvFlowBufferGetResource(m_blockList);

        dparams.gridDim = smoothXGridDim;
        dparams.shader = m_particleSurfaceSmoothXCS;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_fieldFront);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
        NvFlowContextDispatch(context, &dparams);

        swap(m_fieldFront, m_fieldBack);

        dparams.gridDim = smoothYDim;
        dparams.shader = m_particleSurfaceSmoothYCS;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_fieldFront);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
        NvFlowContextDispatch(context, &dparams);

        swap(m_fieldFront, m_fieldBack);

        dparams.gridDim = smoothZDim;
        dparams.shader = m_particleSurfaceSmoothZCS;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_fieldFront);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
        NvFlowContextDispatch(context, &dparams);

        swap(m_fieldFront, m_fieldBack);
    } else {
        auto mapped = (ParticleSurfaceSmoothParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->linearParams = m_linearParams;
            mapped->dispatchNumBlocks = m_dispatchBlocks;
            mapped->padd1 = asuint(params->surfaceThreshold);
            mapped->padd2 = 0.f;
            mapped->padd3 = 0.f;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        int kernelRadiusi = int(smoothRadius);
        NvFlowDim smoothGridDim;
        smoothGridDim.x = (m_blockConfig.blockDim.x * m_dispatchBlocks + 7) / 8;
        smoothGridDim.y = (m_blockConfig.blockDim.y + 7) / 8;
        smoothGridDim.z = (m_blockConfig.blockDim.z + 7) / 8;
        for (int n = 0; n < kernelRadiusi; ++n) {
            NvFlowDispatchParams dparams = {};
            dparams.shader = m_particleSurfaceSmoothCS;
            dparams.gridDim = smoothGridDim;
            dparams.rootConstantBuffer = m_constantBuffer;
            dparams.secondConstantBuffer = m_atomicConst;
            dparams.readOnly[0] = NvFlowTexture3DGetResource(m_fieldFront);
            dparams.readOnly[1] = NvFlowTexture3DGetResource(m_blockTable);
            dparams.readOnly[2] = NvFlowBufferGetResource(m_blockList);
            dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
            NvFlowContextDispatch(context, &dparams);
            swap(m_fieldFront, m_fieldBack);
        }
    }

    // Update linear
    {
        NvFlowDim gridDim;
        gridDim.x = (m_linearParams.linearBlockDim.w + 127) / 128;
        gridDim.y = m_dispatchBlocks;
        gridDim.z = 1;

        auto mapped = (ParticleSurfaceUpdateLinearParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->params = m_linearParams;
            mapped->dispatchNumBlocks = m_dispatchBlocks;
            mapped->surfaceThreshold = params->surfaceThreshold;
            mapped->padd2 = 0;
            mapped->padd3 = 0;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_particleSurfaceUpdateLinearCS;
        dparams.gridDim = gridDim;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.secondConstantBuffer = m_atomicConst;
        dparams.readOnly[0] = NvFlowBufferGetResource(m_blockList);
        dparams.readOnly[1] = NvFlowTexture3DGetResource(m_fieldFront);
        dparams.readOnly[2] = NvFlowTexture3DGetResource(m_blockTable);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_fieldBack);
        NvFlowContextDispatch(context, &dparams);

        swap(m_fieldFront, m_fieldBack);
    }

    NvFlowContextProfileGroupEnd(context);
}

void ParticleSurface::allocFunc(NvFlowContext *context,
                                const NvFlowGridEmitCustomAllocParams *params) {
    NvFlowDim gridDim;
    gridDim.x = (m_particleCount + 127) / 128;
    gridDim.y = 1;
    gridDim.z = 1;
    auto T = matrixTranslation(params->gridLocation.x, params->gridLocation.y,
                               params->gridLocation.z);
    auto S = matrixScaling(params->gridHalfSize.x, params->gridHalfSize.y,
                           params->gridHalfSize.z);
    auto modelMatrix = S * T;
    auto particleToMaskT = transpose(inverse(modelMatrix));

    auto mapped =
        (ParticleSurfaceAllocParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
    if (mapped) {
        mapped->positionToMask = particleToMaskT;
        mapped->maskDim = make_uint4(params->maskDim, 0);
        mapped->particleCount = make_uint4(m_particleCount, 0, 0, 0);
        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    NvFlowDispatchParams dispatchParams = {};
    dispatchParams.shader = m_particleSurfaceEmitAllocCS;
    dispatchParams.gridDim = gridDim;
    dispatchParams.rootConstantBuffer = m_constantBuffer;
    dispatchParams.readOnly[0] = NvFlowBufferGetResource(m_uploadBuffer);
    dispatchParams.readWrite[0] = params->maskResourceRW;
    NvFlowContextDispatch(context, &dispatchParams);
}

void ParticleSurface::emitVelocityFunc(NvFlowContext *context, uint32_t *dataFrontIdx,
                                       const NvFlowGridEmitCustomEmitParams *params,
                                       const NvFlowParticleSurfaceEmitParams *emitParams) {
    EmitParams emitShaderParams = {};
    emitShaderParams.deltaTime = make_float4(emitParams->deltaTime);
    emitShaderParams.coupleRate = make_float4(emitParams->velocityCoupleRate, 0.f);
    emitShaderParams.emitValue = make_float4(emitParams->velocityLinear, 0.f);
    emitFunc(context, dataFrontIdx, params, &emitShaderParams);
}

void ParticleSurface::emitDensityFunc(NvFlowContext *context, uint32_t *dataFrontIdx,
                                      const NvFlowGridEmitCustomEmitParams *params,
                                      const NvFlowParticleSurfaceEmitParams *emitParams) {
    EmitParams emitShaderParams = {};
    emitShaderParams.deltaTime = make_float4(emitParams->deltaTime);
    emitShaderParams.coupleRate =
        make_float4(emitParams->temperatureCoupleRate, emitParams->fuelCoupleRate, 0.f,
                    emitParams->smokeCoupleRate);
    emitShaderParams.emitValue =
        make_float4(emitParams->temperature, emitParams->fuel, 0.f, emitParams->smoke);

    emitFunc(context, dataFrontIdx, params, &emitShaderParams);
}

NvFlowGridExport *ParticleSurface::debugGridExport(NvFlowContext *context) {
    NvFlowDim gridDim;
    gridDim.x = (m_linearParams.linearBlockDim.w + 127) / 128;
    gridDim.y = m_dispatchBlocks;
    gridDim.z = 1;
    auto mapped = (ParticleSurfaceUpdateLinearParams *)NvFlowConstantBufferMap(
        context, m_constantBuffer);
    if (mapped) {
        mapped->params = m_linearParams;
        mapped->dispatchNumBlocks = m_dispatchBlocks;
        mapped->surfaceThreshold = m_surfaceThreshold;
        mapped->padd2 = 0;
        mapped->padd3 = 0;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);
    }

    NvFlowDispatchParams dispatchParams = {};
    dispatchParams.shader = m_particleSurfaceDebugVisCS;
    dispatchParams.gridDim = gridDim;
    dispatchParams.rootConstantBuffer = m_constantBuffer;
    dispatchParams.secondConstantBuffer = m_atomicConst;
    dispatchParams.readOnly[0] = NvFlowTexture3DGetResource(m_fieldFront);
    dispatchParams.readOnly[1] = NvFlowTexture3DGetResource(m_blockTable);
    dispatchParams.readOnly[2] = NvFlowBufferGetResource(m_blockList);
    dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_debugVisTex);
    NvFlowContextDispatch(context, &dispatchParams);
    return &m_export;
}

void ParticleSurface::emitFunc(NvFlowContext *context, unsigned int *dataFrontIdx,
                               const NvFlowGridEmitCustomEmitParams *params,
                               const EmitParams *emitParams) {
    for (uint32_t layerIdx = 0; layerIdx < params->numLayers; ++layerIdx) {
        NvFlowGridEmitCustomEmitLayerParams layerParams = {};
        NvFlowGridEmitCustomGetLayerParams(params, 0, &layerParams);
        auto T = matrixTranslation(layerParams.gridLocation.x, layerParams.gridLocation.y,
                                   layerParams.gridLocation.z);
        auto S = matrixScaling(layerParams.gridHalfSize.x, layerParams.gridHalfSize.y,
                               layerParams.gridHalfSize.z);
        auto gridToWorld = S * T;
        T = matrixTranslation(m_desc.initialLocation.x, m_desc.initialLocation.y,
                              m_desc.initialLocation.z);
        S = matrixScaling(m_desc.halfSize.x, m_desc.halfSize.y, m_desc.halfSize.z);
        auto fieldToWorld = S * T;
        auto gridToField = gridToWorld * inverse(fieldToWorld);
        auto gridToFieldT = transpose(gridToField);
        NvFlowDim gridDim;
        gridDim.x = (layerParams.shaderParams.blockDim.x * layerParams.numBlocks + 7) >> 3;
        gridDim.y = (layerParams.shaderParams.blockDim.y + 7) >> 3;
        gridDim.z = (layerParams.shaderParams.blockDim.z + 7) >> 3;
        NvFlowFloat4 vdimInv;
        (NvFlowFloat3 &)vdimInv =
            1.f / make_float3((const NvFlowUint3 &)layerParams.shaderParams.blockDim *
                              (const NvFlowUint3 &)layerParams.shaderParams.gridDim);
        vdimInv.w = 0.f;

        auto mapped = (ParticleSurfaceEmitEmitParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->customEmitParams = layerParams.shaderParams;
            mapped->surfaceParams = m_linearParams;
            mapped->gridToField = gridToFieldT;
            mapped->vdimInv = vdimInv;
            mapped->deltaTime = emitParams->deltaTime;
            mapped->coupleRate = emitParams->coupleRate;
            mapped->emitValue = emitParams->emitValue;
            mapped->surfaceThreshold = m_surfaceThreshold;
            mapped->padd1 = 0;
            mapped->padd2 = 0;
            mapped->padd3 = 0;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        uint32_t localDataFrontIdx = *dataFrontIdx;
        NvFlowDispatchParams dispatchParams = {};
        dispatchParams.shader = m_particleSurfaceEmitEmitCS;
        dispatchParams.gridDim = gridDim;
        dispatchParams.rootConstantBuffer = m_constantBuffer;
        dispatchParams.readOnly[0] = layerParams.blockList;
        dispatchParams.readOnly[1] = layerParams.blockTable;
        dispatchParams.readOnly[2] =
            NvFlowResourceRWGetResource(layerParams.dataRW[localDataFrontIdx]);
        dispatchParams.readOnly[3] = NvFlowTexture3DGetResource(m_fieldFront);
        dispatchParams.readOnly[4] = NvFlowTexture3DGetResource(m_blockTable);
        dispatchParams.readWrite[0] = layerParams.dataRW[localDataFrontIdx ^ 1];
        NvFlowContextDispatch(context, &dispatchParams);
    }
    *dataFrontIdx ^= 1;
}

void ParticleSurface::computeSparseTextureConfig(const NvFlowDim &virtualDim,
                                                 float residentScale,
                                                 const NvFlowDim &blockDim) {
    auto &blockConfig = m_blockConfig;
    blockConfig.virtualDim = virtualDim;
    blockConfig.blockDim = blockDim;
    blockConfig.linearBlockDim = blockDim + 2;
    blockConfig.linearBlockOffset = make_dim(1);
    blockConfig.gridDim = blockConfig.virtualDim / blockConfig.blockDim;
    blockConfig.maxVirtualBlocks =
        blockConfig.gridDim.z * blockConfig.gridDim.y * blockConfig.gridDim.x;
    blockConfig.maxBlocks = int(residentScale * blockConfig.maxVirtualBlocks);

    float rgridDimf = pow(float(blockConfig.maxBlocks), 1.f / 3.f);
    blockConfig.poolGridDim.z = ceil(rgridDimf);
    rgridDimf = sqrt(float(blockConfig.maxBlocks) / blockConfig.poolGridDim.z);
    blockConfig.poolGridDim.y = ceil(rgridDimf);

    uint32_t yzslice = blockConfig.poolGridDim.y * blockConfig.poolGridDim.z;
    blockConfig.poolGridDim.x = (blockConfig.maxBlocks + yzslice - 1) / yzslice;

    blockConfig.poolDim = blockConfig.linearBlockDim * blockConfig.poolGridDim;

    blockConfig.maxBlocks =
        blockConfig.poolGridDim.z * blockConfig.poolGridDim.y * blockConfig.poolGridDim.x;
}

void ParticleSurface::generateLinearParams() {
    auto make_uint4 = [](const NvFlowDim &v) {
        return NvFlowUint4{v.x, v.y, v.z, v.z * v.y * v.x};
    };
    auto make_float4 = [](const NvFlowDim &v) {
        return NvFlowFloat4{float(v.x), float(v.y), float(v.z), float(v.z * v.y * v.x)};
    };
    auto make_float4_inv = [](const NvFlowDim &v) {
        return 1.f /
               NvFlowFloat4{float(v.x), float(v.y), float(v.z), float(v.z * v.y * v.x)};
    };

    auto &config = m_blockConfig;
    NvFlowUint4 blockDim = make_uint4(config.blockDim);
    NvFlowUint4 blockDimBits;
    blockDimBits.x = log2ui(blockDim.x);
    blockDimBits.y = log2ui(blockDim.y);
    blockDimBits.z = log2ui(blockDim.z);
    blockDimBits.w = log2ui(blockDim.w);

    NvFlowUint4 poolGridDim = NvFlow::make_uint4(config.poolGridDim, 1);
    NvFlowUint4 gridDim = NvFlow::make_uint4(config.gridDim, 1);

    NvFlowFloat4 blockDimInv = make_float4_inv(config.blockDim);
    NvFlowUint4 linearBlockDim = make_uint4(config.linearBlockDim);
    NvFlowUint4 linearBlockOffset = NvFlow::make_uint4(config.linearBlockOffset, 0);

    NvFlowFloat4 dimInv = make_float4_inv(config.poolDim);
    NvFlowFloat4 vdim = make_float4(config.virtualDim);
    NvFlowFloat4 vdimInv = make_float4_inv(config.virtualDim);

    m_linearParams.isVTR = NvFlow::make_uint4(0);
    m_linearParams.blockDim = blockDim;
    m_linearParams.blockDimBits = blockDimBits;
    m_linearParams.poolGridDim = poolGridDim;
    m_linearParams.gridDim = gridDim;
    m_linearParams.blockDimInv = blockDimInv;
    m_linearParams.linearBlockDim = linearBlockDim;
    m_linearParams.linearBlockOffset = linearBlockOffset;
    m_linearParams.dimInv = dimInv;
    m_linearParams.vdim = vdim;
    m_linearParams.vdimInv = vdimInv;
}

static void particleSurfaceMax(uint32_t *cbsize, uint32_t size) {
    if (size > *cbsize)
        *cbsize = size;
}

static uint32_t particleSurfaceConstantSize() {
    uint32_t cbsize = 0;
    particleSurfaceMax(&cbsize, 0x60u);
    particleSurfaceMax(&cbsize, 0x20u);
    particleSurfaceMax(&cbsize, 0x110u);
    particleSurfaceMax(&cbsize, 0x110u);
    particleSurfaceMax(&cbsize, 0xC0u);
    particleSurfaceMax(&cbsize, 0x190u);
    return cbsize;
}

#include "particleSurfaceBlockListClearCS.hlsl.h"
#include "particleSurfaceBlockTableClearCS.hlsl.h"
#include "particleSurfaceAllocCS.hlsl.h"
#include "particleSurfaceBlockTableAllocCS.hlsl.h"
#include "particleSurfaceClearCS.hlsl.h"
#include "particleSurfaceSplatCS.hlsl.h"
#include "particleSurfaceSmoothCS.hlsl.h"
#include "particleSurfaceSmoothXCS.hlsl.h"
#include "particleSurfaceSmoothYCS.hlsl.h"
#include "particleSurfaceSmoothZCS.hlsl.h"
#include "particleSurfaceUpdateLinearCS.hlsl.h"
#include "particleSurfaceDebugVisCS.hlsl.h"
#include "particleSurfaceEmitAllocCS.hlsl.h"
#include "particleSurfaceEmitEmitCS.hlsl.h"

ParticleSurface::ParticleSurface(NvFlowContext *context,
                                 const NvFlowParticleSurfaceDesc *desc)
    : m_desc{},
      m_modelMatrix{},
      m_blockConfig{},
      m_linearParams{},
      m_numBlocks{0},
      m_atomicState{0},
      m_dispatchBlocks{0},
      m_surfaceThreshold{0},
      m_export{this},
      m_uploadBuffer{0},
      m_particleCount{0},
      m_fieldFront{0},
      m_fieldBack{0},
      m_debugVisTex{0},
      m_blockTable{0},
      m_blockList{0},
      m_atomicBuf{0},
      m_atomicConst{0},
      m_constantBuffer{0},
      m_particleSurfaceBlockListClearCS{0},
      m_particleSurfaceBlockTableClearCS{0},
      m_particleSurfaceAllocCS{0},
      m_particleSurfaceBlockTableAllocCS{0},
      m_particleSurfaceClearCS{0},
      m_particleSurfaceSplatCS{0},
      m_particleSurfaceSmoothCS{0},
      m_particleSurfaceSmoothXCS{0},
      m_particleSurfaceSmoothYCS{0},
      m_particleSurfaceSmoothZCS{0},
      m_particleSurfaceUpdateLinearCS{0},
      m_particleSurfaceDebugVisCS{0},
      m_particleSurfaceEmitAllocCS{0},
      m_particleSurfaceEmitEmitCS{0} {
    m_desc = *desc;
    auto T = matrixTranslation(m_desc.initialLocation.x, m_desc.initialLocation.y,
                               m_desc.initialLocation.z);
    auto S = matrixScaling(m_desc.halfSize.x, m_desc.halfSize.y, m_desc.halfSize.z);
    m_modelMatrix = S * T;
    m_particleToFieldT = transpose(inverse(m_modelMatrix));

    const NvFlowDim blockDim = make_dim(16);
    computeSparseTextureConfig(m_desc.virtualDim, m_desc.residentScale, blockDim);
    generateLinearParams();

    NvFlowBufferDesc bufDesc = {};
    bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
    bufDesc.dim = m_desc.maxParticles;
    bufDesc.uploadAccess = 1;
    bufDesc.downloadAccess = 1;
    m_uploadBuffer = NvFlowCreateBuffer(context, &bufDesc);

    NvFlowTexture3DDesc texDesc = {};
    texDesc.format = eNvFlowFormat_r16g16_float;
    texDesc.dim = m_blockConfig.poolDim;
    texDesc.uploadAccess = 0;
    texDesc.downloadAccess = 0;
    m_fieldFront = NvFlowCreateTexture3D(context, &texDesc);
    m_fieldBack = NvFlowCreateTexture3D(context, &texDesc);

    texDesc.dim = m_blockConfig.poolDim;
    texDesc.format = eNvFlowFormat_r16g16b16a16_float;
    m_debugVisTex = NvFlowCreateTexture3D(context, &texDesc);

    NvFlowTexture3DDesc blockTableDesc = {};
    blockTableDesc.format = eNvFlowFormat_r32_uint;
    blockTableDesc.dim = m_blockConfig.gridDim;
    blockTableDesc.uploadAccess = 0;
    blockTableDesc.downloadAccess = 0;
    m_blockTable = NvFlowCreateTexture3D(context, &blockTableDesc);

    NvFlowBufferDesc bufferDesc = {};
    bufferDesc.format = eNvFlowFormat_r32_uint;
    bufferDesc.dim = m_blockConfig.maxBlocks;
    bufferDesc.uploadAccess = 0;
    bufferDesc.downloadAccess = 0;
    m_blockList = NvFlowCreateBuffer(context, &bufferDesc);

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

    uint32_t cbMaxSize = particleSurfaceConstantSize();
    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = cbMaxSize;
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_particleSurfaceBlockListClearCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceBlockListClearCS));
    m_particleSurfaceBlockTableClearCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceBlockTableClearCS));
    m_particleSurfaceAllocCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceAllocCS));
    m_particleSurfaceBlockTableAllocCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceBlockTableAllocCS));
    m_particleSurfaceClearCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceClearCS));
    m_particleSurfaceSplatCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceSplatCS));
    m_particleSurfaceSmoothCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceSmoothCS));
    m_particleSurfaceSmoothXCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceSmoothXCS));
    m_particleSurfaceSmoothYCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceSmoothYCS));
    m_particleSurfaceSmoothZCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceSmoothZCS));
    m_particleSurfaceUpdateLinearCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceUpdateLinearCS));
    m_particleSurfaceDebugVisCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceDebugVisCS));
    m_particleSurfaceEmitAllocCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceEmitAllocCS));
    m_particleSurfaceEmitEmitCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(particleSurfaceEmitEmitCS));
}

ParticleSurface::~ParticleSurface() {
    SafeRelease(m_uploadBuffer);
    SafeRelease(m_fieldFront);
    SafeRelease(m_fieldBack);
    SafeRelease(m_debugVisTex);
    SafeRelease(m_blockTable);
    SafeRelease(m_blockList);
    SafeRelease(m_atomicBuf);
    SafeRelease(m_atomicConst);
    SafeRelease(m_constantBuffer);

    SafeRelease(m_particleSurfaceBlockListClearCS);
    SafeRelease(m_particleSurfaceBlockTableClearCS);
    SafeRelease(m_particleSurfaceAllocCS);
    SafeRelease(m_particleSurfaceBlockTableAllocCS);
    SafeRelease(m_particleSurfaceClearCS);
    SafeRelease(m_particleSurfaceSplatCS);
    SafeRelease(m_particleSurfaceSmoothCS);
    SafeRelease(m_particleSurfaceSmoothXCS);
    SafeRelease(m_particleSurfaceSmoothYCS);
    SafeRelease(m_particleSurfaceSmoothZCS);
    SafeRelease(m_particleSurfaceUpdateLinearCS);
    SafeRelease(m_particleSurfaceDebugVisCS);
    SafeRelease(m_particleSurfaceEmitAllocCS);
    SafeRelease(m_particleSurfaceEmitEmitCS);
}

NvFlowParticleSurface *FlowCreateParticleSurface(NvFlowContext *context,
                                                 const NvFlowParticleSurfaceDesc *desc) {
    return new ParticleSurface(context, desc);
}

}  // namespace NvFlow