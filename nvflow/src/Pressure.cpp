#include "Pressure.h"
#include "Object.h"
#include "SparseTexturePool.h"
#include "ClientHelper.h"

namespace NvFlow {

union PressureShaderParams {
    struct Divergence {
        NvFlowShaderPointParams pressureParams;

        NvFlowShaderPointParams velocityParams;
    } _1;

    struct Jacobi {
        NvFlowShaderPointParams levelParams;
        NvFlowShaderPointParams pressureParams;
        NvFlowFloat4 dx2;
    } _2;

    struct Restrict {
        NvFlowShaderPointParams outLevelParams;
        NvFlowShaderPointParams inLevelParams;
        NvFlowShaderPointParams pressureParams;
        NvFlowFloat4 dx2Inv;
    } _3;

    struct Prolong {
        NvFlowShaderPointParams outLevelParams;
        NvFlowShaderPointParams inLevelParams;
        NvFlowShaderPointParams pressureParams;
    } _4;

    struct Subtract {
        NvFlowShaderPointParams velocityParams;

        NvFlowShaderPointParams pressureParams;

        NvFlowFloat4 vdimInv;
    } _5;
};

struct PressureImpl : Object, Pressure {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void execute(NvFlowContext *context, SparseTextureFront *velocity,
                 SparseTextureFront *pressureIn, SparseFadeField *fadeField,
                 const PressureParams *params) override;

    // Details
    void shiftToLevel(NvFlowShaderPointParams *params, uint32_t level);

    void pressure(NvFlowContext *context, SparseTextureFront *pressureIn, uint32_t level,
                  const PressureParams *params);

    void smooth(NvFlowContext *context, SparseTextureFront *pressureIn, uint32_t level);

    PressureImpl(NvFlowContext *context, const PressureDesc *desc);
    ~PressureImpl();

    PressureDesc m_desc;
    NvFlowComputeShader *m_divergenceCS;
    NvFlowComputeShader *m_jacobiCS;
    NvFlowComputeShader *m_restrictCS;
    NvFlowComputeShader *m_prolongCS;
    NvFlowComputeShader *m_subtractCS;
    NvFlowConstantBuffer *m_constantBuffer;
    uint32_t fineIterations;
    uint32_t coarseIterations;
    uint32_t maxLevels;
};

uint64_t PressureImpl::getGPUBytesUsed() {
    return 0;
}

void PressureImpl::execute(NvFlowContext *context, SparseTextureFront *velocity,
                           SparseTextureFront *pressureIn, SparseFadeField *fadeField,
                           const PressureParams *params) {
    auto result = pressureIn->acquireTexture(context);
    auto pressureOutHandle = result.writePointHandle(context);
    auto velocityInHandle = velocity->front.readPointHandle(context);
    auto pressureOutLayeredView = pressureOutHandle.layeredView();
    auto velocityInLayeredView = velocityInHandle.layeredView();

    for (uint32_t layerIdx = 0; layerIdx < pressureOutHandle.numLayers; ++layerIdx) {
        auto pressureOutLayerView = pressureOutHandle.layerView(layerIdx);
        auto velocityInLayerView = velocityInHandle.layerView(layerIdx);

        // Divergence
        auto mapped = (PressureShaderParams::Divergence *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        mapped->pressureParams = pressureOutLayeredView.params;
        mapped->velocityParams = velocityInLayeredView.pointParams;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDim gridDim;
        gridDim.x = (pressureOutLayeredView.params.blockDim.x *
                         pressureOutLayerView.mapping.numBlocks +
                     7) /
                    8;
        gridDim.y = (pressureOutLayeredView.params.blockDim.y + 7) / 8;
        gridDim.z = (pressureOutLayeredView.params.blockDim.z + 7) / 8;

        NvFlowDispatchParams paramsa = {};
        paramsa.shader = m_divergenceCS;
        paramsa.gridDim = gridDim;
        paramsa.rootConstantBuffer = m_constantBuffer;
        paramsa.readOnly[0] = pressureOutLayerView.mapping.blockList;
        paramsa.readOnly[1] = pressureOutLayerView.mapping.blockTable;
        paramsa.readOnly[2] = velocityInLayerView.data;
        paramsa.readOnly[3] = velocityInLayerView.mapping.blockTable;
        paramsa.readWrite[0] = pressureOutLayerView.data;
        NvFlowContextDispatch(context, &paramsa);
    }
    pressureIn->swap(result);

    pressure(context, pressureIn, 0, params);

    auto velocityResult = velocity->acquireTexture(context);
    auto velocityOutHandle = velocityResult.writePointHandle(context);
    velocityInHandle = velocity->front.readPointHandle(context);
    auto pressureInHandle = pressureIn->front.readPointHandle(context);
    auto velocityOutLayeredView = velocityOutHandle.layeredView();
    velocityInLayeredView = velocityInHandle.layeredView();
    auto pressureInLayeredView = pressureInHandle.layeredView();

    for (uint32_t j = 0; j < pressureInHandle.numLayers; ++j) {
        auto velocityOutLayerView = velocityOutHandle.layerView(j);
        auto velocityInLayerView = velocityInHandle.layerView(j);
        auto pressureInLayerView = pressureInHandle.layerView(j);

        NvFlowDim vdim = velocity->getDesc().virtualDim;
        NvFlowFloat4 vDimInv = 1.f / make_float4(vdim, 1.f);

        auto mapped = (PressureShaderParams::Subtract *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        mapped->velocityParams = velocityOutLayeredView.params;
        mapped->pressureParams = pressureInLayeredView.pointParams;
        mapped->vdimInv = vDimInv;
        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDim gridDim;
        gridDim.x = (velocityOutLayeredView.params.blockDim.x *
                         velocityOutLayerView.mapping.numBlocks +
                     7) /
                    8;
        gridDim.y = (velocityOutLayeredView.params.blockDim.y + 7) / 8;
        gridDim.z = (velocityOutLayeredView.params.blockDim.z + 7) / 8;

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_subtractCS;
        dparams.gridDim = gridDim;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = velocityOutLayerView.mapping.blockList;
        dparams.readOnly[1] = velocityOutLayerView.mapping.blockTable;
        dparams.readOnly[2] = velocityInLayerView.data;
        dparams.readOnly[3] = pressureInLayerView.data;
        dparams.readOnly[4] = pressureInLayerView.mapping.blockTable;
        NvFlowResource *fadeResource = 0;
        if (fadeField) {
            fadeResource = fadeField->getFadeField(fadeField->userData, 1, j);
        }

        dparams.readOnly[5] = fadeResource;
        dparams.readWrite[0] = velocityOutLayerView.data;
        NvFlowContextDispatch(context, &dparams);
    }

    velocity->swap(velocityResult);
}

void PressureImpl::shiftToLevel(NvFlowShaderPointParams *params, uint32_t level) {
    params->blockDim.x >>= level;
    params->blockDim.y >>= level;
    params->blockDim.z >>= level;
    params->blockDim.w >>= 3 * level;
    params->blockDimBits.x -= level;
    params->blockDimBits.y -= level;
    params->blockDimBits.z -= level;
    params->blockDimBits.w -= 3 * level;
}

void PressureImpl::pressure(NvFlowContext *context, SparseTextureFront *pressureIn,
                            uint32_t level, const PressureParams *params) {
    smooth(context, pressureIn, level);

    if (level != maxLevels - 1) {
        auto prevPressure = pressureIn->front;
        prevPressure.addRefTexture();

        {
            float dx2Inv = 1.f / float(1 << (2 * level));

            auto result = pressureIn->acquireTexture(context);
            auto pressureOutHandle = result.writePointHandle(context);
            auto pressureInHandle = pressureIn->front.readPointHandle(context);
            auto pressureOutLayeredView = pressureOutHandle.layeredView();
            auto pressureInLayeredView = pressureInHandle.layeredView();

            for (uint32_t layerIdx = 0; layerIdx < pressureOutHandle.numLayers;
                 ++layerIdx) {
                auto pressureOutLayerView = pressureOutHandle.layerView(layerIdx);
                auto pressureInLayerView = pressureInHandle.layerView(layerIdx);

                NvFlowShaderPointParams curLevelParams = pressureOutLayeredView.params;
                shiftToLevel(&curLevelParams, level);

                NvFlowShaderPointParams nextLevelParams =
                    params->legacyMode ? curLevelParams : pressureOutLayeredView.params;
                shiftToLevel(&nextLevelParams, level + 1);

                auto mapped = (PressureShaderParams::Restrict *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                mapped->outLevelParams = nextLevelParams;
                mapped->inLevelParams = curLevelParams;
                mapped->pressureParams = pressureOutLayeredView.params;
                mapped->dx2Inv = make_float4(dx2Inv);
                NvFlowConstantBufferUnmap(context, m_constantBuffer);

                NvFlowDim gridDim;
                gridDim.x =
                    (nextLevelParams.blockDim.w * pressureOutLayerView.mapping.numBlocks +
                     255) /
                    0x100;
                gridDim.y = 1;
                gridDim.z = 1;

                NvFlowDispatchParams dparams = {};
                dparams.shader = m_restrictCS;
                dparams.gridDim = gridDim;
                dparams.rootConstantBuffer = m_constantBuffer;
                dparams.readOnly[0] = pressureOutLayerView.mapping.blockList;
                dparams.readOnly[1] = pressureOutLayerView.mapping.blockTable;
                dparams.readOnly[2] = pressureInLayerView.data;
                dparams.readWrite[0] = pressureOutLayerView.data;
                NvFlowContextDispatch(context, &dparams);
            }

            pressureIn->swap(result);
        }

        pressure(context, pressureIn, level + 1, params);

        {
            auto result = pressureIn->acquireTexture(context);
            auto pressureOutHandle = result.writePointHandle(context);
            auto prevPressureInHandle = prevPressure.readPointHandle(context);
            auto pressureInHandle = pressureIn->front.readPointHandle(context);

            auto pressureOutLayeredView = pressureOutHandle.layeredView();
            auto prevPressureInLayeredView = prevPressureInHandle.layeredView();
            auto pressureInLayeredView = pressureInHandle.layeredView();

            for (uint32_t j = 0; j < pressureOutHandle.numLayers; ++j) {
                auto pressureOutLayerView = pressureOutHandle.layerView(j);
                auto prevPressureInLayerView = prevPressureInHandle.layerView(j);
                auto pressureInLayerView = pressureInHandle.layerView(j);

                auto curLevelParams = pressureOutLayeredView.params;
                shiftToLevel(&curLevelParams, level);

                auto nextLevelParams =
                    params->legacyMode ? curLevelParams : pressureOutLayeredView.params;
                shiftToLevel(&nextLevelParams, level + 1);

                auto mapped = (PressureShaderParams::Prolong *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                mapped->outLevelParams = curLevelParams;
                mapped->inLevelParams = nextLevelParams;
                mapped->pressureParams = pressureOutLayeredView.params;
                NvFlowConstantBufferUnmap(context, m_constantBuffer);

                NvFlowDim gridDim;
                gridDim.x =
                    (curLevelParams.blockDim.w * pressureOutLayerView.mapping.numBlocks +
                     255) /
                    0x100;
                gridDim.y = 1;
                gridDim.z = 1;

                NvFlowDispatchParams dparams = {};
                dparams.shader = m_prolongCS;
                dparams.gridDim = gridDim;
                dparams.rootConstantBuffer = m_constantBuffer;
                dparams.readOnly[0] = pressureOutLayerView.mapping.blockList;
                dparams.readOnly[1] = pressureOutLayerView.mapping.blockTable;
                dparams.readOnly[2] = prevPressureInLayerView.data;
                dparams.readOnly[3] = pressureInLayerView.data;
                dparams.readWrite[0] = pressureOutLayerView.data;
                NvFlowContextDispatch(context, &dparams);
            }

            pressureIn->swap(result);
        }

        prevPressure.releaseTexture();
        smooth(context, pressureIn, level);
    }
}

void PressureImpl::smooth(NvFlowContext *context, SparseTextureFront *pressureIn,
                          uint32_t level) {
    auto pressure = pressureIn;
    uint32_t coarseIterations = 1;
    float dx2 = float(1 << (2 * level));
    if (level == maxLevels - 1)
        coarseIterations = this->coarseIterations;
    else
        coarseIterations = this->fineIterations;

    uint32_t smoothIterations = coarseIterations;

    for (uint32_t i = 0; i < smoothIterations; ++i) {
        auto result = pressure->acquireTexture(context);
        auto pressureOutHandle = result.writePointHandle(context);
        auto pressureInHandle = pressure->front.readPointHandle(context);
        auto pressureOutLayeredView = pressureOutHandle.layeredView();
        auto pressureInLayeredView = pressureInHandle.layeredView();
        for (uint32_t layerIdx = 0; layerIdx < pressureOutHandle.numLayers; ++layerIdx) {
            auto pressureOutLayerView = pressureOutHandle.layerView(layerIdx);
            auto pressureInLayerView = pressureInHandle.layerView(layerIdx);

            auto levelParams = pressureOutLayeredView.params;
            shiftToLevel(&levelParams, level);
            auto mapped = (PressureShaderParams::Jacobi *)NvFlowConstantBufferMap(
                context, m_constantBuffer);
            mapped->levelParams = levelParams;
            mapped->pressureParams = pressureOutLayeredView.params;
            mapped->dx2 = make_float4(dx2);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);

            NvFlowDim gridDim;
            gridDim.x =
                (levelParams.blockDim.w * pressureOutLayerView.mapping.numBlocks + 255) /
                0x100;
            gridDim.y = 1;
            gridDim.z = 1;

            NvFlowDispatchParams dispatchParams = {};
            dispatchParams.shader = m_jacobiCS;
            dispatchParams.gridDim = gridDim;
            dispatchParams.rootConstantBuffer = m_constantBuffer;
            dispatchParams.readOnly[0] = pressureOutLayerView.mapping.blockList;
            dispatchParams.readOnly[1] = pressureOutLayerView.mapping.blockTable;
            dispatchParams.readOnly[2] = pressureInLayerView.data;
            dispatchParams.readWrite[0] = pressureOutLayerView.data;
            NvFlowContextDispatch(context, &dispatchParams);
        }
        pressure->swap(result);
    }
}

#include "divergenceCS.hlsl.h"
#include "jacobiCS.hlsl.h"
#include "restrictCS.hlsl.h"
#include "prolongCS.hlsl.h"
#include "subtractCS.hlsl.h"

PressureImpl::PressureImpl(NvFlowContext *context, const PressureDesc *desc)
    : m_divergenceCS(0),
      m_jacobiCS(0),
      m_restrictCS(0),
      m_prolongCS(0),
      m_subtractCS(0),
      m_constantBuffer(0),
      fineIterations(1),
      coarseIterations(4),
      maxLevels(4) {
    m_desc = *desc;

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_divergenceCS = createShader(NVFLOW_CREATE_SHADER_ARGS(divergenceCS));
    m_jacobiCS = createShader(NVFLOW_CREATE_SHADER_ARGS(jacobiCS));
    m_restrictCS = createShader(NVFLOW_CREATE_SHADER_ARGS(restrictCS));
    m_prolongCS = createShader(NVFLOW_CREATE_SHADER_ARGS(prolongCS));
    m_subtractCS = createShader(NVFLOW_CREATE_SHADER_ARGS(subtractCS));

    NvFlowConstantBufferDesc bufDesc;
    bufDesc.sizeInBytes = sizeof(PressureShaderParams);
    bufDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &bufDesc);
}

PressureImpl::~PressureImpl() {
    SafeRelease(m_divergenceCS);
    SafeRelease(m_jacobiCS);
    SafeRelease(m_restrictCS);
    SafeRelease(m_prolongCS);
    SafeRelease(m_subtractCS);
    SafeRelease(m_constantBuffer);
}

Pressure *createPressure(NvFlowContext *context, const PressureDesc *desc) {
    return new PressureImpl(context, desc);
}

}  // namespace NvFlow