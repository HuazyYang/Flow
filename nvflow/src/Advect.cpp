#include "Advect.h"
#include "ClientHelper.h"

namespace NvFlow {

struct AdvectShaderParams {
    NvFlowShaderLinearParams outputParams;
    NvFlowShaderLinearParams valueParams;
    NvFlowShaderLinearParams velocityParams;
    NvFlowInt4 vidxOldOffset;
    NvFlowFloat4 deltaTime;
    NvFlowFloat4 damping;
};

struct CombustionShaderParams {
    NvFlowFloat3 gravity;
    float burnPerFuel;
    float ignitionTemp;
    float burnPerTemp;
    float fuelPerBurn;
    float tempPerBurn;
    float smokePerBurn;
    float divergencePerBurn;
    float buoyancyPerTemp;
    float coolingRate;
    NvFlowFloat4 deltaTime;
};

struct AdvectSinglePassShaderParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowShaderPointParams outputParams;
    NvFlowShaderLinearParams valueParams;
    NvFlowShaderLinearParams velocityParams;
    NvFlowShaderLinearParams densityParams;
    NvFlowFloat4 vidxOldOffsetf;
    NvFlowFloat4 vidxNormOldOffsetf;
    NvFlowFloat4 deltaTime;
    NvFlowFloat4 blendFactor;
    NvFlowFloat4 blendThreshold;
    NvFlowFloat4 damping;
    NvFlowFloat4 linearFade;
    NvFlowUint4 combustMode;
    CombustionShaderParams combust;
    NvFlowUint4 emitterCount;
    uint32_t materialIdx;
    uint32_t matPad0;
    uint32_t matPad1;
    uint32_t matPad2;
};

struct MacCormackShaderParams {
    NvFlowFloat4x4 gridToWorld;
    NvFlowShaderPointParams outputParams;
    NvFlowShaderLinearParams valueParams;
    NvFlowShaderLinearParams predictParams;
    NvFlowShaderLinearParams velocityParams;
    NvFlowShaderLinearParams densityParams;
    NvFlowFloat4 vidxOldOffsetf;
    NvFlowFloat4 vidxNormOldOffsetf;
    NvFlowFloat4 deltaTime;
    NvFlowFloat4 blendFactor;
    NvFlowFloat4 blendThreshold;
    NvFlowFloat4 damping;
    NvFlowFloat4 linearFade;
    NvFlowUint4 combustMode;
    NvFlow::CombustionShaderParams combust;
    NvFlowUint4 emitterCount;
    unsigned int materialIdx;
    unsigned int matPad0;
    unsigned int matPad1;
    unsigned int matPad2;
};

struct AdvectImpl : Object, Advect {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void advect(NvFlowContext *context, SparseTextureFront *value,
                SparseTextureFront *velocity, const AdvectParams *params) override;

    void advectCombustDensity(NvFlowContext *context, SparseTextureFront *density,
                              SparseTextureFront *velocity,
                              SparseTextureFront *coarseDensity, SparseFadeField *fadeField,
                              NvFlowResource *emitterParameters,
                              const AdvectParams *params) override;

    void advectCombustVelocity(NvFlowContext *context, SparseTextureFront *velocity,
                               SparseTextureFront *density,
                               SparseTextureFront *coarseDensity,
                               SparseFadeField *fadeField,
                               NvFlowResource *emitterParameters,
                               const AdvectParams *params) override;

    // Details
    AdvectImpl(NvFlowContext *context, const AdvectDesc *desc);
    ~AdvectImpl();

    void advect(NvFlowContext *context, SparseTextureFront *value,
                SparseTextureFront *velocity, SparseTextureFront *density,
                SparseTextureFront *coarseDensity, SparseFadeField *fadeField,
                NvFlowResource *emitterParameters, unsigned int mode,
                const AdvectParams *params);

    void advectMultiPass(NvFlowContext *context, SparseTextureFront *value,
                         SparseTextureFront *velocity, SparseTextureFront *density,
                         SparseTextureFront *coarseDensity, SparseFadeField *fadeField,
                         NvFlowResource *emitterParameters, uint32_t mode,
                         const AdvectParams *params);

    void advectSinglePass(NvFlowContext *context, SparseTextureFront *value,
                          SparseTextureFront *velocity, SparseTextureFront *density,
                          SparseTextureFront *coarseDensity, SparseFadeField *fadeField,
                          NvFlowResource *emitterParameters, uint32_t mode,
                          const AdvectParams *params);

    AdvectDesc m_desc;
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowComputeShader *m_advectCS;
    NvFlowComputeShader *m_advectCS_SST;
    NvFlowComputeShader *m_advectCS_VTR;
    NvFlowComputeShader *m_macCormackCS;
    NvFlowComputeShader *m_macCormackCS_densityEmit_SST;
    NvFlowComputeShader *m_macCormackCS_densityNoEmit_SST;
    NvFlowComputeShader *m_macCormackCS_densitySimpleEmit_SST;
    NvFlowComputeShader *m_macCormackCS_velocityEmit_SST;
    NvFlowComputeShader *m_macCormackCS_velocityNoEmit_SST;
    NvFlowComputeShader *m_macCormackCS_velocitySimpleEmit_SST;
    NvFlowComputeShader *m_macCormackCS_densityEmit_VTR;
    NvFlowComputeShader *m_macCormackCS_densityNoEmit_VTR;
    NvFlowComputeShader *m_macCormackCS_densitySimpleEmit_VTR;
    NvFlowComputeShader *m_macCormackCS_velocityEmit_VTR;
    NvFlowComputeShader *m_macCormackCS_velocityNoEmit_VTR;
    NvFlowComputeShader *m_macCormackCS_velocitySimpleEmit_VTR;
    NvFlowComputeShader *m_advectSinglePassCS;
    NvFlowComputeShader *m_advectSinglePassCS_densityEmit_SST;
    NvFlowComputeShader *m_advectSinglePassCS_densityNoEmit_SST;
    NvFlowComputeShader *m_advectSinglePassCS_densitySimpleEmit_SST;
    NvFlowComputeShader *m_advectSinglePassCS_velocityEmit_SST;
    NvFlowComputeShader *m_advectSinglePassCS_velocityNoEmit_SST;
    NvFlowComputeShader *m_advectSinglePassCS_velocitySimpleEmit_SST;
    NvFlowComputeShader *m_advectSinglePassCS_densityEmit_VTR;
    NvFlowComputeShader *m_advectSinglePassCS_densityNoEmit_VTR;
    NvFlowComputeShader *m_advectSinglePassCS_densitySimpleEmit_VTR;
    NvFlowComputeShader *m_advectSinglePassCS_velocityEmit_VTR;
    NvFlowComputeShader *m_advectSinglePassCS_velocityNoEmit_VTR;
    NvFlowComputeShader *m_advectSinglePassCS_velocitySimpleEmit_VTR;
    NvFlowComputeShader *m_advectFirstOrderCS_densityEmit_SST;
    NvFlowComputeShader *m_advectFirstOrderCS_densityNoEmit_SST;
    NvFlowComputeShader *m_advectFirstOrderCS_velocityEmit_SST;
    NvFlowComputeShader *m_advectFirstOrderCS_velocityNoEmit_SST;
    NvFlowComputeShader *m_advectFirstOrderCS_densityEmit_VTR;
    NvFlowComputeShader *m_advectFirstOrderCS_densityNoEmit_VTR;
    NvFlowComputeShader *m_advectFirstOrderCS_velocityEmit_VTR;
    NvFlowComputeShader *m_advectFirstOrderCS_velocityNoEmit_VTR;
};

#include "advectCS.hlsl.h"
#include "advectCS_SST.hlsl.h"
#include "advectCS_VTR.hlsl.h"

#include "macCormackCS.hlsl.h"
#include "macCormackCS_densityEmit_SST.hlsl.h"
#include "macCormackCS_densityNoEmit_SST.hlsl.h"
#include "macCormackCS_densitySimpleEmit_SST.hlsl.h"
#include "macCormackCS_velocityEmit_SST.hlsl.h"
#include "macCormackCS_velocityNoEmit_SST.hlsl.h"
#include "macCormackCS_velocitySimpleEmit_SST.hlsl.h"
#include "macCormackCS_densityEmit_VTR.hlsl.h"
#include "macCormackCS_densityNoEmit_VTR.hlsl.h"
#include "macCormackCS_densitySimpleEmit_VTR.hlsl.h"
#include "macCormackCS_velocityEmit_VTR.hlsl.h"
#include "macCormackCS_velocityNoEmit_VTR.hlsl.h"
#include "macCormackCS_velocitySimpleEmit_VTR.hlsl.h"

#include "advectSinglePassCS.hlsl.h"
#include "advectSinglePassCS_densityEmit_SST.hlsl.h"
#include "advectSinglePassCS_densityNoEmit_SST.hlsl.h"
#include "advectSinglePassCS_densitySimpleEmit_SST.hlsl.h"
#include "advectSinglePassCS_velocityEmit_SST.hlsl.h"
#include "advectSinglePassCS_velocityNoEmit_SST.hlsl.h"
#include "advectSinglePassCS_velocitySimpleEmit_SST.hlsl.h"
#include "advectSinglePassCS_densityEmit_VTR.hlsl.h"
#include "advectSinglePassCS_densityNoEmit_VTR.hlsl.h"
#include "advectSinglePassCS_densitySimpleEmit_VTR.hlsl.h"
#include "advectSinglePassCS_velocityEmit_VTR.hlsl.h"
#include "advectSinglePassCS_velocityNoEmit_VTR.hlsl.h"
#include "advectSinglePassCS_velocitySimpleEmit_VTR.hlsl.h"

#include "advectFirstOrderCS_densityEmit_SST.hlsl.h"
#include "advectFirstOrderCS_densityNoEmit_SST.hlsl.h"
#include "advectFirstOrderCS_velocityEmit_SST.hlsl.h"
#include "advectFirstOrderCS_velocityNoEmit_SST.hlsl.h"
#include "advectFirstOrderCS_densityEmit_VTR.hlsl.h"
#include "advectFirstOrderCS_densityNoEmit_VTR.hlsl.h"
#include "advectFirstOrderCS_velocityEmit_VTR.hlsl.h"
#include "advectFirstOrderCS_velocityNoEmit_VTR.hlsl.h"

uint64_t AdvectImpl::getGPUBytesUsed() {
    return 0;
}

void AdvectImpl::advect(NvFlowContext *context, SparseTextureFront *value,
                        SparseTextureFront *velocity, const AdvectParams *params) {
    advect(context, value, velocity, nullptr, nullptr, nullptr, nullptr, 0, params);
}

void AdvectImpl::advectCombustDensity(NvFlowContext *context, SparseTextureFront *density,
                                      SparseTextureFront *velocity,
                                      SparseTextureFront *coarseDensity,
                                      SparseFadeField *fadeField,
                                      NvFlowResource *emitterParameters,
                                      const AdvectParams *params) {
    advect(context, density, velocity, nullptr, coarseDensity, fadeField, emitterParameters,
           1, params);
}

void AdvectImpl::advectCombustVelocity(NvFlowContext *context, SparseTextureFront *velocity,
                                       SparseTextureFront *density,
                                       SparseTextureFront *coarseDensity,
                                       SparseFadeField *fadeField,
                                       NvFlowResource *emitterParameters,
                                       const AdvectParams *params) {
    advect(context, velocity, velocity, density, coarseDensity, fadeField,
           emitterParameters, 2, params);
}

AdvectImpl::AdvectImpl(NvFlowContext *context, const AdvectDesc *desc)
    : m_constantBuffer(0),
      m_advectCS(0),
      m_advectCS_SST(0),
      m_advectCS_VTR(0),
      m_macCormackCS(0),
      m_macCormackCS_densityEmit_SST(0),
      m_macCormackCS_densityNoEmit_SST(0),
      m_macCormackCS_densitySimpleEmit_SST(0),
      m_macCormackCS_velocityEmit_SST(0),
      m_macCormackCS_velocityNoEmit_SST(0),
      m_macCormackCS_velocitySimpleEmit_SST(0),
      m_macCormackCS_densityEmit_VTR(0),
      m_macCormackCS_densityNoEmit_VTR(0),
      m_macCormackCS_densitySimpleEmit_VTR(0),
      m_macCormackCS_velocityEmit_VTR(0),
      m_macCormackCS_velocityNoEmit_VTR(0),
      m_macCormackCS_velocitySimpleEmit_VTR(0),
      m_advectSinglePassCS(0),
      m_advectSinglePassCS_densityEmit_SST(0),
      m_advectSinglePassCS_densityNoEmit_SST(0),
      m_advectSinglePassCS_densitySimpleEmit_SST(0),
      m_advectSinglePassCS_velocityEmit_SST(0),
      m_advectSinglePassCS_velocityNoEmit_SST(0),
      m_advectSinglePassCS_velocitySimpleEmit_SST(0),
      m_advectSinglePassCS_densityEmit_VTR(0),
      m_advectSinglePassCS_densityNoEmit_VTR(0),
      m_advectSinglePassCS_densitySimpleEmit_VTR(0),
      m_advectSinglePassCS_velocityEmit_VTR(0),
      m_advectSinglePassCS_velocityNoEmit_VTR(0),
      m_advectSinglePassCS_velocitySimpleEmit_VTR(0),
      m_advectFirstOrderCS_densityEmit_SST(0),
      m_advectFirstOrderCS_densityNoEmit_SST(0),
      m_advectFirstOrderCS_velocityEmit_SST(0),
      m_advectFirstOrderCS_velocityNoEmit_SST(0),
      m_advectFirstOrderCS_densityEmit_VTR(0),
      m_advectFirstOrderCS_densityNoEmit_VTR(0),
      m_advectFirstOrderCS_velocityEmit_VTR(0),
      m_advectFirstOrderCS_velocityNoEmit_VTR(0) {
    m_desc = *desc;

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_advectCS = createShader(NVFLOW_CREATE_SHADER_ARGS(advectCS));
    m_advectCS_SST = createShader(NVFLOW_CREATE_SHADER_ARGS(advectCS_SST));
    m_advectCS_VTR = createShader(NVFLOW_CREATE_SHADER_ARGS(advectCS_VTR));
    m_macCormackCS = createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS));
    m_macCormackCS_densityEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_densityEmit_SST));
    m_macCormackCS_densityNoEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_densityNoEmit_SST));
    m_macCormackCS_densitySimpleEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_densitySimpleEmit_SST));
    m_macCormackCS_velocityEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_velocityEmit_SST));
    m_macCormackCS_velocityNoEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_velocityNoEmit_SST));
    m_macCormackCS_velocitySimpleEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_velocitySimpleEmit_SST));
    m_macCormackCS_densityEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_densityEmit_VTR));
    m_macCormackCS_densityNoEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_densityNoEmit_VTR));
    m_macCormackCS_densitySimpleEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_densitySimpleEmit_VTR));
    m_macCormackCS_velocityEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_velocityEmit_VTR));
    m_macCormackCS_velocityNoEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_velocityNoEmit_VTR));
    m_macCormackCS_velocitySimpleEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(macCormackCS_velocitySimpleEmit_VTR));
    m_advectSinglePassCS = createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS));
    m_advectSinglePassCS_densityEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_densityEmit_SST));
    m_advectSinglePassCS_densityNoEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_densityNoEmit_SST));
    m_advectSinglePassCS_densitySimpleEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_densitySimpleEmit_SST));
    m_advectSinglePassCS_velocityEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_velocityEmit_SST));
    m_advectSinglePassCS_velocityNoEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_velocityNoEmit_SST));
    m_advectSinglePassCS_velocitySimpleEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_velocitySimpleEmit_SST));
    m_advectSinglePassCS_densityEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_densityEmit_VTR));
    m_advectSinglePassCS_densityNoEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_densityNoEmit_VTR));
    m_advectSinglePassCS_densitySimpleEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_densitySimpleEmit_VTR));
    m_advectSinglePassCS_velocityEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_velocityEmit_VTR));
    m_advectSinglePassCS_velocityNoEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_velocityNoEmit_VTR));
    m_advectSinglePassCS_velocitySimpleEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectSinglePassCS_velocitySimpleEmit_VTR));
    m_advectFirstOrderCS_densityEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_densityEmit_SST));
    m_advectFirstOrderCS_densityNoEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_densityNoEmit_SST));
    m_advectFirstOrderCS_velocityEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_velocityEmit_SST));
    m_advectFirstOrderCS_velocityNoEmit_SST =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_velocityNoEmit_SST));
    m_advectFirstOrderCS_densityEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_densityEmit_VTR));
    m_advectFirstOrderCS_densityNoEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_densityNoEmit_VTR));
    m_advectFirstOrderCS_velocityEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_velocityEmit_VTR));
    m_advectFirstOrderCS_velocityNoEmit_VTR =
        createShader(NVFLOW_CREATE_SHADER_ARGS(advectFirstOrderCS_velocityNoEmit_VTR));

    NvFlowConstantBufferDesc buffDesc;
    buffDesc.sizeInBytes = 1072;
    buffDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &buffDesc);
    auto velocityTemp = m_desc.velocity->acquireTexture(context);
    velocityTemp.releaseTexture();
}

AdvectImpl::~AdvectImpl() {
    SafeRelease(m_constantBuffer);
    SafeRelease(m_advectCS);
    SafeRelease(m_advectCS_SST);
    SafeRelease(m_advectCS_VTR);
    SafeRelease(m_macCormackCS);
    SafeRelease(m_macCormackCS_densityEmit_SST);
    SafeRelease(m_macCormackCS_densityNoEmit_SST);
    SafeRelease(m_macCormackCS_densitySimpleEmit_SST);
    SafeRelease(m_macCormackCS_velocityEmit_SST);
    SafeRelease(m_macCormackCS_velocityNoEmit_SST);
    SafeRelease(m_macCormackCS_velocitySimpleEmit_SST);
    SafeRelease(m_macCormackCS_densityEmit_VTR);
    SafeRelease(m_macCormackCS_densityNoEmit_VTR);
    SafeRelease(m_macCormackCS_densitySimpleEmit_VTR);
    SafeRelease(m_macCormackCS_velocityEmit_VTR);
    SafeRelease(m_macCormackCS_velocityNoEmit_VTR);
    SafeRelease(m_macCormackCS_velocitySimpleEmit_VTR);
    SafeRelease(m_advectSinglePassCS);
    SafeRelease(m_advectSinglePassCS_densityEmit_SST);
    SafeRelease(m_advectSinglePassCS_densityNoEmit_SST);
    SafeRelease(m_advectSinglePassCS_densitySimpleEmit_SST);
    SafeRelease(m_advectSinglePassCS_velocityEmit_SST);
    SafeRelease(m_advectSinglePassCS_velocityNoEmit_SST);
    SafeRelease(m_advectSinglePassCS_velocitySimpleEmit_SST);
    SafeRelease(m_advectSinglePassCS_densityEmit_VTR);
    SafeRelease(m_advectSinglePassCS_densityNoEmit_VTR);
    SafeRelease(m_advectSinglePassCS_densitySimpleEmit_VTR);
    SafeRelease(m_advectSinglePassCS_velocityEmit_VTR);
    SafeRelease(m_advectSinglePassCS_velocityNoEmit_VTR);
    SafeRelease(m_advectSinglePassCS_velocitySimpleEmit_VTR);
    SafeRelease(m_advectFirstOrderCS_densityEmit_SST);
    SafeRelease(m_advectFirstOrderCS_densityNoEmit_SST);
    SafeRelease(m_advectFirstOrderCS_velocityEmit_SST);
    SafeRelease(m_advectFirstOrderCS_velocityNoEmit_SST);
    SafeRelease(m_advectFirstOrderCS_densityEmit_VTR);
    SafeRelease(m_advectFirstOrderCS_densityNoEmit_VTR);
    SafeRelease(m_advectFirstOrderCS_velocityEmit_VTR);
    SafeRelease(m_advectFirstOrderCS_velocityNoEmit_VTR);
}

void AdvectImpl::advect(NvFlowContext *context, SparseTextureFront *value,
                        SparseTextureFront *velocity, SparseTextureFront *density,
                        SparseTextureFront *coarseDensity, SparseFadeField *fadeField,
                        NvFlowResource *emitterParameters, unsigned int mode,
                        const AdvectParams *params) {
    if (params->singlePassAdvection) {
        advectSinglePass(context, value, velocity, density, coarseDensity, fadeField,
                         emitterParameters, mode, params);
    } else {
        advectMultiPass(context, value, velocity, density, coarseDensity, fadeField,
                        emitterParameters, mode, params);
    }
}

void AdvectImpl::advectMultiPass(NvFlowContext *context, SparseTextureFront *value,
                                 SparseTextureFront *velocity, SparseTextureFront *density,
                                 SparseTextureFront *coarseDensity,
                                 SparseFadeField *fadeField,
                                 NvFlowResource *emitterParameters, uint32_t mode,
                                 const AdvectParams *params) {
    auto result = value->acquireTexture(context);
    auto result2 = value->acquireTexture(context);

    auto valueWrite = result.writeLinearHandle(context);
    auto valueRead = value->front.readLinearHandle(context);
    auto velocityRead = velocity->front.readLinearHandle(context);

    auto valueWriteLayered = valueWrite.layeredView();
    auto valueReadLayered = valueRead.layeredView();
    auto velocityReadLayered = velocityRead.layeredView();

    NvFlowFloat3 gridLocationOffset = params->gridNewLocation - params->gridOldLocation;
    NvFlowInt4 vidxOldOffset;
    (NvFlowInt3 &)vidxOldOffset =
        make_int3((const NvFlowFloat3 &)valueWriteLayered.params.vdim *
                  (gridLocationOffset / (2.f * params->gridHalfSize)));
    vidxOldOffset.w = 0;
    NvFlowFloat4 vidxOldOffsetf = make_float4(vidxOldOffset);

    NvFlowFloat4 vidxNormOldOffsetf;
    (NvFlowFloat3 &)vidxNormOldOffsetf =
        (const NvFlowFloat3 &)valueWriteLayered.params.vdimInv *
        (const NvFlowFloat3 &)vidxOldOffsetf;
    vidxNormOldOffsetf.w = 0;

    for (uint32_t layerIdx = 0; layerIdx < valueWrite.numLayers; ++layerIdx) {
        auto valueWriteLayer = valueWrite.layerView(layerIdx);
        auto valueReadLayer = valueRead.layerView(layerIdx);
        auto velocityReadLayer = velocityRead.layerView(layerIdx);

        auto mappedCB =
            (AdvectShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        mappedCB->outputParams = valueWriteLayered.params;
        mappedCB->valueParams = valueReadLayered.params;
        mappedCB->velocityParams = velocityReadLayered.params;
        mappedCB->vidxOldOffset = vidxOldOffset;
        NvFlowFloat4 deltaTime;
        (NvFlowFloat3 &)deltaTime = params->deltaTime / params->valueCellSize;
        deltaTime.w = params->deltaTime;
        mappedCB->deltaTime = deltaTime;
        mappedCB->damping = make_float4(1.f);
        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDim gridDimVTR;
        gridDimVTR.x =
            (valueWriteLayered.params.blockDim.x * valueWriteLayer.mapping.numBlocks + 7) / 8;
        gridDimVTR.y = (valueWriteLayered.params.blockDim.y + 7) / 8;
        gridDimVTR.z = (valueWriteLayered.params.blockDim.z + 7)  / 8;
        NvFlowDim gridDimSST;
        gridDimSST.x = (valueWriteLayered.params.linearBlockDim.z *
                            valueWriteLayered.params.linearBlockDim.y *
                            valueWriteLayered.params.linearBlockDim.x +
                        127) /
                       0x80;
        gridDimSST.y = valueWriteLayer.mapping.numBlocks;
        gridDimSST.z = 1;

        bool enableVTR = m_desc.enableVTR;

        NvFlowDispatchParams paramsa = {};
        paramsa.shader = enableVTR ? m_advectCS_VTR : m_advectCS_SST;
        paramsa.gridDim = enableVTR ? gridDimVTR : gridDimSST;
        paramsa.rootConstantBuffer = m_constantBuffer;
        paramsa.readOnly[0] = valueWriteLayer.mapping.blockList;
        paramsa.readOnly[1] = valueWriteLayer.mapping.blockTable;
        paramsa.readOnly[2] = valueReadLayer.data;
        paramsa.readOnly[3] = valueReadLayer.mapping.blockTable;
        paramsa.readOnly[4] = velocityReadLayer.data;
        paramsa.readOnly[5] = velocityReadLayer.mapping.blockTable;
        paramsa.readWrite[0] = valueWriteLayer.data;
        NvFlowContextDispatch(context, &paramsa);
    }

    bool hasDensity2 = density != nullptr;
    bool combustVelocity2 = coarseDensity != nullptr && mode == 2;
    bool combustDensity2 = coarseDensity != nullptr && mode == 1;

    auto valueWrite2 = result2.writePointHandle(context);
    auto valueRead2 = value->front.readLinearHandle(context);
    auto valueRead1 = result.readLinearHandle(context);
    auto velocityRead2 = velocity->front.readLinearHandle(context);

    SparseReadPointHandle densityRead2;
    if (hasDensity2)
        densityRead2 = density->front.readPointHandle(context);
    else
        ZeroMemory(&densityRead2, sizeof(densityRead2));

    SparseWritePointHandle coarseDensityWrite2;
    if (combustVelocity2)
        coarseDensityWrite2 = coarseDensity->front.writePointHandle(context);
    else
        ZeroMemory(&coarseDensityWrite2, sizeof(coarseDensityWrite2));

    SparseReadLinearHandle coarseDensityRead2;
    if (combustDensity2)
        coarseDensityRead2 = coarseDensity->front.readLinearHandle(context);
    else
        ZeroMemory(&coarseDensityRead2, sizeof(coarseDensityRead2));

    auto valueWrite2Layered = valueWrite2.layeredView();
    auto valueRead2Layered = valueRead2.layeredView();
    auto valueRead1Layered = valueRead1.layeredView();
    auto velocityRead2Layered = velocityRead2.layeredView();

    SparseReadPointLayeredView densityRead2Layered;
    if (hasDensity2)
        densityRead2Layered = densityRead2.layeredView();
    else
        ZeroMemory(&densityRead2Layered, sizeof(densityRead2Layered));

    SparseWritePointLayeredView coarseDensityWrite2Layered;
    if (combustVelocity2)
        coarseDensityWrite2Layered = coarseDensityWrite2.layeredView();
    else
        ZeroMemory(&coarseDensityWrite2Layered, sizeof(coarseDensityWrite2Layered));

    SparseReadLinearLayeredView coarseDensityRead2Layered;
    if (combustDensity2)
        coarseDensityRead2Layered = coarseDensityRead2.layeredView();
    else
        ZeroMemory(&coarseDensityRead2Layered, sizeof(coarseDensityRead2Layered));

    for (uint32_t j = 0; j < valueWrite2.numLayers; ++j) {
        AdvectPerLayerParams perLayer;
        params->getPerLayer(&perLayer, params->userdata, j);

        auto valueWrite2Layer = valueWrite2.layerView(j);
        auto valueRead2Layer = valueRead2.layerView(j);
        auto valueRead1Layer = valueRead1.layerView(j);
        auto velocityRead2Layer = velocityRead2.layerView(j);

        SparseReadPointLayerView densityRead2Layer;
        if (hasDensity2)
            densityRead2Layer = densityRead2.layerView(j);
        else
            ZeroMemory(&densityRead2Layer, sizeof(densityRead2Layer));

        SparseWritePointLayerView coarseDensityWrite2Layer;
        SparseReadLinearLayerView coarseDensityRead2Layer;

        if (combustVelocity2)
            coarseDensityWrite2Layer = coarseDensityWrite2.layerView(j);
        else
            ZeroMemory(&coarseDensityWrite2Layer, sizeof(coarseDensityWrite2Layer));

        if (combustDensity2)
            coarseDensityRead2Layer = coarseDensityRead2.layerView(j);
        else
            ZeroMemory(&coarseDensityRead2Layer, sizeof(coarseDensityRead2Layer));

        auto mappedCB = (MacCormackShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        mappedCB->gridToWorld = transpose(params->gridToWorld);
        mappedCB->outputParams = valueWrite2Layered.params;
        mappedCB->valueParams = valueRead2Layered.params;
        mappedCB->predictParams = valueRead1Layered.params;
        mappedCB->velocityParams = velocityRead2Layered.params;
        mappedCB->densityParams = densityRead2Layered.downsampleParams;
        mappedCB->vidxOldOffsetf = vidxOldOffsetf;
        mappedCB->vidxNormOldOffsetf = vidxNormOldOffsetf;
        (NvFlowFloat3 &)mappedCB->deltaTime = params->deltaTime / params->valueCellSize;
        mappedCB->deltaTime.w = params->deltaTime;
        mappedCB->blendFactor = perLayer.blendFactor;
        mappedCB->blendThreshold = perLayer.blendThreshold;

        mappedCB->damping = pow(1.f - perLayer.damping, params->deltaTime);
        mappedCB->linearFade = perLayer.fade;
        mappedCB->combustMode = make_uint4(mode);
        mappedCB->combust.gravity = params->gravity;
        mappedCB->combust.burnPerFuel = 1.f / perLayer.fuelPerBurn;
        mappedCB->combust.ignitionTemp = perLayer.ignitionTemp;
        mappedCB->combust.burnPerTemp = perLayer.burnPerTemp;
        mappedCB->combust.fuelPerBurn = perLayer.fuelPerBurn;
        mappedCB->combust.tempPerBurn = perLayer.tempPerBurn;
        mappedCB->combust.smokePerBurn = perLayer.smokePerBurn;
        mappedCB->combust.divergencePerBurn = perLayer.divergencePerBurn;
        mappedCB->combust.buoyancyPerTemp = perLayer.buoyancyPerTemp;
        mappedCB->combust.coolingRate = perLayer.coolingRate;
        mappedCB->combust.deltaTime = make_float4(params->deltaTime);

        mappedCB->emitterCount = params->emitterCount;
        mappedCB->materialIdx = perLayer.materialIdx;

        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDim gridDim;
        gridDim.x =
            (valueWrite2Layered.params.blockDim.x * valueWrite2Layer.mapping.numBlocks +
             7) / 8;
        gridDim.y = (valueWrite2Layered.params.blockDim.y + 7) / 8;
        gridDim.z = (valueWrite2Layered.params.blockDim.z + 7) / 8;

        bool enableVTR = m_desc.enableVTR;

        NvFlowDispatchParams paramsa = {};
        NvFlowComputeShader *shader = m_macCormackCS;
        if (mode == 1) {
            if (params->emitterCount.z) {
                if (params->emitterCount.w) {
                    shader = enableVTR ? m_macCormackCS_densityEmit_VTR
                                       : m_macCormackCS_densityEmit_SST;
                } else {
                    shader = enableVTR ? m_macCormackCS_densitySimpleEmit_VTR
                                       : m_macCormackCS_densitySimpleEmit_SST;
                }
            } else {
                shader = enableVTR ? m_macCormackCS_densityNoEmit_VTR
                                   : m_macCormackCS_densityNoEmit_SST;
            }
        } else if (mode == 2) {
            if (params->emitterCount.z) {
                if (params->emitterCount.w) {
                    shader = enableVTR ? m_macCormackCS_velocityEmit_VTR
                                       : m_macCormackCS_velocityEmit_SST;
                } else {
                    shader = enableVTR ? m_macCormackCS_velocitySimpleEmit_VTR
                                       : m_macCormackCS_velocitySimpleEmit_SST;
                }
            } else {
                shader = enableVTR ? m_macCormackCS_velocityNoEmit_VTR
                                   : m_macCormackCS_velocityNoEmit_SST;
            }
        }
        paramsa.shader = shader;

        paramsa.gridDim = gridDim;
        paramsa.rootConstantBuffer = m_constantBuffer;
        paramsa.readOnly[0] = valueWrite2Layer.mapping.blockList;
        paramsa.readOnly[1] = valueWrite2Layer.mapping.blockTable;
        paramsa.readOnly[2] = valueRead2Layer.data;
        paramsa.readOnly[3] = valueRead2Layer.mapping.blockTable;
        paramsa.readOnly[4] = valueRead1Layer.data;
        paramsa.readOnly[5] = valueRead1Layer.mapping.blockTable;
        paramsa.readOnly[6] = velocityRead2Layer.data;
        paramsa.readOnly[7] = velocityRead2Layer.mapping.blockTable;

        if (hasDensity2) {
            paramsa.readOnly[8] = densityRead2Layer.data;
            paramsa.readOnly[9] = densityRead2Layer.mapping.blockTable;
        }

        NvFlowResource *fadeFieldResource = 0;
        if (fadeField) {
            fadeFieldResource = fadeField->getFadeField(fadeField->userData, mode == 2, j);
        }

        paramsa.readOnly[10] = fadeFieldResource;
        if (combustDensity2)
            paramsa.readOnly[11] = coarseDensityRead2Layer.data;

        paramsa.readOnly[12] = emitterParameters;
        paramsa.readOnly[13] = 0;
        paramsa.readWrite[0] = valueWrite2Layer.data;
        if (combustVelocity2)
            paramsa.readWrite[1] = coarseDensityWrite2Layer.data;

        NvFlowContextDispatch(context, &paramsa);
    }

    result.releaseTexture();
    value->swap(result2);
}

void AdvectImpl::advectSinglePass(NvFlowContext *context, SparseTextureFront *value,
                                  SparseTextureFront *velocity, SparseTextureFront *density,
                                  SparseTextureFront *coarseDensity,
                                  SparseFadeField *fadeField,
                                  NvFlowResource *emitterParameters, uint32_t mode,
                                  const AdvectParams *params) {
    auto result = value->acquireTexture(context);
    bool hasDensity = density != nullptr;
    bool combustVelocity = coarseDensity != nullptr && mode == 2;
    bool combustDensity = coarseDensity != nullptr && mode == 1;

    auto valueWriteHandle = result.writePointHandle(context);
    auto valueReadHandle = value->front.readLinearHandle(context);
    auto velocityReadHandle = velocity->front.readLinearHandle(context);

    SparseReadPointHandle densityReadHandle;
    SparseWritePointHandle coarseDensityWriteHandle;
    SparseReadLinearHandle coarseDensityReadHandle;

    if (hasDensity)
        densityReadHandle = density->front.readPointHandle(context);
    else
        ZeroMemory(&densityReadHandle, sizeof(densityReadHandle));

    if (combustVelocity)
        coarseDensityWriteHandle = coarseDensity->front.writePointHandle(context);
    else
        ZeroMemory(&coarseDensityWriteHandle, sizeof(coarseDensityWriteHandle));

    if (combustDensity)
        coarseDensityReadHandle = coarseDensity->front.readLinearHandle(context);
    else
        ZeroMemory(&coarseDensityReadHandle, sizeof(coarseDensityReadHandle));

    auto valueWriteLayered = valueWriteHandle.layeredView();
    auto valueReadLayered = valueReadHandle.layeredView();
    auto velocityReadLayered = velocityReadHandle.layeredView();

    SparseReadPointLayeredView densityReadLayered;
    if (hasDensity)
        densityReadLayered = densityReadHandle.layeredView();
    else
        ZeroMemory(&densityReadLayered, sizeof(densityReadLayered));

    SparseWritePointLayeredView coarseDensityWriteLayered;
    if (combustVelocity)
        coarseDensityWriteLayered = coarseDensityWriteHandle.layeredView();
    else
        ZeroMemory(&coarseDensityWriteLayered, sizeof(coarseDensityWriteLayered));

    SparseReadLinearLayeredView coarseDensityReadLayered;
    if (combustDensity)
        coarseDensityReadLayered = coarseDensityReadHandle.layeredView();
    else
        ZeroMemory(&coarseDensityReadLayered, sizeof(coarseDensityReadLayered));

    NvFlowFloat3 gridLocationOffset = params->gridNewLocation - params->gridOldLocation;
    NvFlowInt4 vidxOldOffset;
    (NvFlowInt3 &)vidxOldOffset =
        make_int3((const NvFlowFloat3 &)valueReadLayered.params.vdim *
                  (gridLocationOffset / (2.f * params->gridHalfSize)));
    vidxOldOffset.w = 0;
    NvFlowFloat4 vidxOldOffsetf = make_float4(vidxOldOffset);

    NvFlowFloat4 vidxNormOldOffsetf;
    (NvFlowFloat3 &)vidxNormOldOffsetf =
        (const NvFlowFloat3 &)valueReadLayered.params.vdimInv *
        (const NvFlowFloat3 &)vidxOldOffsetf;
    vidxNormOldOffsetf.w = 0;

    for (uint32_t layerIdx = 0; layerIdx < valueWriteHandle.numLayers; ++layerIdx) {
        AdvectPerLayerParams perLayer;
        params->getPerLayer(&perLayer, params->userdata, layerIdx);

        auto valueWriteLayer = valueWriteHandle.layerView(layerIdx);
        auto valueReadLayer = valueReadHandle.layerView(layerIdx);
        auto velocityReadLayer = velocityReadHandle.layerView(layerIdx);

        SparseReadPointLayerView densityReadLayer;
        if (hasDensity)
            densityReadLayer = densityReadHandle.layerView(layerIdx);
        else
            ZeroMemory(&densityReadLayer, sizeof(densityReadLayer));

        SparseWritePointLayerView coarseDensityWriteLayer;
        SparseReadLinearLayerView coarseDensityReadLayer;

        if (combustVelocity)
            coarseDensityWriteLayer = coarseDensityWriteHandle.layerView(layerIdx);
        else
            ZeroMemory(&coarseDensityWriteLayer, sizeof(coarseDensityWriteLayer));

        if (combustDensity)
            coarseDensityReadLayer = coarseDensityReadHandle.layerView(layerIdx);
        else
            ZeroMemory(&coarseDensityReadLayer, sizeof(coarseDensityReadLayer));

        auto mappedCB = (AdvectSinglePassShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        mappedCB->gridToWorld = transpose(params->gridToWorld);
        mappedCB->outputParams = valueWriteLayered.params;
        mappedCB->valueParams = valueReadLayered.params;
        mappedCB->velocityParams = velocityReadLayered.params;
        mappedCB->densityParams = densityReadLayered.downsampleParams;
        mappedCB->vidxOldOffsetf = vidxOldOffsetf;
        mappedCB->vidxNormOldOffsetf = vidxNormOldOffsetf;
        (NvFlowFloat3 &)mappedCB->deltaTime = params->deltaTime / params->valueCellSize;
        mappedCB->deltaTime.w = params->deltaTime;
        mappedCB->blendFactor = perLayer.blendFactor;
        mappedCB->blendThreshold = perLayer.blendThreshold;

        mappedCB->damping = pow(1.f - perLayer.damping, params->deltaTime);
        mappedCB->linearFade = perLayer.fade;
        mappedCB->combustMode = make_uint4(mode);
        mappedCB->combust.gravity = params->gravity;
        mappedCB->combust.burnPerFuel = 1.f / perLayer.fuelPerBurn;
        mappedCB->combust.ignitionTemp = perLayer.ignitionTemp;
        mappedCB->combust.burnPerTemp = perLayer.burnPerTemp;
        mappedCB->combust.fuelPerBurn = perLayer.fuelPerBurn;
        mappedCB->combust.tempPerBurn = perLayer.tempPerBurn;
        mappedCB->combust.smokePerBurn = perLayer.smokePerBurn;
        mappedCB->combust.divergencePerBurn = perLayer.divergencePerBurn;
        mappedCB->combust.buoyancyPerTemp = perLayer.buoyancyPerTemp;
        mappedCB->combust.coolingRate = perLayer.coolingRate;
        mappedCB->combust.deltaTime = make_float4(params->deltaTime);

        mappedCB->emitterCount = params->emitterCount;
        mappedCB->materialIdx = perLayer.materialIdx;

        NvFlowConstantBufferUnmap(context, m_constantBuffer);

        NvFlowDim gridDim;
        gridDim.x =
            (valueWriteLayered.params.blockDim.x * valueWriteLayer.mapping.numBlocks + 7) / 8;
        gridDim.y = (valueWriteLayered.params.blockDim.y + 7) / 8;
        gridDim.z = (valueWriteLayered.params.blockDim.z + 7) / 8;

        bool enableVTR = m_desc.enableVTR;

        NvFlowDispatchParams paramsa = {};
        NvFlowComputeShader *shader = m_advectSinglePassCS;
        if (perLayer.blendFactor == make_float4(0.f)) {
            if (mode == 1) {
                if (params->emitterCount.z) {
                    shader = enableVTR ? m_advectFirstOrderCS_densityEmit_VTR
                                       : m_advectFirstOrderCS_densityEmit_SST;
                } else {
                    shader = enableVTR ? m_advectFirstOrderCS_densityNoEmit_VTR
                                       : m_advectFirstOrderCS_densityNoEmit_SST;
                }
            } else if (mode == 2) {
                if (params->emitterCount.z) {
                    shader = enableVTR ? m_advectFirstOrderCS_velocityEmit_VTR
                                       : m_advectFirstOrderCS_velocityEmit_SST;
                } else {
                    shader = enableVTR ? m_advectFirstOrderCS_velocityNoEmit_VTR
                                       : m_advectFirstOrderCS_velocityNoEmit_SST;
                }
            }
        } else {
            if (mode == 1) {
                if (params->emitterCount.z) {
                    if (params->emitterCount.w) {
                        shader = enableVTR ? m_advectSinglePassCS_densityEmit_VTR
                                           : m_advectSinglePassCS_densityEmit_SST;
                    } else {
                        shader = enableVTR ? m_advectSinglePassCS_densitySimpleEmit_VTR
                                           : m_advectSinglePassCS_densitySimpleEmit_SST;
                    }
                } else {
                    shader = enableVTR ? m_advectSinglePassCS_densityNoEmit_VTR
                                       : m_advectSinglePassCS_densityNoEmit_SST;
                }
            } else if (mode == 2) {
                if (params->emitterCount.z) {
                    if (params->emitterCount.w) {
                        shader = enableVTR ? m_advectSinglePassCS_velocityEmit_VTR
                                           : m_advectSinglePassCS_velocityEmit_SST;
                    } else {
                        shader = enableVTR ? m_advectSinglePassCS_velocitySimpleEmit_VTR
                                           : m_advectSinglePassCS_velocitySimpleEmit_SST;
                    }
                } else {
                    shader = enableVTR ? m_advectSinglePassCS_velocityNoEmit_VTR
                                       : m_advectSinglePassCS_velocityNoEmit_SST;
                }
            }
        }
        paramsa.shader = shader;

        paramsa.gridDim = gridDim;
        paramsa.rootConstantBuffer = m_constantBuffer;
        paramsa.readOnly[0] = valueWriteLayer.mapping.blockList;
        paramsa.readOnly[1] = valueWriteLayer.mapping.blockTable;
        paramsa.readOnly[2] = valueReadLayer.data;
        paramsa.readOnly[3] = valueReadLayer.mapping.blockTable;
        paramsa.readOnly[4] = velocityReadLayer.data;
        paramsa.readOnly[5] = velocityReadLayer.mapping.blockTable;

        if (hasDensity) {
            paramsa.readOnly[6] = densityReadLayer.data;
            paramsa.readOnly[7] = densityReadLayer.mapping.blockTable;
        }

        NvFlowResource *fadeFieldResource = 0;
        if (fadeField) {
            fadeFieldResource =
                fadeField->getFadeField(fadeField->userData, mode == 2, layerIdx);
        }

        paramsa.readOnly[8] = fadeFieldResource;
        if (combustDensity)
            paramsa.readOnly[9] = coarseDensityReadLayer.data;
        paramsa.readOnly[10] = emitterParameters;
        paramsa.readOnly[11] = 0;
        paramsa.readWrite[0] = valueWriteLayer.data;
        if (combustVelocity)
            paramsa.readWrite[1] = coarseDensityWriteLayer.data;

        NvFlowContextDispatch(context, &paramsa);
    }
    value->swap(result);
}

Advect *createAdvect(NvFlowContext *context, const AdvectDesc *desc) {
    return new AdvectImpl(context, desc);
}

}  // namespace NvFlow