#ifndef NVFLOW_PARTICLESURFACE_H
#define NVFLOW_PARTICLESURFACE_H
#include "NvFlowImpl.h"
#include "Object.h"
#include "ClientHelper.h"
#include "ParticleSurfaceExport.h"
#include "NvFlowContextImpl.h"

struct NvFlowParticleSurface : NvFlowObject {
    virtual void updateParticles(NvFlowContext *context,
                                 const NvFlowParticleSurfaceData *data) = 0;
    virtual void updateSurface(NvFlowContext *context,
                               const NvFlowParticleSurfaceParams *params) = 0;
    virtual void allocFunc(NvFlowContext *context,
                           const NvFlowGridEmitCustomAllocParams *params) = 0;
    virtual void emitVelocityFunc(NvFlowContext *context, uint32_t *dataFrontIdx,
                                  const NvFlowGridEmitCustomEmitParams *params,
                                  const NvFlowParticleSurfaceEmitParams *emitParams) = 0;
    virtual void emitDensityFunc(NvFlowContext *context, uint32_t *dataFrontIdx,
                                 const NvFlowGridEmitCustomEmitParams *params,
                                 const NvFlowParticleSurfaceEmitParams *emitParams) = 0;
    virtual NvFlowGridExport *debugGridExport(NvFlowContext *context) = 0;
};

namespace NvFlow {

NvFlowParticleSurface *FlowCreateParticleSurface(NvFlowContext *context,
                                                 const NvFlowParticleSurfaceDesc *desc);

struct ParticleSurface : Object, NvFlowParticleSurface {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void updateParticles(NvFlowContext *context,
                         const NvFlowParticleSurfaceData *data) override;
    void updateSurface(NvFlowContext *context,
                       const NvFlowParticleSurfaceParams *params) override;
    void allocFunc(NvFlowContext *context,
                   const NvFlowGridEmitCustomAllocParams *params) override;
    void emitVelocityFunc(NvFlowContext *context, uint32_t *dataFrontIdx,
                          const NvFlowGridEmitCustomEmitParams *params,
                          const NvFlowParticleSurfaceEmitParams *emitParams) override;
    void emitDensityFunc(NvFlowContext *context, uint32_t *dataFrontIdx,
                         const NvFlowGridEmitCustomEmitParams *params,
                         const NvFlowParticleSurfaceEmitParams *emitParams) override;
    NvFlowGridExport *debugGridExport(NvFlowContext *context) override;

    // Details
    struct BlockConfig {
        NvFlowDim virtualDim;
        NvFlowDim blockDim;
        NvFlowDim linearBlockDim;
        NvFlowDim linearBlockOffset;
        NvFlowDim gridDim;
        NvFlowDim poolGridDim;
        NvFlowDim poolDim;
        unsigned int maxVirtualBlocks;
        unsigned int maxBlocks;
    };

    struct EmitParams {
        NvFlowFloat4 deltaTime;
        NvFlowFloat4 coupleRate;
        NvFlowFloat4 emitValue;
    };

    void emitFunc(NvFlowContext *context, unsigned int *dataFrontIdx,
                  const NvFlowGridEmitCustomEmitParams *params,
                  const EmitParams *emitParams);

    void computeSparseTextureConfig(const NvFlowDim &virtualDim, float residentScale,
                                    const NvFlowDim &blockDim);

    void generateLinearParams();

    ParticleSurface(NvFlowContext *context, const NvFlowParticleSurfaceDesc *desc);
    ~ParticleSurface();

    NvFlowParticleSurfaceDesc m_desc;
    NvFlowFloat4x4 m_modelMatrix;
    NvFlowFloat4x4 m_particleToFieldT;
    BlockConfig m_blockConfig;
    NvFlowShaderLinearParams m_linearParams;
    unsigned int m_numBlocks;
    unsigned int m_atomicState;
    unsigned int m_dispatchBlocks;
    float m_surfaceThreshold;
    ParticleSurfaceExport m_export;
    NvFlowBuffer *m_uploadBuffer;
    unsigned int m_particleCount;
    NvFlowTexture3D *m_fieldFront;
    NvFlowTexture3D *m_fieldBack;
    NvFlowTexture3D *m_debugVisTex;
    NvFlowTexture3D *m_blockTable;
    NvFlowBuffer *m_blockList;
    NvFlowBuffer *m_atomicBuf;
    NvFlowConstantBuffer *m_atomicConst;
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowComputeShader *m_particleSurfaceBlockListClearCS;
    NvFlowComputeShader *m_particleSurfaceBlockTableClearCS;
    NvFlowComputeShader *m_particleSurfaceAllocCS;
    NvFlowComputeShader *m_particleSurfaceBlockTableAllocCS;
    NvFlowComputeShader *m_particleSurfaceClearCS;
    NvFlowComputeShader *m_particleSurfaceSplatCS;
    NvFlowComputeShader *m_particleSurfaceSmoothCS;
    NvFlowComputeShader *m_particleSurfaceSmoothXCS;
    NvFlowComputeShader *m_particleSurfaceSmoothYCS;
    NvFlowComputeShader *m_particleSurfaceSmoothZCS;
    NvFlowComputeShader *m_particleSurfaceUpdateLinearCS;
    NvFlowComputeShader *m_particleSurfaceDebugVisCS;
    NvFlowComputeShader *m_particleSurfaceEmitAllocCS;
    NvFlowComputeShader *m_particleSurfaceEmitEmitCS;
};

}  // namespace NvFlow

#endif /* NVFLOW_PARTICLESURFACE_H */
