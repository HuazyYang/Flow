#ifndef BLOCKMANAGER_H
#define BLOCKMANAGER_H

#include "Object.h"
#include "NvFlowContextImpl.h"
#include "ClientHelper.h"
#include "VectorCached.h"

namespace NvFlow {

struct SparseFadeField {
    void *userData;
    NvFlowResource *(*getFadeField)(void *userData, bool, NvFlowUint);
};

struct SparseMapping : NvFlowObject {};

struct BlockManagerDesc {
    bool enableVTR;
    bool lowLatencyMapping;
    NvFlowDim velocityVirtualDim;
    NvFlowDim densityVirtualDim;
};

struct BlockManager : NvFlowObject {};

struct BlockMangerDesc {};

struct BlockManagerImpl : Object, BlockManager {
    struct PerLayer {
        NvFlowTexture3D *m_velocitySummaryTex;
        NvFlowTexture3D *m_densitySummaryTex;
    };

    struct FieldFadePerLayer {
        NvFlowTexture3D *m_fieldFadeVelocity;
        NvFlowTexture3D *m_fieldFadeDensity;
    };

    BlockManagerDesc m_desc;
    NvFlowUint3 m_velDownsample;
    SparseMapping *m_fieldMapping;
    NvFlowComputeShader *m_velocitySummaryCS;
    NvFlowComputeShader *m_densitySummaryCS;
    NvFlowComputeShader *m_densitySummaryCoarseCS;
    NvFlowComputeShader *m_blockManager1CS;
    NvFlowComputeShader *m_blockManager2CS;
    NvFlowComputeShader *m_blockManager2CS_big;
    NvFlowComputeShader *m_clearTexture3dCS_r;
    NvFlowComputeShader *m_clearTexture3dCS_rgba;
    NvFlowComputeShader *m_sparseFadeCS;
    NvFlowConstantBuffer *m_constantBuffer;
    SparseFadeField m_fadeField;
    unsigned int m_state;
    bool m_gridLocationDirty;
    NvFlowFloat3 m_gridHalfSize;
    NvFlowFloat3 m_gridLocation;
    NvFlowFloat3 m_oldGridLocation;
    NvFlowInt3 m_mapIdxReadOffset;
    uint64_t m_mappingVersion;
    VectorCached<BlockManagerImpl::PerLayer, 16> m_layers;
    VectorCached<BlockManagerImpl::FieldFadePerLayer, 16> m_fieldFadeLayers;

    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()
    uint64_t getGPUBytesUsed() override;

    BlockManagerImpl(NvFlowContext *ctx, const BlockMangerDesc *desc);
}

}  // namespace NvFlow

#endif /* BLOCKMANAGER_H */
