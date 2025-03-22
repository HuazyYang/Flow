#include "SparseTexturePool.h"
#include "ClientHelper.h"
#include "SparseTextureMemoryPool.h"
#include "NvFlowContextImpl.h"
#include <nvflow/NvFlowShader.h>
#include "SparseMapping.h"

namespace NvFlow {

struct SparseTexturePoolLayerMapping {
    NvFlowTexture3D *blockTable;
    NvFlowTexture3D *blockTable1D;
    NvFlowBuffer *blockList;
    NvFlowBuffer *atomicBuf;
    uint32_t blockListSize;
};

struct SparseTexturePoolLayer {
    SparseTextureMemoryMapping *m_memoryMapping;
    SparseTexturePoolLayerMapping m_mapping[2];
    bool layerEnabled;
    bool layerEnabledOld;
};

struct SparseTextureLayer {
    NvFlowResourceRW *m_resourcePoint;
    NvFlowResourceRW *m_resourceLinear;
};

struct SparseTexture {
    uint32_t m_refCount = 0;
    uint32_t m_numLayers = 0;
    uint32_t m_mappingIdx = 0;
    SparseTextureMemoryHandle m_textureMemoryPoint = {};
    SparseTextureMemoryHandle m_textureMemoryLinear = {};
    bool pointDirty = 0;
    bool linearDirty = 0;
    bool pointReadActive = 0;
    bool linearReadActive = 0;
};

struct SparseMappingUpdateShaderParams {
    NvFlowUint4 frameTableDim;
    NvFlowUint4 layerIdx;
};

struct SparseTexturePoolFrameTable {
    NvFlowTexture3D *frameTable;
};

struct SparseLayeredBlockList {
    VectorCached<NvFlowUint2, 1> listCPU;
};

struct SparseTexturePoolImpl : Object, SparseTexturePoolInternal {
    enum Status {
        eStatusPushMapping = 0,
        eStatusWaitForDownload = 1,
        eStatusUpdateMapping = 2
    };

    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    SparseTextureHandle acquireTexture(NvFlowContext *context) override;
    SparseTextureHandle acquireTextureNoAllocate(NvFlowContext *context) override;
    uint32_t addRefTexture(SparseTextureHandle handle) override;
    uint32_t releaseTexture(SparseTextureHandle handle) override;
    SparseReadPointHandle readPointHandle(NvFlowContext *context,
                                          SparseTextureHandle handle) override;
    SparseReadPointLayeredView readPointLayeredView(SparseReadPointHandle handle) override;
    SparseReadPointLayerView readPointLayerView(SparseReadPointHandle handle,
                                                uint32_t layerIdx) override;
    void readPointHandleRelease(NvFlowContext *context,
                                SparseTextureHandle handle) override;
    SparseReadLinearHandle readLinearHandle(NvFlowContext *context,
                                            SparseTextureHandle handle) override;
    SparseReadLinearLayeredView readLinearLayeredView(
        SparseReadLinearHandle handle) override;
    SparseReadLinearLayerView readLinearLayerView(SparseReadLinearHandle handle,
                                                  uint32_t layerIdx) override;
    void readLinearHandleRelease(NvFlowContext *context,
                                 SparseTextureHandle handle) override;

    SparseWritePointHandle writePointHandle(NvFlowContext *context,
                                            SparseTextureHandle handle) override;
    SparseWritePointLayeredView writePointLayeredView(
        SparseWritePointHandle handle) override;
    SparseWritePointLayerView writePointLayerView(SparseWritePointHandle handle,
                                                  uint32_t layerIdx) override;
    SparseWriteLinearHandle writeLinearHandle(NvFlowContext *context,
                                              SparseTextureHandle handle) override;
    SparseWriteLinearLayeredView writeLinearLayeredView(
        SparseWriteLinearHandle handle) override;
    SparseWriteLinearLayerView writeLinearLayerView(SparseWriteLinearHandle handle,
                                                    uint32_t layerIdx) override;
    SparseTexturePoolConfig getConfig() override;
    SparseTexturePoolDesc getDesc() override;

    NvFlowResult commitMapping(NvFlowContext *context, uint64_t version) override;

    // Details
    SparseTexturePoolImpl(NvFlowContext *context, const SparseTexturePoolDesc *desc);
    ~SparseTexturePoolImpl();

    void syncLayers(NvFlowContext *context, uint32_t numLayers);

    SparseTexture *handleToPointer(SparseTextureHandle handle);

    uint32_t handleToIndex(SparseTextureHandle handle);

    bool canCommitMapping(NvFlowContext *context, uint64_t version);

    SparseBlockMappingHandle genBlockMappingHandle(SparseTexturePoolLayerMapping *mapping);

    SparseLayeredBlockListView getLayeredBlockListView(SparseTexture *tex);

    void getTextureMemory(NvFlowContext *context, SparseTexture *tex, uint32_t texIdx,
                          SparseTextureMemoryHandle *texMem,
                          SparseTextureMemoryHandle *texMemOther);

    void getTextureMemoryNewMapping(NvFlowContext *context, SparseTexture *tex,
                                    uint32_t texIdx, SparseTextureMemoryHandle *texMem,
                                    SparseTextureMemoryHandle *texMemOther);

    void getTextureMemoryPreserveMapping(NvFlowContext *context, SparseTexture *tex,
                                         uint32_t texIdx, SparseTextureMemoryHandle *texMem,
                                         SparseTextureMemoryHandle *texMemOther);

    void pushMapping(NvFlowContext *context, uint64_t version, SparseMapping *mappingIn);

    void releaseTextureMemory(SparseTextureHandle handle, bool point, bool linear);

    SparseLayeredBlockListView getLayeredBlockList(SparseTexture *tex);

    void updateMapping(NvFlowContext *context, uint32_t maxUpdateTextures);

    SparseTextureMemoryPoolInternal *m_memoryPool = nullptr;
    SparseTexturePoolDesc m_desc = {};
    SparseTextureMemoryPoolDesc m_memDesc = {};
    SparseTextureMemoryBlockConfig m_blockConfig = {};
    NvFlowDim m_gridDim = make_dim(0);
    uint32_t m_maxBlocks = 0;
    uint32_t m_numLayers = 0;
    NvFlowConstantBuffer *m_constantBuffer = 0;
    NvFlowComputeShader *m_updateLinearCS = 0;
    NvFlowComputeShader *m_updatePointCS = 0;
    NvFlowComputeShader *m_sparseClearCS = 0;
    NvFlowComputeShader *m_sparseClear2CS = 0;
    NvFlowComputeShader *m_sparseScaleCS = 0;
    NvFlowComputeShader *m_sparseDeallocateCS = 0;
    NvFlowComputeShader *m_sparseFreeListCS = 0;
    NvFlowComputeShader *m_sparseAllocateCS = 0;
    NvFlowComputeShader *m_sparseBlockListCS = 0;
    NvFlowShaderPointParams m_pointParams = {};
    NvFlowShaderLinearParams m_linearParams = {};
    NvFlowShaderLinearParams m_downsampleParams = {};
    SparseTexturePoolImpl::Status m_status = eStatusPushMapping;
    // padding
    uint64_t m_updateMappingVersion = 0;
    uint64_t m_committedVersion = 0;
    uint32_t m_mappingFrontIdx = 0;
    uint32_t m_mappingBackIdx = 1;
    SparseTexturePoolFrameTable m_frameTable = {};
    VectorCached<SparseTexture, 8> m_texture;
    VectorCached<SparseTexturePoolLayer, 8> m_texturePoolLayer;
    VectorCached2D<SparseTextureLayer, 8, 8> m_textureLayer;
    SparseLayeredBlockList m_layeredBlockList[2];
};

#include "updateLinearCS.hlsl.h"
#include "updatePointCS.hlsl.h"
#include "sparseClearCS.hlsl.h"
#include "sparseClear2CS.hlsl.h"
#include "sparseScaleCS.hlsl.h"
#include "sparseDeallocateCS.hlsl.h"
#include "sparseFreeListCS.hlsl.h"
#include "sparseAllocateCS.hlsl.h"
#include "sparseBlockListCS.hlsl.h"

SparseWriteLinearHandle SparseTexturePoolImpl::writeLinearHandle(
    NvFlowContext *context, SparseTextureHandle handle) {
    SparseWriteLinearHandle result = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        uint32_t texIdx = handleToIndex(handle);
        getTextureMemoryNewMapping(context, tex, texIdx, &tex->m_textureMemoryLinear,
                                   &tex->m_textureMemoryPoint);
        tex->pointDirty = 1;
        tex->linearDirty = 0;
        tex->pointReadActive = 0;
        tex->linearReadActive = 0;
    }

    result.pool = this;
    result.uid = handle.uid;
    result.numLayers = tex->m_numLayers;
    return result;
}

SparseWriteLinearLayeredView SparseTexturePoolImpl::writeLinearLayeredView(
    SparseWriteLinearHandle handleIn) {
    SparseTextureHandle handle = {handleIn.pool, handleIn.uid};
    SparseWriteLinearLayeredView layeredView = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        layeredView.params = m_linearParams;
        layeredView.layeredBlockList = getLayeredBlockList(tex);
    }
    return layeredView;
}

SparseWriteLinearLayerView SparseTexturePoolImpl::writeLinearLayerView(
    SparseWriteLinearHandle handleIn, uint32_t layerIdx) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseWriteLinearLayerView layerView = {};
    auto tex = handleToPointer(handle);
    if (tex && layerIdx < tex->m_numLayers) {
        uint32_t texIdx = handleToIndex(handle);
        auto &texLayer = m_textureLayer[texIdx][layerIdx];
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];
        layerView.data = texLayer.m_resourceLinear;
        layerView.mapping =
            genBlockMappingHandle(&texPoolLayer.m_mapping[tex->m_mappingIdx]);
    }

    return layerView;
}

SparseTexturePoolConfig SparseTexturePoolImpl::getConfig() {
    SparseTexturePoolConfig config;
    config.maxBlocks = m_maxBlocks;
    config.gridDim = m_gridDim;
    config.poolGridDim = m_memDesc.poolGridDim;
    config.virtualDim = m_desc.virtualDim;
    return config;
}

SparseTexturePoolDesc SparseTexturePoolImpl::getDesc() {
    return m_desc;
}

SparseTexturePoolImpl::SparseTexturePoolImpl(NvFlowContext *context,
                                             const SparseTexturePoolDesc *desc) {
    m_desc = *desc;
    m_memoryPool = implCast<SparseTextureMemoryPoolInternal>(m_desc.memoryPool);
    m_memoryPool->addRef();
    m_memDesc = m_memoryPool->getDesc();
    m_blockConfig = m_memoryPool->getBlockConfig();
    m_gridDim = m_desc.virtualDim / m_blockConfig.blockDim;
    m_maxBlocks =
        m_memDesc.poolGridDim.z * m_memDesc.poolGridDim.y * m_memDesc.poolGridDim.x;

    m_layeredBlockList[0].listCPU.reserve(m_maxBlocks);
    m_layeredBlockList[1].listCPU.reserve(m_maxBlocks);

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(NvFlowShaderLinearParams);
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

    m_updateLinearCS = createShader(NVFLOW_CREATE_SHADER_ARGS(updateLinearCS));
    m_updatePointCS = createShader(NVFLOW_CREATE_SHADER_ARGS(updatePointCS));
    m_sparseClearCS = createShader(NVFLOW_CREATE_SHADER_ARGS0(sparseClearCS), L"sparseMappingClearCS");
    m_sparseClear2CS = createShader(NVFLOW_CREATE_SHADER_ARGS0(sparseClear2CS), L"sparseMappingClear2CS");
    m_sparseScaleCS = createShader(NVFLOW_CREATE_SHADER_ARGS0(sparseScaleCS), L"sparseMappingScaleCS");
    m_sparseDeallocateCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseDeallocateCS));
    m_sparseFreeListCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseFreeListCS));
    m_sparseAllocateCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseAllocateCS));
    m_sparseBlockListCS = createShader(NVFLOW_CREATE_SHADER_ARGS(sparseBlockListCS));

    auto &config = m_blockConfig;
    NvFlowDim texDim = config.enableVTR ? m_desc.virtualDim : config.poolDim;

    auto make_uint4 = [](NvFlowDim dim) {
        return NvFlowUint4{dim.x, dim.y, dim.z, dim.z * dim.y * dim.x};
    };

    auto make_float4 = [](NvFlowDim dim) {
        return NvFlowFloat4{float(dim.x), float(dim.y), float(dim.z),
                            float(dim.z * dim.y * dim.x)};
    };

    auto make_float4_inv = [](NvFlowDim dim) {
        return NvFlowFloat4{1.f / dim.x, 1.f / dim.y, 1.f / dim.z,
                            1.f / (dim.z * dim.y * dim.x)};
    };

    NvFlowUint4 isVTR = NvFlow::make_uint4(config.enableVTR, 0, 0, 0);
    NvFlowUint4 blockDim = make_uint4(config.blockDim);
    NvFlowUint4 blockDimBits;
    blockDimBits.x = log2ui(blockDim.x);
    blockDimBits.y = log2ui(blockDim.y);
    blockDimBits.z = log2ui(blockDim.z);
    blockDimBits.w = log2ui(blockDim.w);

    NvFlowUint4 poolGridDim = NvFlow::make_uint4(config.poolGridDim, 1);
    NvFlowUint4 gridDim = NvFlow::make_uint4(m_gridDim, 1);
    NvFlowFloat4 blockDimInv = make_float4_inv(config.blockDim);
    NvFlowUint4 linearBlockDim = make_uint4(config.linearBlockDim);
    NvFlowUint4 linearBlockOffset = NvFlow::make_uint4(config.linearBlockOffset, 0);
    NvFlowFloat4 dimInv = make_float4_inv(texDim);
    NvFlowFloat4 vdim = make_float4(m_desc.virtualDim);
    NvFlowFloat4 vdimInv = make_float4_inv(m_desc.virtualDim);

    m_pointParams.isVTR = isVTR;
    m_pointParams.blockDim = blockDim;
    m_pointParams.blockDimBits = blockDimBits;
    m_pointParams.poolGridDim = poolGridDim;
    m_pointParams.gridDim = gridDim;

    m_linearParams.isVTR = isVTR;
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

    m_downsampleParams.isVTR = isVTR;
    m_downsampleParams.blockDim = blockDim;
    m_downsampleParams.blockDimBits = blockDimBits;
    m_downsampleParams.poolGridDim = poolGridDim;
    m_downsampleParams.gridDim = gridDim;
    m_downsampleParams.blockDimInv = blockDimInv;
    m_downsampleParams.linearBlockDim = blockDim;
    m_downsampleParams.linearBlockOffset = NvFlow::make_uint4(0);
    m_downsampleParams.dimInv = dimInv;
    m_downsampleParams.vdim = vdim;
    m_downsampleParams.vdimInv = vdimInv;

    NvFlowTexture3DDesc frameTableDesc = {};
    frameTableDesc.format = eNvFlowFormat_r32g32_uint;
    frameTableDesc.dim = m_memDesc.poolGridDim;
    frameTableDesc.uploadAccess = 0;
    frameTableDesc.downloadAccess = 1;
    m_frameTable.frameTable = NvFlowCreateTexture3D(context, &frameTableDesc);

    NvFlowTexture3DDesc clearTexDesc;
    NvFlowTexture3DGetDesc(m_frameTable.frameTable, &clearTexDesc);

    NvFlowDispatchParams dispatchParams = {};
    dispatchParams.shader = m_sparseClear2CS;
    dispatchParams.gridDim = (clearTexDesc.dim + 7) >> 3;
    dispatchParams.rootConstantBuffer = nullptr;
    dispatchParams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_frameTable.frameTable);
    NvFlowContextDispatch(context, &dispatchParams);

    SparseMappingHandle mapped = m_desc.initialMapping->mapMask(context);
    syncLayers(context, mapped.numLayers);
    m_desc.initialMapping->unmapMask(context);
}

SparseTexturePoolImpl::~SparseTexturePoolImpl() {
    for (uint32_t texIdx = 0; texIdx < m_textureLayer.sizeY(); ++texIdx) {
        for (uint32_t layerIdx = 0; layerIdx < m_textureLayer.sizeX(); ++layerIdx) {
            auto &data = m_textureLayer[texIdx][layerIdx];
            data.m_resourceLinear = 0;
            data.m_resourcePoint = 0;
        }
    }

    for (auto &texPoolLayer : m_texturePoolLayer) {
        SafeRelease(texPoolLayer.m_memoryMapping);
        for (auto &layerMapping : texPoolLayer.m_mapping) {
            SafeRelease(layerMapping.blockTable);
            SafeRelease(layerMapping.blockTable1D);
            SafeRelease(layerMapping.blockList);
            SafeRelease(layerMapping.atomicBuf);
        }
    }

    for (auto &tex : m_texture) {
        m_memoryPool->releaseTextureMemory({tex.m_textureMemoryPoint.uid});
        m_memoryPool->releaseTextureMemory({tex.m_textureMemoryLinear.uid});
        tex.m_textureMemoryPoint.uid = 0;
        tex.m_textureMemoryLinear.uid = 0;
        tex.m_mappingIdx = 0;
        tex.m_numLayers = 0;
        tex.m_refCount = 0;
    }

    SafeRelease(m_frameTable.frameTable);
    SafeRelease(m_memoryPool);

    SafeRelease(m_constantBuffer);
    SafeRelease(m_updateLinearCS);
    SafeRelease(m_updatePointCS);
    SafeRelease(m_sparseClearCS);
    SafeRelease(m_sparseClear2CS);
    SafeRelease(m_sparseScaleCS);
    SafeRelease(m_sparseDeallocateCS);
    SafeRelease(m_sparseFreeListCS);
    SafeRelease(m_sparseAllocateCS);
    SafeRelease(m_sparseBlockListCS);
}

void SparseTexturePoolImpl::syncLayers(NvFlowContext *context, uint32_t numLayers) {
    if (numLayers > m_numLayers) {
        m_numLayers = numLayers;
        while (m_texturePoolLayer.size() < numLayers) {
            uint32_t allocIdx = m_texturePoolLayer.allocateBack();
            auto &texPoolLayer = m_texturePoolLayer[allocIdx];
            ZeroMemory(&texPoolLayer, sizeof(texPoolLayer));

            SparseTextureMemoryMappingDesc memMappingDesc = {};
            memMappingDesc.virtualDim = m_desc.virtualDim;

            NvFlowTexture3DDesc blockTableDesc = {};
            blockTableDesc.format = eNvFlowFormat_r32_uint;
            blockTableDesc.dim = m_gridDim;
            blockTableDesc.uploadAccess = 0;
            blockTableDesc.downloadAccess = 0;

            NvFlowTexture3DDesc blockTable1Ddesc = {};
            blockTable1Ddesc.format = eNvFlowFormat_r32_uint;
            blockTable1Ddesc.dim = m_gridDim;
            blockTable1Ddesc.uploadAccess = 0;
            blockTable1Ddesc.downloadAccess = 1;

            NvFlowBufferDesc blockListDesc = {};
            blockListDesc.format = eNvFlowFormat_r32_uint;
            blockListDesc.dim = m_maxBlocks;
            blockListDesc.uploadAccess = 0;
            blockListDesc.downloadAccess = 0;

            NvFlowBufferDesc atomicBufDesc = {};
            atomicBufDesc.format = eNvFlowFormat_r32_uint;
            atomicBufDesc.dim = 64;
            atomicBufDesc.uploadAccess = 1;
            atomicBufDesc.downloadAccess = 1;

            texPoolLayer.m_memoryMapping =
                m_memoryPool->createMemoryMapping(context, &memMappingDesc);

            for (int idx = 0; idx < 2; ++idx) {
                auto &layerMapping = texPoolLayer.m_mapping[idx];
                layerMapping.blockTable = NvFlowCreateTexture3D(context, &blockTableDesc);

                layerMapping.blockTable1D =
                    NvFlowCreateTexture3D(context, &blockTable1Ddesc);

                layerMapping.blockList = NvFlowCreateBuffer(context, &blockListDesc);

                layerMapping.atomicBuf = NvFlowCreateBuffer(context, &atomicBufDesc);

                layerMapping.blockListSize = 0;

                NvFlowTexture3DDesc clearTexDesc;
                NvFlowTexture3DGetDesc(layerMapping.blockTable, &clearTexDesc);
                NvFlowDim gridDim = (clearTexDesc.dim + 7) / 8;

                NvFlowDispatchParams dispatchParams = {};
                dispatchParams.shader = m_sparseClearCS;
                dispatchParams.gridDim = gridDim;
                dispatchParams.rootConstantBuffer = 0;
                dispatchParams.readWrite[0] =
                    NvFlowTexture3DGetResourceRW(layerMapping.blockTable);
                NvFlowContextDispatch(context, &dispatchParams);
            }
        }

        while (m_textureLayer.sizeX() < numLayers) {
            uint32_t layerIdx = m_textureLayer.allocateBackX();
            for (uint32_t texIdx = 0; texIdx < m_textureLayer.sizeY(); ++texIdx) {
                auto &texLayer = m_textureLayer[texIdx][layerIdx];
                texLayer.m_resourcePoint = nullptr;
                texLayer.m_resourceLinear = nullptr;
            }
        }
    }
}

SparseTexture *SparseTexturePoolImpl::handleToPointer(SparseTextureHandle handle) {
    if (handle.uid > 0 && handle.uid <= m_texture.size()) return &m_texture[handle.uid - 1];
    return nullptr;
}

uint32_t SparseTexturePoolImpl::handleToIndex(SparseTextureHandle handle) {
    return handle.uid - 1;
}

bool SparseTexturePoolImpl::canCommitMapping(NvFlowContext *context, uint64_t version) {
    if (m_status != eStatusUpdateMapping) return false;

    bool isComplete = 1;
    for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];
        if (texPoolLayer.layerEnabled || texPoolLayer.layerEnabledOld) {
            isComplete &= texPoolLayer.m_memoryMapping->canCommitMapping(context, version);
        }
    }

    return isComplete;
}

NvFlowResult SparseTexturePoolImpl::commitMapping(NvFlowContext *context,
                                                  uint64_t version) {
    if (!canCommitMapping(context, version)) return eNvFlowFail;

    for (int32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];
        if (texPoolLayer.layerEnabled || texPoolLayer.layerEnabledOld)
            texPoolLayer.m_memoryMapping->commitMapping(context, version);
    }

    swap(m_mappingBackIdx, m_mappingFrontIdx);
    m_status = eStatusPushMapping;
    m_committedVersion = version;
    return eNvFlowSuccess;
}

SparseLayeredBlockListView SparseTexturePoolImpl::getLayeredBlockListView(
    SparseTexture *tex) {
    auto &layeredBlockList = m_layeredBlockList[tex->m_mappingIdx].listCPU;
    SparseLayeredBlockListView result;
    result.dataCPU = layeredBlockList.data();
    result.numBlocks = layeredBlockList.size();
    return result;
}

void SparseTexturePoolImpl::getTextureMemory(NvFlowContext *context, SparseTexture *tex,
                                             uint32_t texIdx,
                                             SparseTextureMemoryHandle *texMem,
                                             SparseTextureMemoryHandle *texMemOther) {
    if (!texMem->uid) {
        *texMem = m_memoryPool->acquireTextureMemory(context);
        if (m_memDesc.enableVTR) {
            *texMemOther = *texMem;
            m_memoryPool->addRefTextureMemory(*texMemOther);
        }
    }
}

void SparseTexturePoolImpl::getTextureMemoryNewMapping(
    NvFlowContext *context, SparseTexture *tex, uint32_t texIdx,
    SparseTextureMemoryHandle *texMem, SparseTextureMemoryHandle *texMemOther) {
    getTextureMemory(context, tex, texIdx, texMem, texMemOther);
    tex->m_mappingIdx = m_mappingFrontIdx;
    for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];
        auto &texLayer = m_textureLayer[texIdx][layerIdx];
        texLayer.m_resourcePoint =
            texPoolLayer.m_memoryMapping->getResourceRW(context, tex->m_textureMemoryPoint);
        texLayer.m_resourceLinear = texPoolLayer.m_memoryMapping->getResourceRW(
            context, tex->m_textureMemoryLinear);
    }
    tex->m_numLayers = m_numLayers;
}

void SparseTexturePoolImpl::getTextureMemoryPreserveMapping(
    NvFlowContext *context, SparseTexture *tex, uint32_t texIdx,
    SparseTextureMemoryHandle *texMem, SparseTextureMemoryHandle *texMemOther) {
    if (!texMem->uid) {
        getTextureMemory(context, tex, texIdx, texMem, texMemOther);
        for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
            auto &texPoolLayer = m_texturePoolLayer[layerIdx];
            auto &texLayer = m_textureLayer[texIdx][layerIdx];
            texLayer.m_resourcePoint = texPoolLayer.m_memoryMapping->getResourceRW(
                context, tex->m_textureMemoryPoint);
            texLayer.m_resourceLinear = texPoolLayer.m_memoryMapping->getResourceRW(
                context, tex->m_textureMemoryLinear);
        }
    }
}

void SparseTexturePoolImpl::pushMapping(NvFlowContext *context, uint64_t version,
                                        SparseMapping *mappingIn) {
    auto mapping = implCast<SparseMappingInternal>(mappingIn);
    if (m_status == eStatusPushMapping) {
        auto mapped = mapping->mapMaskScaled(context, m_gridDim);
        syncLayers(context, mapped.numLayers);

        for (uint32_t layerIdx = 0; layerIdx < mapped.numLayers; ++layerIdx) {
            auto mappedLayer = mapping->mapMaskLayerScaled(context, mapped, layerIdx);
            auto &texPoolLayer = m_texturePoolLayer[layerIdx];
            texPoolLayer.layerEnabledOld = texPoolLayer.layerEnabled;
            texPoolLayer.layerEnabled = mappedLayer.enable;
            auto &layerMapping = texPoolLayer.m_mapping[m_mappingBackIdx];
            auto &layerMappingOld = texPoolLayer.m_mapping[m_mappingFrontIdx];
            if (texPoolLayer.layerEnabled || texPoolLayer.layerEnabledOld) {
                auto atomicBuf = layerMapping.atomicBuf;
                uint32_t *atomicData = (uint32_t *)NvFlowBufferMap(context, atomicBuf);
                if (atomicData) {
                    NvFlowBufferDesc atomicBuf_desc;
                    NvFlowBufferGetDesc(atomicBuf, &atomicBuf_desc);
                    ZeroMemory(atomicData, atomicBuf_desc.dim * sizeof(uint32_t));
                    NvFlowBufferUnmap(context, atomicBuf);
                }

                auto mappingUpdataParams =
                    (SparseMappingUpdateShaderParams *)NvFlowConstantBufferMap(
                        context, m_constantBuffer);
                if (mappingUpdataParams) {
                    mappingUpdataParams->frameTableDim =
                        make_uint4(m_memDesc.poolGridDim, 0);
                    mappingUpdataParams->layerIdx = make_uint4(layerIdx, 0, 0, 0);
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                NvFlowDispatchParams dispatchParams = {};
                dispatchParams.shader = m_sparseDeallocateCS;
                dispatchParams.gridDim = (m_gridDim + 7) >> 3;
                dispatchParams.rootConstantBuffer = m_constantBuffer;
                dispatchParams.readOnly[0] = NvFlowTexture3DGetResource(mappedLayer.mask);
                dispatchParams.readOnly[1] =
                    NvFlowTexture3DGetResource(layerMappingOld.blockTable);
                dispatchParams.readWrite[0] =
                    NvFlowTexture3DGetResourceRW(layerMapping.blockTable);
                dispatchParams.readWrite[1] =
                    NvFlowTexture3DGetResourceRW(m_frameTable.frameTable);
                NvFlowContextDispatch(context, &dispatchParams);

                ZeroMemory(&dispatchParams, sizeof(dispatchParams));
                dispatchParams.shader = m_sparseFreeListCS;
                dispatchParams.gridDim = (m_memDesc.poolGridDim + 7) >> 3;
                dispatchParams.rootConstantBuffer = m_constantBuffer;
                dispatchParams.readOnly[0] =
                    NvFlowTexture3DGetResource(m_frameTable.frameTable);
                dispatchParams.readWrite[0] =
                    NvFlowBufferGetResourceRW(layerMapping.blockList);
                dispatchParams.readWrite[1] =
                    NvFlowBufferGetResourceRW(layerMapping.atomicBuf);
                NvFlowContextDispatch(context, &dispatchParams);

                ZeroMemory(&dispatchParams, sizeof(dispatchParams));
                dispatchParams.shader = m_sparseAllocateCS;
                dispatchParams.gridDim = (m_gridDim + 7) >> 3;
                dispatchParams.readOnly[0] = NvFlowTexture3DGetResource(mappedLayer.mask);
                dispatchParams.readOnly[1] =
                    NvFlowBufferGetResource(layerMapping.blockList);
                dispatchParams.readWrite[0] =
                    NvFlowTexture3DGetResourceRW(layerMapping.blockTable);
                dispatchParams.readWrite[1] =
                    NvFlowTexture3DGetResourceRW(m_frameTable.frameTable);
                dispatchParams.readWrite[2] =
                    NvFlowBufferGetResourceRW(layerMapping.atomicBuf);
                dispatchParams.readWrite[3] =
                    NvFlowTexture3DGetResourceRW(layerMapping.blockTable1D);
                NvFlowContextDispatch(context, &dispatchParams);

                ZeroMemory(&dispatchParams, sizeof(dispatchParams));
                dispatchParams.shader = m_sparseBlockListCS;
                dispatchParams.gridDim = (m_memDesc.poolGridDim + 7) >> 3;
                dispatchParams.rootConstantBuffer = m_constantBuffer;
                dispatchParams.readOnly[0] =
                    NvFlowTexture3DGetResource(m_frameTable.frameTable);
                dispatchParams.readWrite[0] =
                    NvFlowBufferGetResourceRW(layerMapping.blockList);
                dispatchParams.readWrite[1] =
                    NvFlowBufferGetResourceRW(layerMapping.atomicBuf);
                NvFlowContextDispatch(context, &dispatchParams);
                NvFlowBufferDownload(context, layerMapping.atomicBuf);
                NvFlowTexture3DDownload(context, layerMapping.blockTable1D);
            } else {
                layerMapping.blockListSize = 0;
            }

            mapping->unmapMaskLayerScaled(context, mapped, layerIdx);
        }

        mapping->unmapMaskScaled(context);

        NvFlowTexture3DDownload(context, m_frameTable.frameTable);
        m_status = eStatusWaitForDownload;
        m_updateMappingVersion = version;
    }
}

void SparseTexturePoolImpl::releaseTextureMemory(SparseTextureHandle handle, bool point,
                                                 bool linear) {
    auto tex = handleToPointer(handle);
    auto texIdx = handleToIndex(handle);
    for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
        auto &texLayer = m_textureLayer[texIdx][layerIdx];

        if (point) texLayer.m_resourcePoint = nullptr;
        if (linear) texLayer.m_resourceLinear = nullptr;

        if (point) m_memoryPool->releaseTextureMemory({tex->m_textureMemoryPoint.uid});
        if (linear) m_memoryPool->releaseTextureMemory({tex->m_textureMemoryLinear.uid});

        if (point) tex->m_textureMemoryPoint.uid = 0;
        if (linear) tex->m_textureMemoryLinear.uid = 0;
    }
}

SparseBlockMappingHandle SparseTexturePoolImpl::genBlockMappingHandle(
    SparseTexturePoolLayerMapping *mapping) {
    SparseBlockMappingHandle blockMapping = {};
    blockMapping.blockTable = NvFlowTexture3DGetResource(mapping->blockTable);
    blockMapping.blockList = NvFlowBufferGetResource(mapping->blockList);
    blockMapping.numBlocks = mapping->blockListSize;
    return blockMapping;
}

SparseLayeredBlockListView SparseTexturePoolImpl::getLayeredBlockList(SparseTexture *tex) {
    auto &layeredBlockList = m_layeredBlockList[tex->m_mappingIdx];
    SparseLayeredBlockListView result = {};
    result.dataCPU = layeredBlockList.listCPU.data();
    result.numBlocks = layeredBlockList.listCPU.size();
    return result;
}

void SparseTexturePoolImpl::updateMapping(NvFlowContext *context,
                                          uint32_t maxUpdateTextures) {
    bool downloadComplete;
    if (m_status == eStatusWaitForDownload) {
        downloadComplete = 1;
        for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
            auto &texPoolLayer = m_texturePoolLayer[layerIdx];
            auto &layerMapping = texPoolLayer.m_mapping[m_mappingBackIdx];
            if (texPoolLayer.layerEnabled || texPoolLayer.layerEnabledOld) {
                auto data = NvFlowTexture3DMapDownload(context, layerMapping.blockTable1D);
                if (data.data)
                    NvFlowTexture3DUnmapDownload(context, layerMapping.blockTable1D);
                else
                    downloadComplete = 0;

                if (NvFlowBufferMapDownload(context, layerMapping.atomicBuf))
                    NvFlowBufferUnmapDownload(context, layerMapping.atomicBuf);
                else
                    downloadComplete = 0;
            }
        }

        {
            auto frameData = NvFlowTexture3DMapDownload(context, m_frameTable.frameTable);
            if (frameData.data)
                NvFlowTexture3DUnmapDownload(context, m_frameTable.frameTable);
            else
                downloadComplete = 0;
        }

        if (downloadComplete) {
            for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
                auto &texPoolLayer = m_texturePoolLayer[layerIdx];
                auto &layerMapping = texPoolLayer.m_mapping[m_mappingBackIdx];
                if (texPoolLayer.layerEnabled || texPoolLayer.layerEnabledOld) {
                    auto blockTable1DData =
                        NvFlowTexture3DMapDownload(context, layerMapping.blockTable1D);
                    if (blockTable1DData.data) {
                        NvFlowTexture3DDesc blockTable1D_desc;
                        NvFlowTexture3DGetDesc(layerMapping.blockTable1D,
                                               &blockTable1D_desc);
                        uint32_t sizeInBytes =
                            blockTable1D_desc.dim.z * blockTable1DData.depthPitch;

                        texPoolLayer.m_memoryMapping->pushMapping(
                            m_updateMappingVersion, (uint32_t *)blockTable1DData.data,
                            blockTable1DData.rowPitch, blockTable1DData.depthPitch,
                            sizeInBytes);

                        NvFlowTexture3DUnmapDownload(context, layerMapping.blockTable1D);
                    }

                    auto atomicData = (uint32_t *)NvFlowBufferMapDownload(
                        context, layerMapping.atomicBuf);
                    if (atomicData) {
                        layerMapping.blockListSize = atomicData[2];
                        NvFlowBufferUnmapDownload(context, layerMapping.atomicBuf);
                    }
                }
            }

            {
                auto frameData =
                    NvFlowTexture3DMapDownload(context, m_frameTable.frameTable);
                if (frameData.data) {
                    auto &layeredBlockList = m_layeredBlockList[m_mappingBackIdx];
                    layeredBlockList.listCPU.reserve(m_maxBlocks);
                    layeredBlockList.listCPU.clear();
                    auto frameTableMapped = (NvFlowUint2 *)frameData.data;
                    NvFlowDim poolGridDim = m_memDesc.poolGridDim;
                    NvFlowUint blockDimDepthStride = poolGridDim.x * poolGridDim.y;
                    NvFlowUint2 val;
                    uint32_t allocIdx;
                    for (uint32_t k = 0; k < poolGridDim.z; ++k)
                        for (uint32_t j = 0; j < poolGridDim.y; ++j)
                            for (uint32_t i = 0; i < poolGridDim.x; ++i) {
                                val = frameTableMapped[i + ((frameData.rowPitch * j +
                                                             frameData.depthPitch * k) >>
                                                            3)];
                                if (val.x) {
                                    allocIdx = layeredBlockList.listCPU.allocateBack();
                                    layeredBlockList.listCPU[allocIdx] = val;
                                }
                            }

                    NvFlowTexture3DUnmapDownload(context, m_frameTable.frameTable);
                }
            }
            m_status = eStatusUpdateMapping;
        }
    }

    if (m_status == eStatusUpdateMapping) {
        for (uint32_t n = 0; n < m_numLayers; ++n) {
            auto &texPoolLayer = m_texturePoolLayer[n];
            if (texPoolLayer.layerEnabled || texPoolLayer.layerEnabledOld)
                texPoolLayer.m_memoryMapping->updateMapping(context, maxUpdateTextures);
        }
    }
}

uint64_t SparseTexturePoolImpl::getGPUBytesUsed() {
    uint64_t totalBytes = 0;
    uint64_t GPUBytesUsed;
    if (m_frameTable.frameTable) {
        GPUBytesUsed = m_frameTable.frameTable->getGPUBytesUsed();
        totalBytes += GPUBytesUsed;
    }
    if (m_memoryPool) {
        GPUBytesUsed = m_memoryPool->getGPUBytesUsed();
        totalBytes += GPUBytesUsed;
    }

    for (auto &texPoolLayer : m_texturePoolLayer) {
        for (auto &mapping : texPoolLayer.m_mapping) {
            if (mapping.blockTable) totalBytes += mapping.blockTable->getGPUBytesUsed();
            if (mapping.blockTable1D) totalBytes += mapping.blockTable->getGPUBytesUsed();
            if (mapping.blockList) totalBytes += mapping.blockList->getGPUBytesUsed();
            if (mapping.atomicBuf) totalBytes += mapping.atomicBuf->getGPUBytesUsed();
        }
    }
    return totalBytes;
}

SparseTextureHandle SparseTexturePoolImpl::acquireTexture(NvFlowContext *context) {
    SparseTextureHandle texExisting = acquireTextureNoAllocate(context);
    if (texExisting.uid) {
        return texExisting;
    } else {
        uint32_t allocIdx = m_texture.allocateBack();
        auto &tex = m_texture[allocIdx];
        tex.m_refCount = 1;
        tex.m_numLayers = m_numLayers;
        while (m_textureLayer.sizeY() < m_texture.size()) {
            uint32_t texIdx = m_textureLayer.allocateBackY();
            for (uint32_t layerIdx = 0; layerIdx < m_textureLayer.sizeX(); ++layerIdx) {
                m_textureLayer[texIdx][layerIdx] = SparseTextureLayer{};
            }
        }

        SparseTextureHandle result;
        result.pool = this;
        result.uid = allocIdx + 1;
        return result;
    }
}

SparseTextureHandle SparseTexturePoolImpl::acquireTextureNoAllocate(
    NvFlowContext *context) {
    SparseTextureHandle result = {};
    for (uint32_t idx = 0;; ++idx) {
        if (idx >= m_texture.size()) {
            result.pool = this;
            result.uid = 0;
            return result;
        }
        auto &tex = m_texture[idx];
        if (!tex.m_refCount) {
            ++tex.m_refCount;
            result.pool = this;
            result.uid = idx + 1;
            break;
        }
    }

    return result;
}

uint32_t SparseTexturePoolImpl::addRefTexture(SparseTextureHandle handle) {
    auto tex = handleToPointer(handle);
    if (!tex) return 0;
    return ++tex->m_refCount;
}

uint32_t SparseTexturePoolImpl::releaseTexture(SparseTextureHandle handle) {
    auto tex = handleToPointer(handle);
    if (!tex) return 0;
    if (!--tex->m_refCount) {
        releaseTextureMemory(handle, 1, 1);
    }
    return tex->m_refCount;
}

SparseReadPointHandle SparseTexturePoolImpl::readPointHandle(NvFlowContext *context,
                                                             SparseTextureHandle handle) {
    SparseReadPointHandle result = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        uint32_t texIdx = handleToIndex(handle);
        getTextureMemoryPreserveMapping(context, tex, texIdx, &tex->m_textureMemoryPoint,
                                        &tex->m_textureMemoryLinear);
        tex->pointReadActive = 1;
        if (tex->pointDirty) {
            if (!m_memDesc.enableVTR) {
                NvFlowShaderPointParams *cbData =
                    (NvFlowShaderPointParams *)NvFlowConstantBufferMap(context,
                                                                       m_constantBuffer);
                memcpy(cbData, &m_linearParams, sizeof(m_linearParams));
                NvFlowConstantBufferUnmap(context, m_constantBuffer);

                NvFlowDispatchParams params = {};
                params.shader = m_updatePointCS;

                for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
                    auto &texPoolLayer = m_texturePoolLayer[layerIdx];
                    auto &texLayer = m_textureLayer[texIdx][layerIdx];
                    auto &mapping = texPoolLayer.m_mapping[tex->m_mappingIdx];

                    NvFlowDim gridDim;
                    gridDim.x =
                        (m_linearParams.blockDim.x * mapping.blockListSize + 7) >> 3;
                    gridDim.y = (m_linearParams.blockDim.y + 7) >> 3;
                    gridDim.z = (m_linearParams.blockDim.z + 7) >> 3;
                    params.gridDim = gridDim;

                    params.rootConstantBuffer = m_constantBuffer;
                    params.readOnly[0] = NvFlowBufferGetResource(mapping.blockList);
                    params.readOnly[1] =
                        NvFlowResourceRWGetResource(texLayer.m_resourceLinear);
                    params.readOnly[2] = NvFlowTexture3DGetResource(mapping.blockTable);
                    params.readWrite[0] = texLayer.m_resourcePoint;
                    NvFlowContextDispatch(context, &params);
                }
            }
            tex->pointDirty = 0;
            if (!tex->linearReadActive) readLinearHandleRelease(context, handle);
        }
    }

    result.pool = this;
    result.uid = handle.uid;
    result.numLayers = tex->m_numLayers;

    return result;
}

SparseReadPointLayeredView SparseTexturePoolImpl::readPointLayeredView(
    SparseReadPointHandle handleIn) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseReadPointLayeredView layeredView = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        layeredView.pointParams = m_pointParams;
        layeredView.downsampleParams = m_downsampleParams;
        layeredView.layeredBlockList = getLayeredBlockList(tex);
    }
    return layeredView;
}

SparseReadPointLayerView SparseTexturePoolImpl::readPointLayerView(
    SparseReadPointHandle handleIn, uint32_t layerIdx) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseReadPointLayerView layerView = {};

    auto tex = handleToPointer(handle);
    if (tex && layerIdx < tex->m_numLayers) {
        uint32_t texIdx = handleToIndex(handle);
        auto &texLayer = m_textureLayer[texIdx][layerIdx];
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];

        layerView.data = NvFlowResourceRWGetResource(texLayer.m_resourcePoint);
        layerView.mapping =
            genBlockMappingHandle(&texPoolLayer.m_mapping[tex->m_mappingIdx]);
    }

    return layerView;
}

void SparseTexturePoolImpl::readPointHandleRelease(NvFlowContext *context,
                                                   SparseTextureHandle handle) {
    auto tex = handleToPointer(handle);
    if (tex && !tex->linearDirty) {
        tex->pointReadActive = 0;
        tex->pointDirty = 1;
        if (!m_blockConfig.enableVTR) releaseTextureMemory(handle, 1, 0);
    }
}

SparseReadLinearHandle SparseTexturePoolImpl::readLinearHandle(NvFlowContext *context,
                                                               SparseTextureHandle handle) {
    SparseReadLinearHandle result = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        uint32_t texIdx = handleToIndex(handle);
        getTextureMemoryPreserveMapping(context, tex, texIdx, &tex->m_textureMemoryLinear,
                                        &tex->m_textureMemoryPoint);
        tex->linearReadActive = 1;
        if (tex->linearDirty) {
            if (!m_memDesc.enableVTR) {
                NvFlowShaderLinearParams *cbData =
                    (NvFlowShaderLinearParams *)NvFlowConstantBufferMap(context,
                                                                        m_constantBuffer);
                memcpy(cbData, &m_linearParams, sizeof(m_linearParams));
                NvFlowConstantBufferUnmap(context, m_constantBuffer);

                NvFlowDispatchParams params = {};
                params.shader = m_updateLinearCS;

                for (uint32_t layerIdx = 0; layerIdx < m_numLayers; ++layerIdx) {
                    auto &texPoolLayer = m_texturePoolLayer[layerIdx];
                    auto &texLayer = m_textureLayer[texIdx][layerIdx];
                    auto &mapping = texPoolLayer.m_mapping[tex->m_mappingIdx];

                    NvFlowDim gridDim;
                    gridDim.x = (m_linearParams.linearBlockDim.w + 127) >> 7;
                    gridDim.y = mapping.blockListSize;
                    gridDim.z = 1;
                    params.gridDim = gridDim;

                    params.rootConstantBuffer = m_constantBuffer;
                    params.readOnly[0] = NvFlowBufferGetResource(mapping.blockList);
                    params.readOnly[1] =
                        NvFlowResourceRWGetResource(texLayer.m_resourcePoint);
                    params.readOnly[2] = NvFlowTexture3DGetResource(mapping.blockTable);
                    params.readWrite[0] = texLayer.m_resourceLinear;
                    NvFlowContextDispatch(context, &params);
                }
            }
            tex->linearDirty = 0;
            if (!tex->pointReadActive) readPointHandleRelease(context, handle);
        }
    }

    result.pool = this;
    result.uid = handle.uid;
    result.numLayers = tex->m_numLayers;
    return result;
}

SparseReadLinearLayeredView SparseTexturePoolImpl::readLinearLayeredView(
    SparseReadLinearHandle handleIn) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseReadLinearLayeredView layeredView = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        layeredView.params = m_linearParams;
        layeredView.layeredBlockList = getLayeredBlockList(tex);
    }
    return layeredView;
}

SparseReadLinearLayerView SparseTexturePoolImpl::readLinearLayerView(
    SparseReadLinearHandle handleIn, uint32_t layerIdx) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseReadLinearLayerView layerView = {};

    auto tex = handleToPointer(handle);
    if (tex && layerIdx < tex->m_numLayers) {
        uint32_t texIdx = handleToIndex(handle);
        auto &texLayer = m_textureLayer[texIdx][layerIdx];
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];

        layerView.data = NvFlowResourceRWGetResource(texLayer.m_resourceLinear);
        layerView.mapping =
            genBlockMappingHandle(&texPoolLayer.m_mapping[tex->m_mappingIdx]);
    }

    return layerView;
}

void SparseTexturePoolImpl::readLinearHandleRelease(NvFlowContext *context,
                                                    SparseTextureHandle handle) {
    auto tex = handleToPointer(handle);
    if (tex && !tex->pointDirty) {
        tex->linearReadActive = 0;
        tex->linearDirty = 1;
        if (!m_blockConfig.enableVTR) releaseTextureMemory(handle, 0, 1);
    }
}

SparseWritePointHandle SparseTexturePoolImpl::writePointHandle(NvFlowContext *context,
                                                               SparseTextureHandle handle) {
    SparseWritePointHandle result = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        uint32_t texIdx = handleToIndex(handle);
        getTextureMemoryNewMapping(context, tex, texIdx, &tex->m_textureMemoryPoint,
                                   &tex->m_textureMemoryLinear);
        tex->pointDirty = 0;
        tex->linearDirty = 1;
        tex->pointReadActive = 0;
        tex->linearReadActive = 0;
    }

    result.pool = this;
    result.uid = handle.uid;
    result.numLayers = tex->m_numLayers;
    return result;
}

SparseWritePointLayeredView SparseTexturePoolImpl::writePointLayeredView(
    SparseWritePointHandle handleIn) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseWritePointLayeredView layeredView = {};
    auto tex = handleToPointer(handle);
    if (tex) {
        layeredView.params = m_pointParams;
        layeredView.layeredBlockList = getLayeredBlockList(tex);
    }
    return layeredView;
}

SparseWritePointLayerView SparseTexturePoolImpl::writePointLayerView(
    SparseWritePointHandle handleIn, uint32_t layerIdx) {
    SparseTextureHandle handle = {this, handleIn.uid};
    SparseWritePointLayerView layerView = {};
    auto tex = handleToPointer(handle);
    if (tex && layerIdx < tex->m_numLayers) {
        uint32_t texIdx = handleToIndex(handle);
        auto &texLayer = m_textureLayer[texIdx][layerIdx];
        auto &texPoolLayer = m_texturePoolLayer[layerIdx];
        layerView.data = texLayer.m_resourcePoint;
        layerView.mapping =
            genBlockMappingHandle(&texPoolLayer.m_mapping[tex->m_mappingIdx]);
    }

    return layerView;
}

SparseReadLinearHandle SparseTextureHandle::readLinearHandle(NvFlowContext *context) {
    return pool->readLinearHandle(context, *this);
}

void SparseTextureHandle::readLinearHandleRelease(NvFlowContext *context) {
    pool->readLinearHandleRelease(context, *this);
}

SparseReadPointHandle SparseTextureHandle::readPointHandle(NvFlowContext *context) {
    return pool->readPointHandle(context, *this);
}

void SparseTextureHandle::readPointHandleRelease(NvFlowContext *context) {
    pool->readPointHandleRelease(context, *this);
}

SparseWriteLinearHandle SparseTextureHandle::writeLinearHandle(NvFlowContext *context) {
    return pool->writeLinearHandle(context, *this);
}

SparseWritePointHandle SparseTextureHandle::writePointHandle(NvFlowContext *context) {
    return pool->writePointHandle(context, *this);
}

void SparseTextureHandle::addRefTexture() {
    pool->addRefTexture(*this);
}

void SparseTextureHandle::releaseTexture() {
    pool->releaseTexture(*this);
}

SparseReadLinearLayeredView SparseReadLinearHandle::layeredView() {
    return pool->readLinearLayeredView(*this);
}

SparseReadLinearLayerView SparseReadLinearHandle::layerView(uint32_t layerIdx) {
    return pool->readLinearLayerView(*this, layerIdx);
}

void SparseTextureFront::init(NvFlowContext *context, SparseTexturePool *poolIn) {
    pool = poolIn;
    front = pool->acquireTexture(context);
}

SparseTextureHandle SparseTextureFront::acquireTexture(NvFlowContext *context) {
    return pool->acquireTexture(context);
}

SparseTexturePoolDesc SparseTextureFront::getDesc() {
    return pool->getDesc();
}

SparseTexturePoolConfig SparseTextureFront::getConfig() {
    return pool->getConfig();
}

void SparseTextureFront::swap(const SparseTextureHandle &newFront) {
    SparseTextureHandle oldFront = front;
    front = newFront;
    pool->releaseTexture(oldFront);
}

SparseWritePointLayeredView SparseWritePointHandle::layeredView() {
    return pool->writePointLayeredView(*this);
}

SparseWritePointLayerView SparseWritePointHandle::layerView(uint32_t layerIdx) {
    return pool->writePointLayerView(*this, layerIdx);
}

SparseReadPointLayeredView SparseReadPointHandle::layeredView() {
    return pool->readPointLayeredView(*this);
}

SparseReadPointLayerView SparseReadPointHandle::layerView(uint32_t layerIdx) {
    return pool->readPointLayerView(*this, layerIdx);
}

SparseWriteLinearLayeredView SparseWriteLinearHandle::layeredView() {
    return pool->writeLinearLayeredView(*this);
}

SparseWriteLinearLayerView SparseWriteLinearHandle::layerView(uint32_t layerIdx) {
    return pool->writeLinearLayerView(*this, layerIdx);
}

SparseTexturePool *createSparseTexturePool(NvFlowContext *context,
                                           const SparseTexturePoolDesc *desc) {
    return new SparseTexturePoolImpl(context, desc);
}

}  // namespace NvFlow