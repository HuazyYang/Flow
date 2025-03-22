#include "SparseTextureMemoryPool.h"
#include "ClientHelper.h"
#include "NvFlowContextImpl.h"

namespace NvFlow {

namespace {
template <typename T>
void sparse_swap(T &left, T &right) {
    swap(left, right);
}
}  // namespace

struct SparseTextureMemory {
    uint32_t m_refCount;
    NvFlowTexture3D *m_texture;
    NvFlowHeapSparse *m_heap;
};

struct SparseTextureMemoryPoolImpl : Object, SparseTextureMemoryPoolInternal {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    SparseTextureMemoryPoolDesc getDesc() override;
    SparseTextureMemoryPoolStatus getStatus() override;

    SparseTextureMemoryMapping *createMemoryMapping(
        NvFlowContext *context, const SparseTextureMemoryMappingDesc *desc) override;

    SparseTextureMemoryHandle acquireTextureMemory(NvFlowContext *context) override;
    SparseTextureMemoryHandle acquireTextureMemoryNoAllocate(
        NvFlowContext *context) override;

    uint32_t addRefTextureMemory(SparseTextureMemoryHandle handle) override;
    uint32_t releaseTextureMemory(SparseTextureMemoryHandle handle) override;

    SparseTextureMemoryBlockConfig getBlockConfig() override;

    // Details
    SparseTextureMemoryPoolImpl(NvFlowContext *context,
                                const SparseTextureMemoryPoolDesc *desc);

    ~SparseTextureMemoryPoolImpl();

    SparseTextureMemory *handleToPointer(SparseTextureMemoryHandle handle);

    int64_t handleToIndex(SparseTextureMemoryHandle handle);

    SparseTextureMemoryPoolDesc m_desc;
    VectorCached<SparseTextureMemory, 16> m_textureMemory;
};

SparseTextureMemoryMapping *createSparseTextureMemoryMapping(
    NvFlowContext *context, struct SparseTextureMemoryPoolImpl *memoryPool,
    const SparseTextureMemoryMappingDesc *desc);

uint64_t SparseTextureMemoryPoolImpl::getGPUBytesUsed() {
    uint64_t totalBytes = 0;
    for (uint32_t idx = 0; idx < m_textureMemory.size(); ++idx) {
        auto &textureMemory = m_textureMemory[idx];
        if (textureMemory.m_texture) {
            auto texObj = NvFlowTexture3DGetContextObject(textureMemory.m_texture);
            totalBytes += NvFlowContextObjectGetGPUBytesUsed(texObj);
        } else if (textureMemory.m_heap) {
            auto heapObj = NvFlowHeapSparseGetContextObject(textureMemory.m_heap);
            totalBytes += NvFlowContextObjectGetGPUBytesUsed(heapObj);
        }
    }
    return totalBytes;
}

SparseTextureMemoryPoolDesc SparseTextureMemoryPoolImpl::getDesc() {
    return m_desc;
}

SparseTextureMemoryPoolStatus SparseTextureMemoryPoolImpl::getStatus() {
    SparseTextureMemoryPoolStatus status = {};
    status.textureMemoryCount = m_textureMemory.size();
    return status;
}

SparseTextureMemoryMapping *SparseTextureMemoryPoolImpl::createMemoryMapping(
    NvFlowContext *context, const SparseTextureMemoryMappingDesc *desc) {
    return createSparseTextureMemoryMapping(context, this, desc);
}

SparseTextureMemoryHandle SparseTextureMemoryPoolImpl::acquireTextureMemory(
    NvFlowContext *context) {
    auto texMemExisting = acquireTextureMemoryNoAllocate(context);
    if (texMemExisting.uid) {
        return texMemExisting;
    } else {
        uint32_t allocIdx = m_textureMemory.allocateBack();
        auto &texMem = m_textureMemory[allocIdx];
        texMem.m_refCount = 1;
        texMem.m_texture = nullptr;
        texMem.m_heap = nullptr;

        if (m_desc.enableVTR) {
            auto blockConfig = getBlockConfig();
            uint32_t heapSize = blockConfig.formatSizeInBytes * blockConfig.poolDim.x *
                                blockConfig.poolDim.y * blockConfig.poolDim.z;
            NvFlowHeapSparseDesc heapDesc = {};
            heapDesc.sizeInBytes = heapSize;
            texMem.m_heap = NvFlowCreateHeapSparse(context, &heapDesc);
        } else {
            auto blockConfig = getBlockConfig();
            NvFlowTexture3DDesc poolDesc = {};
            poolDesc.format = m_desc.format;
            poolDesc.dim = blockConfig.poolDim;
            poolDesc.uploadAccess = 0;
            poolDesc.downloadAccess = 0;
            texMem.m_texture = NvFlowCreateTexture3D(context, &poolDesc);
        }
        return SparseTextureMemoryHandle{allocIdx + 1};
    }
}

SparseTextureMemoryHandle SparseTextureMemoryPoolImpl::acquireTextureMemoryNoAllocate(
    NvFlowContext *context) {
    for (uint32_t idx = 0; idx < m_textureMemory.size(); ++idx) {
        auto &texMem = m_textureMemory[idx];
        if (!texMem.m_refCount) {
            ++texMem.m_refCount;
            return SparseTextureMemoryHandle{idx + 1};
        }
    }

    return SparseTextureMemoryHandle{0};
}

uint32_t SparseTextureMemoryPoolImpl::addRefTextureMemory(
    SparseTextureMemoryHandle handle) {
    auto texMem = handleToPointer(handle);
    if (!texMem) return 0;
    return ++texMem->m_refCount;
}

uint32_t SparseTextureMemoryPoolImpl::releaseTextureMemory(
    SparseTextureMemoryHandle handle) {
    auto texMem = handleToPointer(handle);
    if (!texMem) return 0;
    return --texMem->m_refCount;
}

SparseTextureMemoryBlockConfig SparseTextureMemoryPoolImpl::getBlockConfig() {
    SparseTextureMemoryBlockConfig blockConfig = {};

    uint32_t blockInflate = m_desc.enableVTR ? 0 : 2;
    uint32_t blockOffset = m_desc.enableVTR ? 0 : 1;
    blockConfig.blockDim = getTileDim(m_desc.format);
    blockConfig.linearBlockDim = blockInflate + blockConfig.blockDim;
    blockConfig.linearBlockOffset = make_dim(blockOffset);
    blockConfig.poolGridDim = m_desc.poolGridDim;
    blockConfig.poolDim = blockConfig.linearBlockDim * blockConfig.poolGridDim;
    blockConfig.format = m_desc.format;
    blockConfig.formatSizeInBytes = getFormatSizeInBytes(m_desc.format);
    blockConfig.enableVTR = m_desc.enableVTR;
    return blockConfig;
}

SparseTextureMemoryPoolImpl::SparseTextureMemoryPoolImpl(
    NvFlowContext *context, const SparseTextureMemoryPoolDesc *desc)
    : m_desc{}, m_textureMemory{} {
    m_desc = *desc;
}

SparseTextureMemoryPoolImpl::~SparseTextureMemoryPoolImpl() {
    for (auto &textureMemory : m_textureMemory) {
        SafeRelease(textureMemory.m_texture);
        SafeRelease(textureMemory.m_heap);
    }
}

SparseTextureMemory *SparseTextureMemoryPoolImpl::handleToPointer(
    SparseTextureMemoryHandle handle) {
    if (handle.uid > 0 && handle.uid <= m_textureMemory.size()) {
        return &m_textureMemory[handle.uid - 1];
    }
    return nullptr;
}

int64_t SparseTextureMemoryPoolImpl::handleToIndex(SparseTextureMemoryHandle handle) {
    return handle.uid - 1;
}

struct SparseTextureMemoryMappingIntsance {
    NvFlowTexture3DSparse *m_textureFront;
    NvFlowTexture3DSparse *m_textureBack;
    uint64_t m_versionFront;
    uint64_t m_versionBack;
};

struct SparseTextureMemoryMappingImpl : Object, SparseTextureMemoryMapping {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void pushMapping(uint64_t version, uint32_t *blockTableImage, uint32_t rowPitch,
                     uint32_t depthPitch, uint32_t blockTableImageBytes) override;
    void updateMapping(NvFlowContext *context, uint32_t maxUpdateTextures) override;
    bool canCommitMapping(NvFlowContext *context, uint64_t version) override;
    NvFlowResult commitMapping(NvFlowContext *context, uint64_t version) override;
    int64_t frontVersion() override;
    NvFlowResourceRW *getResourceRW(NvFlowContext *context,
                                    SparseTextureMemoryHandle handle) override;

    // Details
    struct BlockTableImage {
        uint64_t version;
        uint32_t rowPitch;
        uint32_t depthPitch;
        uint32_t imageBytes;
        VectorCached<uint32_t, 1> image;
    };

    SparseTextureMemoryMappingImpl(NvFlowContext *context,
                                   SparseTextureMemoryPoolImpl *memoryPool,
                                   const SparseTextureMemoryMappingDesc *desc);
    ~SparseTextureMemoryMappingImpl();

    void sync(NvFlowContext *context);

    bool updateMapping(NvFlowContext *context, bool updateFront,
                       SparseTextureMemoryMappingIntsance *texMemMappingInst,
                       SparseTextureMemory *texMem);

    SparseTextureMemoryMappingDesc m_desc;
    SparseTextureMemoryPoolImpl *m_memoryPool;
    VectorCached<SparseTextureMemoryMappingIntsance, 16> m_textureMemoryMappingInstance;
    BlockTableImage m_blockTableImage0;
    BlockTableImage m_blockTableImage1;
    BlockTableImage *m_blockTableImageFront;
    BlockTableImage *m_blockTableImageBack;
};

SparseTextureMemoryMappingImpl::SparseTextureMemoryMappingImpl(
    NvFlowContext *context, SparseTextureMemoryPoolImpl *memoryPool,
    const SparseTextureMemoryMappingDesc *desc)
    : m_desc{},
      m_textureMemoryMappingInstance{},
      m_blockTableImage0{},
      m_blockTableImage1{},
      m_blockTableImageFront{0},
      m_blockTableImageBack{0} {
    m_desc = *desc;
    m_memoryPool = memoryPool;
    m_memoryPool->addRef();

    m_blockTableImageFront = &m_blockTableImage0;
    m_blockTableImageBack = &m_blockTableImage1;
}

SparseTextureMemoryMappingImpl::~SparseTextureMemoryMappingImpl() {
    for (auto &texMemMappingInst : m_textureMemoryMappingInstance) {
        SafeRelease(texMemMappingInst.m_textureFront);
        SafeRelease(texMemMappingInst.m_textureBack);
    }

    SafeRelease(m_memoryPool);
}

void SparseTextureMemoryMappingImpl::sync(NvFlowContext *context) {
    while (m_textureMemoryMappingInstance.size() < m_memoryPool->m_textureMemory.size()) {
        uint32_t allocIdx = m_textureMemoryMappingInstance.allocateBack();
        auto &texMemMappingInst = m_textureMemoryMappingInstance[allocIdx];
        texMemMappingInst.m_textureFront = nullptr;
        texMemMappingInst.m_textureBack = nullptr;
        texMemMappingInst.m_versionFront = 0;
        texMemMappingInst.m_versionBack = 0;
        if (m_memoryPool->m_desc.enableVTR) {
            NvFlowTexture3DSparseDesc texDesc = {};
            texDesc.dim = m_desc.virtualDim;
            texDesc.format = m_memoryPool->m_desc.format;
            texMemMappingInst.m_textureFront =
                NvFlowCreateTexture3DSparse(context, &texDesc);
            texMemMappingInst.m_textureBack =
                NvFlowCreateTexture3DSparse(context, &texDesc);
        }
    }

    while (m_textureMemoryMappingInstance.size() < m_memoryPool->m_textureMemory.size()) {
        auto &texMemMappingInst = m_textureMemoryMappingInstance.back();
        SafeRelease(texMemMappingInst.m_textureFront);
        SafeRelease(texMemMappingInst.m_textureBack);
        texMemMappingInst.m_versionFront = 0;
        texMemMappingInst.m_versionBack = 0;
        m_textureMemoryMappingInstance.pop_back();
    }
}

bool SparseTextureMemoryMappingImpl::updateMapping(
    NvFlowContext *context, bool updateFront,
    SparseTextureMemoryMappingIntsance *texMemMappingInst, SparseTextureMemory *texMem) {
    uint64_t &mappingInstVersion =
        updateFront ? texMemMappingInst->m_versionFront : texMemMappingInst->m_versionBack;
    NvFlowTexture3DSparse *textureVTR =
        updateFront ? texMemMappingInst->m_textureFront : texMemMappingInst->m_textureBack;
    auto *blockTable = updateFront ? m_blockTableImageFront : m_blockTableImageBack;

    if (mappingInstVersion == blockTable->version) return false;

    if (m_memoryPool->m_desc.enableVTR) {
        NvFlowContextUpdateSparseMapping(context, textureVTR, texMem->m_heap,
                                         blockTable->image.data(), blockTable->rowPitch,
                                         blockTable->depthPitch);
    }

    mappingInstVersion = blockTable->version;
    return true;
}

SparseTextureMemoryMapping *createSparseTextureMemoryMapping(
    NvFlowContext *context, SparseTextureMemoryPoolImpl *memoryPool,
    const SparseTextureMemoryMappingDesc *desc) {
    return new SparseTextureMemoryMappingImpl(context, memoryPool, desc);
}

uint64_t SparseTextureMemoryMappingImpl::getGPUBytesUsed() {
    uint64_t totalBytes = 0;
    for (uint32_t idx = 0; idx < m_textureMemoryMappingInstance.size(); ++idx) {
        auto &texMemMappingInst = m_textureMemoryMappingInstance[idx];
        if (texMemMappingInst.m_textureFront) {
            auto texObj =
                NvFlowTexture3DSparseGetContextObject(texMemMappingInst.m_textureFront);
            totalBytes += NvFlowContextObjectGetGPUBytesUsed(texObj);
        }
        if (texMemMappingInst.m_textureBack) {
            auto texObj =
                NvFlowTexture3DSparseGetContextObject(texMemMappingInst.m_textureBack);
            totalBytes += NvFlowContextObjectGetGPUBytesUsed(texObj);
        }
    }
    return totalBytes;
}

void SparseTextureMemoryMappingImpl::pushMapping(uint64_t version,
                                                 uint32_t *blockTableImage,
                                                 uint32_t rowPitch, uint32_t depthPitch,
                                                 uint32_t blockTableImageBytes) {
    auto blockTable = m_blockTableImageBack;
    blockTable->version = version;
    blockTable->rowPitch = rowPitch;
    blockTable->depthPitch = depthPitch;
    blockTable->imageBytes = blockTableImageBytes;
    blockTable->image.resize(blockTableImageBytes / sizeof(uint32_t));
    memcpy(blockTable->image.data(), blockTableImage, blockTableImageBytes);
}

void SparseTextureMemoryMappingImpl::updateMapping(NvFlowContext *context,
                                                   uint32_t maxUpdateTextures) {
    sync(context);

    uint32_t updateCount = maxUpdateTextures;
    for (uint32_t idx = 0; idx < m_textureMemoryMappingInstance.size() && updateCount;
         ++idx) {
        auto &texMem = m_memoryPool->m_textureMemory[idx];
        auto &texMemMappingInst = m_textureMemoryMappingInstance[idx];

        if (updateMapping(context, 0, &texMemMappingInst, &texMem)) --updateCount;
    }
}

bool SparseTextureMemoryMappingImpl::canCommitMapping(NvFlowContext *context,
                                                      uint64_t version) {
    sync(context);

    const bool allComplete = 1;
    for (uint32_t idx = 0; idx < m_textureMemoryMappingInstance.size(); ++idx) {
        auto &texMemMappingInst = m_textureMemoryMappingInstance[idx];
        if (texMemMappingInst.m_versionBack != version) return 0;
    }
    return allComplete;
}

NvFlowResult SparseTextureMemoryMappingImpl::commitMapping(NvFlowContext *context,
                                                           uint64_t version) {
    if (!canCommitMapping(context, version)) return eNvFlowFail;

    for (uint32_t idx = 0; idx < m_textureMemoryMappingInstance.size(); ++idx) {
        auto &texMemMappingInst = m_textureMemoryMappingInstance[idx];
        sparse_swap(texMemMappingInst.m_textureFront, texMemMappingInst.m_textureBack);
        sparse_swap(texMemMappingInst.m_versionFront, texMemMappingInst.m_versionBack);
    }
    sparse_swap(m_blockTableImageFront, m_blockTableImageBack);
    return eNvFlowSuccess;
}

int64_t SparseTextureMemoryMappingImpl::frontVersion() {
    return m_blockTableImageFront->version;
}

NvFlowResourceRW *SparseTextureMemoryMappingImpl::getResourceRW(
    NvFlowContext *context, SparseTextureMemoryHandle handle) {
    sync(context);

    auto texMem = m_memoryPool->handleToPointer(handle);
    if (!texMem) return nullptr;

    uint32_t idx = m_memoryPool->handleToIndex(handle);
    auto texMemMappingInst = m_textureMemoryMappingInstance[idx];

    updateMapping(context, 1, &texMemMappingInst, texMem);

    if (m_memoryPool->m_desc.enableVTR) {
        return NvFlowTexture3DSparseGetResourceRW(texMemMappingInst.m_textureFront);
    } else
        return NvFlowTexture3DGetResourceRW(texMem->m_texture);
}

NvFlowDim computeSparseTexturePoolGridDim(NvFlowFormat format, const NvFlowDim &virtualDim,
                                          float residentScale) {
    NvFlowDim poolGridDim;
    NvFlowDim blockDim;
    NvFlowDim gridDim;
    uint32_t maxVirtualBlocks;
    uint32_t maxBlocks;
    float rgridDimf;

    blockDim = getTileDim(format);
    gridDim = virtualDim / blockDim;
    maxVirtualBlocks = gridDim.z * gridDim.y * gridDim.x;
    maxBlocks = int(residentScale * maxVirtualBlocks);
    rgridDimf = pow(float(maxBlocks), 1.f / 3.f);
    poolGridDim.z = ceil(rgridDimf);
    rgridDimf = sqrt(float(maxBlocks) / poolGridDim.z);
    poolGridDim.y = ceil(rgridDimf);
    poolGridDim.x =
        (maxBlocks + poolGridDim.y * poolGridDim.z - 1) / (poolGridDim.y * poolGridDim.z);

    return poolGridDim;
}

SparseTextureMemoryPool *createSparseTextureMemoryPool(
    NvFlowContext *context, const SparseTextureMemoryPoolDesc *desc) {
    return new SparseTextureMemoryPoolImpl(context, desc);
}

}  // namespace NvFlow