#include "GridProxy.h"
#include "Object.h"
#include "ClientHelper.h"
#include "NvFlowContextImpl.h"
#include "GridImport.h"

namespace NvFlow {

struct GridProxySingleGPU : Object, NvFlowGridProxy {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void push(NvFlowGridExport *gridExport,
              const NvFlowGridProxyFlushParams *params) override;
    void flush(const NvFlowGridProxyFlushParams *params) override;
    NvFlowGridExport *getGridExport(NvFlowContext *renderContext) override;

    // Details
    GridProxySingleGPU(const NvFlowGridProxyDesc *desc);
    ~GridProxySingleGPU();

    NvFlowGridExport *m_gridExport;
    NvFlowContext *m_pushContext;
};

struct GridProxyInterQueueGPU : Object, NvFlowGridProxy {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void push(NvFlowGridExport *gridExport,
              const NvFlowGridProxyFlushParams *params) override;
    void flush(const NvFlowGridProxyFlushParams *params) override;
    NvFlowGridExport *getGridExport(NvFlowContext *renderContext) override;

    // Details
    struct Target {
        Target();
        uint64_t m_pushID;
        int m_state;
        uint64_t m_eventID;
        unsigned int m_minHeaderHeight;
        unsigned int m_minDataHeight;
        NvFlowGridImportStateCPU *m_gridImportStateCPU;
        NvFlowTexture2D *m_gridTextureHeader;
        NvFlowTexture2D *m_gridTextureData;
        NvFlowTexture2D *m_renderTextureHeader;
        NvFlowTexture2D *m_renderTextureData;
        NvFlowContextEventQueue *m_gridEventQueue;
        NvFlowContextEventQueue *m_renderEventQueue;
    };

    GridProxyInterQueueGPU(const NvFlowGridProxyDesc *desc);
    ~GridProxyInterQueueGPU();

    unsigned int m_headerWidthBits;
    unsigned int m_headerWidth;
    unsigned int m_headerHeight;
    unsigned int m_dataWidthBits;
    unsigned int m_dataWidth;
    unsigned int m_dataHeight;
    uint64_t m_currentPushID;
    Target m_targets[eNvFlowGridTextureChannelCount];
    NvFlowGridImport *m_gridImport;
    NvFlowComputeShader *m_serializeCS;
    NvFlowComputeShader *m_serializeHeaderCS;
    NvFlowConstantBuffer *m_gridConstantBuffer;
    NvFlowComputeShader *m_deserializeCS;
    NvFlowComputeShader *m_deserializeHeaderCS;
    NvFlowComputeShader *m_blockTableClearCS;
    NvFlowConstantBuffer *m_renderConstantBuffer;
    NvFlowTexture3D *m_vBlockIdxToBlockID;
};

struct SerializeShaderParams {
    NvFlowShaderLinearParams valueParams;

    NvFlowUint headerWidth;
    NvFlowUint headerHeight;
    NvFlowUint dataWidth;
    NvFlowUint dataHeight;

    NvFlowUint headerWidthBits;
    NvFlowUint dataWidthBits;
    NvFlowUint numBlocks;
    NvFlowUint blockStart;
};
using DeserializeShaderParams = SerializeShaderParams;

struct GridProxyInterQueueCommonMemory : Object, NvFlowGridProxy {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void push(NvFlowGridExport *gridExport,
              const NvFlowGridProxyFlushParams *params) override;
    void flush(const NvFlowGridProxyFlushParams *params) override;
    NvFlowGridExport *getGridExport(NvFlowContext *renderContext) override;

    // Details
    struct Target {
        Target();

        struct PerLayer {
            NvFlowResourceReference *m_renderData;
            NvFlowResourceReference *m_renderBlockTable;
            NvFlowResourceReference *m_renderBlockList;
        };

        uint64_t m_pushID;
        int m_state;
        uint64_t m_eventID;
        NvFlowGridImportStateCPU *m_gridImportStateCPU;
        NvFlowGridImport *m_gridImport;
        VectorCached<PerLayer, 8> perLayer;
        NvFlowContextEventQueue *m_gridEventQueue;
        NvFlowContextEventQueue *m_renderEventQueue;
    };

    GridProxyInterQueueCommonMemory(const NvFlowGridProxyDesc *desc);
    ~GridProxyInterQueueCommonMemory();

    uint64_t m_currentPushID;
    unsigned int m_frontTargetIdx;
    Target m_targets[4];
    NvFlowComputeShader *m_gridProxyInterQueueCS;
    NvFlowConstantBuffer *m_gridConstantBuffer;
};

struct GridProxyInterQueueShaderParams {
    NvFlowShaderLinearParams srcParams;
    NvFlowShaderLinearParams params;
};

struct GridProxyMultiGPU : Object, NvFlowGridProxy {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void push(NvFlowGridExport *gridExport,
              const NvFlowGridProxyFlushParams *params) override;
    void flush(const NvFlowGridProxyFlushParams *params) override;
    NvFlowGridExport *getGridExport(NvFlowContext *renderContext) override;

    struct Target {
        Target();
        uint64_t m_pushID;
        int m_state;
        uint64_t m_eventID;
        unsigned int m_minHeaderHeight;
        unsigned int m_minDataHeight;
        unsigned int pipeIdx;
        NvFlowGridImportStateCPU *m_gridImportStateCPU;
        NvFlowTexture2D *m_renderCopyTextureHeader;
        NvFlowTexture2D *m_renderCopyTextureData;
        NvFlowTexture2D *m_renderTextureHeader;
        NvFlowTexture2D *m_renderTextureData;
        NvFlowContextEventQueue *m_renderEventQueue;
    };

    struct Pipe {
        Pipe();
        int m_state;
        uint64_t m_eventID;
        NvFlowTexture2D *m_gridTextureHeader;
        NvFlowTexture2D *m_gridCopyTextureHeader;
        NvFlowTexture2DCrossAdapter *m_gridCrossAdapterHeader;
        NvFlowTexture2DCrossAdapter *m_renderCrossAdapterHeader;
        NvFlowTexture2D *m_gridTextureData;
        NvFlowTexture2D *m_gridCopyTextureData;
        NvFlowTexture2DCrossAdapter *m_gridCrossAdapterData;
        NvFlowTexture2DCrossAdapter *m_renderCrossAdapterData;
        NvFlowContextEventQueue *m_gridEventQueue;
        NvFlowContextEventQueue *m_gridCopyEventQueue;
        NvFlowContextEventQueue *m_renderCopyEventQueue;
        NvFlowFence *m_gridFence;
        NvFlowFence *m_gridCopyFence;
        NvFlowFence *m_renderCopyFence;
        uint64_t m_gridFenceValue;
        uint64_t m_sharedFenceValue;
    };

    GridProxyMultiGPU(const NvFlowGridProxyDesc *desc);
    ~GridProxyMultiGPU();

    unsigned int m_headerWidthBits;
    unsigned int m_headerWidth;
    unsigned int m_headerHeight;
    unsigned int m_dataWidthBits;
    unsigned int m_dataWidth;
    unsigned int m_dataHeight;
    uint64_t m_currentPushID;
    Target m_targets[8];
    Pipe m_pipes[4];
    NvFlowGridImport *m_gridImport;
    NvFlowComputeShader *m_serializeCS;
    NvFlowComputeShader *m_serializeHeaderCS;
    NvFlowConstantBuffer *m_gridConstantBuffer;
    NvFlowComputeShader *m_deserializeCS;
    NvFlowComputeShader *m_deserializeHeaderCS;
    NvFlowComputeShader *m_blockTableClearCS;
    NvFlowConstantBuffer *m_renderConstantBuffer;
    NvFlowTexture3D *m_vBlockIdxToBlockID;
};

///
/// Implements
///

uint64_t GridProxySingleGPU::getGPUBytesUsed() {
    return 0;
}

void GridProxySingleGPU::push(NvFlowGridExport *gridExport,
                              const NvFlowGridProxyFlushParams *params) {
    m_gridExport = gridExport;
    m_pushContext = params->gridContext;
}

void GridProxySingleGPU::flush(const NvFlowGridProxyFlushParams *params) {}

NvFlowGridExport *GridProxySingleGPU::getGridExport(NvFlowContext *renderContext) {
    return m_gridExport;
}

GridProxySingleGPU::GridProxySingleGPU(const NvFlowGridProxyDesc *desc) {
    m_gridExport = desc->gridExport;
    m_pushContext = desc->gridContext;
}

GridProxySingleGPU::~GridProxySingleGPU() {}

uint64_t GridProxyInterQueueGPU::getGPUBytesUsed() {
    return 0;
}

void GridProxyInterQueueGPU::push(NvFlowGridExport *gridExport,
                                  const NvFlowGridProxyFlushParams *params) {
    auto gridContext = params->gridContext;
    uint32_t targetIdx;
    for (targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        if (!target.m_state) break;
    }

    if (targetIdx < eNvFlowGridTextureChannelCount) {
        auto &target = m_targets[targetIdx];
        target.m_pushID = ++m_currentPushID;

        NvFlowContextProfileGroupBegin(gridContext, L"GridProxySerialize");

        auto exportHandle = NvFlowGridExportGetHandle(gridExport, gridContext,
                                                      eNvFlowGridTextureChannelDensity);
        NvFlowGridExportLayeredView exportLayeredView = {};
        NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);
        NvFlowGridImportUpdateStateCPU(target.m_gridImportStateCPU, gridContext,
                                       gridExport);
        uint32_t blockCells = exportLayeredView.mapping.shaderParams.blockDim.w;
        uint32_t totalCells = exportLayeredView.mapping.shaderParams.blockDim.w *
                              exportLayeredView.mapping.layeredNumBlocks;
        uint32_t headerHeight =
            (exportLayeredView.mapping.maxBlocks + m_headerWidth - 1) / m_headerWidth;
        uint32_t dataHeight = (totalCells + m_dataWidth - 1) / m_dataWidth;

        target.m_minHeaderHeight = headerHeight;
        target.m_minDataHeight = dataHeight;
        uint32_t blockStart = 0;
        for (uint32_t layerIdx = 0; layerIdx < exportHandle.numLayerViews; ++layerIdx) {
            NvFlowGridExportLayerView exportLayerView = {};
            NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);

            auto mapped = (SerializeShaderParams *)NvFlowConstantBufferMap(
                gridContext, m_gridConstantBuffer);
            if (mapped) {
                mapped->valueParams = exportLayeredView.mapping.shaderParams;
                mapped->headerWidth = m_headerWidth;
                mapped->headerHeight = m_headerHeight;
                mapped->dataWidth = m_dataWidth;
                mapped->dataHeight = m_dataHeight;
                mapped->headerWidthBits = m_headerWidthBits;
                mapped->dataWidthBits = m_dataWidthBits;
                mapped->numBlocks = exportLayerView.mapping.numBlocks;
                mapped->blockStart = blockStart;
                NvFlowConstantBufferUnmap(gridContext, m_gridConstantBuffer);
            }

            NvFlowDim gridDim;
            gridDim.x = (exportLayerView.mapping.numBlocks + 63) / 64;
            gridDim.y = 1;
            gridDim.z = 1;
            NvFlowDispatchParams dispatchParams = {};
            dispatchParams.shader = m_serializeHeaderCS;
            dispatchParams.gridDim = gridDim;
            dispatchParams.rootConstantBuffer = m_gridConstantBuffer;
            dispatchParams.readOnly[0] = exportLayerView.mapping.blockList;
            dispatchParams.readOnly[1] = exportLayerView.mapping.blockTable;
            dispatchParams.readWrite[0] =
                NvFlowTexture2DGetResourceRW(target.m_gridTextureHeader);
            NvFlowContextDispatch(gridContext, &dispatchParams);

            gridDim.x = (exportLayeredView.mapping.shaderParams.blockDim.x *
                             exportLayerView.mapping.numBlocks +
                         7) /
                        8;
            gridDim.y = (exportLayeredView.mapping.shaderParams.blockDim.y + 7) / 8;
            gridDim.z = (exportLayeredView.mapping.shaderParams.blockDim.z + 7) / 8;
            ZeroMemory(&dispatchParams, sizeof(dispatchParams));
            dispatchParams.shader = m_serializeCS;
            dispatchParams.gridDim = gridDim;
            dispatchParams.rootConstantBuffer = m_gridConstantBuffer;
            dispatchParams.readOnly[0] = exportLayerView.mapping.blockList;
            dispatchParams.readOnly[1] = exportLayerView.data;
            dispatchParams.readOnly[2] = exportLayerView.mapping.blockTable;
            dispatchParams.readWrite[0] =
                NvFlowTexture2DGetResourceRW(target.m_gridTextureData);
            NvFlowContextDispatch(gridContext, &dispatchParams);
            blockStart += exportLayerView.mapping.numBlocks;
        }

        NvFlowResource *resource;
        resource = NvFlowTexture2DGetResource(target.m_gridTextureHeader);
        NvFlowContextTransitionToCommonState(gridContext, resource);
        resource = NvFlowTexture2DGetResource(target.m_gridTextureData);
        NvFlowContextTransitionToCommonState(gridContext, resource);

        uint64_t uid = target.m_eventID++;
        NvFlowContextEventQueuePush(gridContext, target.m_gridEventQueue, uid);

        NvFlowContextProfileGroupEnd(gridContext);
        target.m_state = 1;
    }

    flush(params);
}

void GridProxyInterQueueGPU::flush(const NvFlowGridProxyFlushParams *params) {
    auto gridContext = params->gridContext;
    for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        if (target.m_state == 1) {
            uint64_t eventID = 0;
            if (NvFlowContextEventQueuePop(gridContext, target.m_gridEventQueue,
                                           &eventID) == eNvFlowSuccess)
                target.m_state = 2;
        }
    }
}

NvFlowGridExport *GridProxyInterQueueGPU::getGridExport(NvFlowContext *renderContext) {
    uint32_t numTargetsReady = eNvFlowGridTextureChannelCount;

    while (numTargetsReady >= 2) {
        numTargetsReady = 0;

        uint64_t minPushID = 0;
        uint32_t minPushID_targetIdx = 0;
        for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount;
             ++targetIdx) {
            auto &target = m_targets[targetIdx];
            if (target.m_state == 2) {
                if (minPushID) {
                    if (target.m_pushID < minPushID) {
                        minPushID = target.m_pushID;
                        minPushID_targetIdx = targetIdx;
                    }
                } else {
                    minPushID = target.m_pushID;
                    minPushID_targetIdx = targetIdx;
                }
                ++numTargetsReady;
            }
        }

        bool isOverloaded = numTargetsReady >= 2;
        if (numTargetsReady) {
            auto &target = m_targets[minPushID_targetIdx];
            if (target.m_state == 2) {
                bool shouldDeserialize = !isOverloaded;
                if (!isOverloaded) {
                    NvFlowGridImportStateCPUParams importParams = {};
                    importParams.stateCPU = target.m_gridImportStateCPU;
                    importParams.channel = eNvFlowGridTextureChannelDensity;
                    importParams.importMode = eNvFlowGridImportModeLinear;

                    auto importHandle = NvFlowGridImportStateCPUGetHandle(
                        m_gridImport, renderContext, &importParams);

                    NvFlowGridImportLayeredView importLayeredView = {};
                    NvFlowGridImportGetLayeredView(importHandle, &importLayeredView);

                    NvFlowContextProfileGroupBegin(renderContext, L"GridProxyDeserialize");

                    uint32_t blockStart = 0;
                    for (uint32_t layerIdx = 0; layerIdx < importHandle.numLayerViews;
                         ++layerIdx) {
                        NvFlowGridImportLayerView importLayerView = {};
                        NvFlowGridImportGetLayerView(importHandle, layerIdx,
                                                     &importLayerView);

                        auto clear = [renderContext, this](NvFlowTexture3D *tex,
                                                           NvFlowResourceRW *resourceRW) {
                            NvFlowTexture3DDesc tex_desc;
                            NvFlowTexture3DGetDesc(tex, &tex_desc);
                            NvFlowDim gridDim = (tex_desc.dim + 7) >> 3;
                            NvFlowDispatchParams dispatchParams = {};
                            dispatchParams.shader = m_blockTableClearCS;
                            dispatchParams.gridDim = gridDim;
                            dispatchParams.rootConstantBuffer = 0;
                            dispatchParams.readWrite[0] = resourceRW;
                            NvFlowContextDispatch(renderContext, &dispatchParams);
                        };

                        clear(m_vBlockIdxToBlockID, importLayerView.blockListRW);
                        clear(m_vBlockIdxToBlockID,
                              NvFlowTexture3DGetResourceRW(m_vBlockIdxToBlockID));

                        auto shaderParams =
                            (DeserializeShaderParams *)NvFlowConstantBufferMap(
                                renderContext, m_renderConstantBuffer);
                        if (shaderParams) {
                            shaderParams->valueParams =
                                importLayeredView.mapping.shaderParams;
                            shaderParams->headerWidth = m_headerWidth;
                            shaderParams->headerHeight = m_headerHeight;
                            shaderParams->dataWidth = m_dataWidth;
                            shaderParams->dataHeight = m_dataHeight;
                            shaderParams->headerWidthBits = m_headerWidthBits;
                            shaderParams->dataWidthBits = m_dataWidthBits;
                            shaderParams->numBlocks = importLayerView.mapping.numBlocks;
                            shaderParams->blockStart = blockStart;
                            shaderParams->valueParams.isVTR = make_uint4(0);
                            NvFlowConstantBufferUnmap(renderContext,
                                                      m_renderConstantBuffer);
                        }

                        NvFlowDim gridDim;
                        gridDim.x = (importLayeredView.mapping.maxBlocks + 63) / 64;
                        gridDim.y = 1;
                        gridDim.z = 1;
                        NvFlowDispatchParams dispatchParams = {};
                        dispatchParams.shader = m_deserializeHeaderCS;
                        dispatchParams.gridDim = gridDim;
                        dispatchParams.rootConstantBuffer = m_renderConstantBuffer;
                        dispatchParams.readOnly[0] =
                            NvFlowTexture2DGetResource(target.m_renderTextureHeader);
                        dispatchParams.readWrite[0] = importLayerView.blockListRW;
                        dispatchParams.readWrite[1] = importLayerView.blockTableRW;
                        dispatchParams.readWrite[2] =
                            NvFlowTexture3DGetResourceRW(m_vBlockIdxToBlockID);
                        NvFlowContextDispatch(renderContext, &dispatchParams);

                        gridDim.x =
                            (importLayeredView.mapping.shaderParams.linearBlockDim.w +
                             127) /
                            128;
                        gridDim.y = importLayerView.mapping.numBlocks + 7;
                        gridDim.z = 1;
                        NvFlowDispatchParams params = {};
                        params.shader = m_deserializeCS;
                        params.gridDim = gridDim;
                        params.rootConstantBuffer = m_renderConstantBuffer;
                        params.readOnly[0] =
                            NvFlowTexture2DGetResource(target.m_renderTextureData);
                        params.readOnly[1] =
                            NvFlowResourceRWGetResource(importLayerView.blockListRW);
                        params.readOnly[2] =
                            NvFlowResourceRWGetResource(importLayerView.blockTableRW);
                        params.readOnly[3] =
                            NvFlowTexture3DGetResource(m_vBlockIdxToBlockID);
                        params.readWrite[0] = importLayerView.dataRW;
                        NvFlowContextDispatch(renderContext, &params);

                        blockStart += importLayerView.mapping.numBlocks;
                    }

                    NvFlowContextProfileGroupEnd(renderContext);
                }

                NvFlowResource *resource;
                resource = NvFlowTexture2DGetResource(target.m_renderTextureHeader);
                NvFlowContextTransitionToCommonState(renderContext, resource);
                resource = NvFlowTexture2DGetResource(target.m_renderTextureData);
                NvFlowContextTransitionToCommonState(renderContext, resource);
                uint64_t uid = target.m_eventID++;
                NvFlowContextEventQueuePush(renderContext, target.m_renderEventQueue, uid);
                target.m_state = 3;
            }
        }
    }

    for (auto &target : m_targets) {
        if (target.m_state == 3) {
            uint64_t eventID = 0;
            if (NvFlowContextEventQueuePop(renderContext, target.m_renderEventQueue,
                                           &eventID) == eNvFlowSuccess)
                target.m_state = 0;
        }
    }

    return NvFlowGridImportGetGridExport(m_gridImport, renderContext);
}

#include "gridViewSerializeCS.hlsl.h"
#include "gridViewSerializeHeaderCS.hlsl.h"
#include "gridViewDeserializeCS.hlsl.h"
#include "gridViewDeserializeHeaderCS.hlsl.h"
#include "sparseClearCS.hlsl.h"

GridProxyInterQueueGPU::GridProxyInterQueueGPU(const NvFlowGridProxyDesc *desc)
    : m_headerWidthBits{0},
      m_headerWidth{0},
      m_headerHeight{0},
      m_dataWidthBits{0},
      m_dataWidth{0},
      m_dataHeight{0},
      m_currentPushID{0},
      m_targets{},
      m_gridImport{0},
      m_serializeCS{0},
      m_serializeHeaderCS{0},
      m_gridConstantBuffer{0},
      m_deserializeCS{0},
      m_deserializeHeaderCS{0},
      m_blockTableClearCS{0},
      m_renderConstantBuffer{0},
      m_vBlockIdxToBlockID{0} {
    NvFlowDim blockTableDim;
    auto exportHandle = NvFlowGridExportGetHandle(desc->gridExport, desc->gridContext,
                                                  eNvFlowGridTextureChannelDensity);
    NvFlowGridExportLayeredView exportLayeredView = {};
    NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

    blockTableDim = make_dim(exportLayeredView.mapping.shaderParams.gridDim);
    float rt = sqrt(float(exportLayeredView.mapping.maxBlocks));
    uint32_t rtlog2 = ceil(log2(rt));

    m_headerWidthBits = rtlog2;
    m_headerWidth = 1 << m_headerWidthBits;
    m_headerHeight =
        (exportLayeredView.mapping.maxBlocks + m_headerWidth - 1) / m_headerWidth;

    uint32_t blockCells = exportLayeredView.mapping.shaderParams.blockDim.z *
                          exportLayeredView.mapping.shaderParams.blockDim.y *
                          exportLayeredView.mapping.shaderParams.blockDim.x;
    uint32_t capacity = blockCells * exportLayeredView.mapping.maxBlocks;
    float x = sqrt(float(capacity));
    m_dataWidthBits = ceil(log2(x));
    m_dataWidth = 1 << m_dataWidthBits;
    m_dataHeight = (capacity + m_dataWidth - 1) / m_dataWidth;

    NvFlowContext *gridContext = desc->gridContext;
    NvFlowContext *renderContext = desc->renderContext;

    NvFlowTexture2DDesc headerTexDesc = {};
    headerTexDesc.format = eNvFlowFormat_r32g32_float;
    headerTexDesc.width = m_headerWidth;
    headerTexDesc.height = m_headerHeight;
    for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        NvFlowContextAPI renderAPIType = NvFlowContextGetContextType(renderContext);
        NvFlowContextAPI gridAPIType = NvFlowContextGetContextType(gridContext);
        if (renderAPIType == gridAPIType) {
            target.m_renderTextureHeader =
                NvFlowCreateTexture2D(renderContext, &headerTexDesc);
            target.m_gridTextureHeader =
                NvFlowShareTexture2D(gridContext, target.m_renderTextureHeader);
        } else {
            target.m_renderTextureHeader =
                NvFlowCreateTexture2DCrossAPI(renderContext, &headerTexDesc);
            target.m_gridTextureHeader =
                NvFlowShareTexture2DCrossAPI(gridContext, target.m_renderTextureHeader);
        }
    }

    NvFlowTexture2DDesc dataTexDesc = {};
    dataTexDesc.format = eNvFlowFormat_r16g16_float;
    dataTexDesc.width = m_dataWidth;
    dataTexDesc.height = m_dataHeight;
    for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        NvFlowContextAPI renderAPIType = NvFlowContextGetContextType(renderContext);
        NvFlowContextAPI gridAPIType = NvFlowContextGetContextType(gridContext);
        if (renderAPIType == gridAPIType) {
            target.m_renderTextureData = NvFlowCreateTexture2D(renderContext, &dataTexDesc);
            target.m_gridTextureData =
                NvFlowShareTexture2D(gridContext, target.m_renderTextureData);
        } else {
            target.m_renderTextureData =
                NvFlowCreateTexture2DCrossAPI(renderContext, &dataTexDesc);
            target.m_gridTextureData =
                NvFlowShareTexture2DCrossAPI(gridContext, target.m_renderTextureData);
        }
    }

    for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        target.m_gridEventQueue = NvFlowCreateContextEventQueue(gridContext);
        target.m_renderEventQueue = NvFlowCreateContextEventQueue(renderContext);
    }

    auto createShader = [](NvFlowContext *context, const BYTE *cs, uint64_t cs_length,
                           const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_serializeCS =
        createShader(gridContext, NVFLOW_CREATE_SHADER_ARGS(gridViewSerializeCS));
    m_serializeHeaderCS =
        createShader(gridContext, NVFLOW_CREATE_SHADER_ARGS(gridViewSerializeHeaderCS));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(SerializeShaderParams);
    cbDesc.uploadAccess = 1;
    m_gridConstantBuffer = NvFlowCreateConstantBuffer(gridContext, &cbDesc);

    m_deserializeCS =
        createShader(renderContext, NVFLOW_CREATE_SHADER_ARGS(gridViewDeserializeCS));
    m_deserializeHeaderCS =
        createShader(renderContext, NVFLOW_CREATE_SHADER_ARGS(gridViewDeserializeHeaderCS));
    m_blockTableClearCS =
        createShader(renderContext, NVFLOW_CREATE_SHADER_ARGS(sparseClearCS));

    ZeroMemory(&cbDesc, sizeof(cbDesc));
    cbDesc.sizeInBytes = sizeof(SerializeShaderParams);
    cbDesc.uploadAccess = 1;
    m_renderConstantBuffer = NvFlowCreateConstantBuffer(renderContext, &cbDesc);

    NvFlowTexture3DDesc blockTableDesc = {};
    blockTableDesc.format = eNvFlowFormat_r32_uint;
    blockTableDesc.dim = blockTableDim;
    blockTableDesc.uploadAccess = 0;
    blockTableDesc.downloadAccess = 0;
    m_vBlockIdxToBlockID = NvFlowCreateTexture3D(renderContext, &blockTableDesc);

    NvFlowGridImportDesc importDesc = {};
    importDesc.gridExport = desc->gridExport;
    m_gridImport = NvFlowCreateGridImport(renderContext, &importDesc);

    for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        target.m_gridImportStateCPU = NvFlowCreateGridImportStateCPU(m_gridImport);
    }
}

GridProxyInterQueueGPU::~GridProxyInterQueueGPU() {
    for (auto &target : m_targets) {
        SafeRelease(target.m_gridTextureHeader);
        SafeRelease(target.m_gridTextureData);
        SafeRelease(target.m_renderTextureHeader);
        SafeRelease(target.m_renderTextureData);
        SafeRelease(target.m_gridEventQueue);
        SafeRelease(target.m_renderEventQueue);
        SafeRelease(target.m_gridImportStateCPU);
    }

    SafeRelease(m_serializeCS);
    SafeRelease(m_serializeHeaderCS);
    SafeRelease(m_gridConstantBuffer);
    SafeRelease(m_deserializeCS);
    SafeRelease(m_deserializeHeaderCS);
    SafeRelease(m_blockTableClearCS);
    SafeRelease(m_renderConstantBuffer);
    SafeRelease(m_vBlockIdxToBlockID);
    SafeRelease(m_gridImport);
}

GridProxyInterQueueGPU::Target::Target() {
    m_pushID = 0;
    m_state = 0;
    m_eventID = 0;
    m_minHeaderHeight = 0;
    m_minDataHeight = 0;
    m_gridImportStateCPU = 0;
    m_gridTextureHeader = 0;
    m_gridTextureData = 0;
    m_renderTextureHeader = 0;
    m_renderTextureData = 0;
    m_gridEventQueue = 0;
    m_renderEventQueue = 0;
}

GridProxyInterQueueCommonMemory::Target::Target() {
    this->m_pushID = 0;
    this->m_state = 0;
    this->m_eventID = 0;
    this->m_gridImportStateCPU = 0;
    this->m_gridImport = 0;
    this->m_gridEventQueue = 0;
    this->m_renderEventQueue = 0;
}

#include "gridProxyInterQueueCS.hlsl.h"

uint64_t GridProxyInterQueueCommonMemory::getGPUBytesUsed() {
    return 0;
}

void GridProxyInterQueueCommonMemory::push(NvFlowGridExport *gridExport,
                                           const NvFlowGridProxyFlushParams *params) {
    auto gridContext = params->gridContext;
    uint32_t targetIdx;
    for (targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        if (!target.m_state) break;
    }

    if (targetIdx < eNvFlowGridTextureChannelCount) {
        auto &target = m_targets[targetIdx];
        target.m_pushID = ++m_currentPushID;

        NvFlowContextProfileGroupBegin(gridContext, L"GridProxySerialize");

        auto exportHandle = NvFlowGridExportGetHandle(gridExport, gridContext,
                                                      eNvFlowGridTextureChannelDensity);
        NvFlowGridExportLayeredView exportLayeredView = {};
        NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);
        NvFlowGridImportUpdateStateCPU(target.m_gridImportStateCPU, gridContext,
                                       gridExport);
        if (exportHandle.numLayerViews) {
            NvFlowGridImportStateCPUParams importParams = {};
            importParams.stateCPU = target.m_gridImportStateCPU;
            importParams.channel = eNvFlowGridTextureChannelDensity;
            importParams.importMode = eNvFlowGridImportModeLinear;

            NvFlowGridImportHandle importHandle = NvFlowGridImportStateCPUGetHandle(
                target.m_gridImport, gridContext, &importParams);

            NvFlowGridImportLayeredView importLayeredView = {};
            NvFlowGridImportGetLayeredView(importHandle, &importLayeredView);
            for (uint32_t layerIdx = 0; layerIdx < importHandle.numLayerViews; ++layerIdx) {
                NvFlowGridExportLayerView exportLayerView{};
                NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);
                NvFlowGridImportLayerView importLayerView = {};
                NvFlowGridImportGetLayerView(importHandle, layerIdx, &importLayerView);
                NvFlowContextCopyResource(gridContext, importLayerView.blockListRW,
                                          exportLayerView.mapping.blockList);
                NvFlowContextCopyResource(gridContext, importLayerView.blockTableRW,
                                          exportLayerView.mapping.blockTable);

                auto shaderParams =
                    (GridProxyInterQueueShaderParams *)NvFlowConstantBufferMap(
                        gridContext, m_gridConstantBuffer);
                if (shaderParams) {
                    shaderParams->params = importLayeredView.mapping.shaderParams;
                    shaderParams->srcParams = exportLayeredView.mapping.shaderParams;
                    NvFlowConstantBufferUnmap(gridContext, m_gridConstantBuffer);
                }

                NvFlowDim gridDim;
                gridDim.x =
                    (importLayeredView.mapping.shaderParams.linearBlockDim.w + 127) / 128;
                gridDim.y = importLayerView.mapping.numBlocks;
                gridDim.z = 1;

                NvFlowDispatchParams dispatchParams = {};
                dispatchParams.shader = m_gridProxyInterQueueCS;
                dispatchParams.gridDim = gridDim;
                dispatchParams.rootConstantBuffer = m_gridConstantBuffer;
                dispatchParams.readOnly[0] = importLayerView.mapping.blockList;
                dispatchParams.readOnly[1] = exportLayerView.data;
                dispatchParams.readOnly[2] = exportLayerView.mapping.blockTable;
                dispatchParams.readWrite[0] = importLayerView.dataRW;
                NvFlowContextDispatch(gridContext, &dispatchParams);

                NvFlowResource *resource;
                resource = NvFlowResourceRWGetResource(importLayerView.blockListRW);
                NvFlowContextTransitionToCommonState(gridContext, resource);
                resource = NvFlowResourceRWGetResource(importLayerView.blockTableRW);
                NvFlowContextTransitionToCommonState(gridContext, resource);
                resource = NvFlowResourceRWGetResource(importLayerView.dataRW);
                NvFlowContextTransitionToCommonState(gridContext, resource);
            }
        }
        uint64_t uid = target.m_eventID++;
        NvFlowContextEventQueuePush(gridContext, target.m_gridEventQueue, uid);

        NvFlowContextProfileGroupEnd(gridContext);
        target.m_state = 1;
    }

    flush(params);
}

void GridProxyInterQueueCommonMemory::flush(const NvFlowGridProxyFlushParams *params) {
    auto gridContext = params->gridContext;
    for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        if (target.m_state == 1) {
            uint64_t eventID = 0;
            if (NvFlowContextEventQueuePop(gridContext, target.m_gridEventQueue,
                                           &eventID) == eNvFlowSuccess)
                target.m_state = 2;
        }
    }
}

NvFlowGridExport *GridProxyInterQueueCommonMemory::getGridExport(
    NvFlowContext *renderContext) {
    uint32_t numTargetsReady = eNvFlowGridTextureChannelCount;
    while (numTargetsReady >= 2) {
        numTargetsReady = 0;
        uint64_t minPushID = 0;
        uint32_t minPushID_targetIdx = 0;
        for (uint32_t targetIdx = 0; targetIdx < eNvFlowGridTextureChannelCount;
             ++targetIdx) {
            auto &target = m_targets[targetIdx];
            if (target.m_state == 2) {
                if (minPushID) {
                    if (target.m_pushID < minPushID) {
                        minPushID = target.m_pushID;
                        minPushID_targetIdx = targetIdx;
                    }
                } else {
                    minPushID = target.m_pushID;
                    minPushID_targetIdx = targetIdx;
                }
                ++numTargetsReady;
            }
        }

        bool isOverloaded = numTargetsReady >= 2;
        if (numTargetsReady) {
            auto &target = m_targets[minPushID_targetIdx];
            if (target.m_state == 2) {
                auto gridExport =
                    NvFlowGridImportGetGridExport(target.m_gridImport, renderContext);
                auto exportHandle = NvFlowGridExportGetHandle(
                    gridExport, renderContext, eNvFlowGridTextureChannelDensity);
                NvFlowGridExportLayeredView exportLayeredView = {};
                NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

                while (target.perLayer.size() < exportHandle.numLayerViews) {
                    uint32_t layerIdx = target.perLayer.allocateBack();
                    auto &perLayer = target.perLayer[layerIdx];
                    perLayer.m_renderData = 0;
                    perLayer.m_renderBlockList = 0;
                    perLayer.m_renderBlockTable = 0;
                }

                for (uint32_t idx = 0; idx < exportHandle.numLayerViews; ++idx) {
                    auto &perLayer = target.perLayer[idx];
                    NvFlowGridExportLayerView exportLayerView = {};
                    NvFlowGridExportGetLayerView(exportHandle, idx, &exportLayerView);
                    if (!perLayer.m_renderData) {
                        perLayer.m_renderData = NvFlowShareResourceReference(
                            renderContext, exportLayerView.data);
                    }
                    if (!perLayer.m_renderBlockTable) {
                        perLayer.m_renderBlockTable = NvFlowShareResourceReference(
                            renderContext, exportLayerView.mapping.blockTable);
                    }
                    if (!perLayer.m_renderBlockList) {
                        perLayer.m_renderBlockList = NvFlowShareResourceReference(
                            renderContext, exportLayerView.mapping.blockList);
                    }
                }

                target.m_state = 3;
            }
        }
    }

    uint64_t maxPushID = 0;
    uint32_t maxTargetIdx = 0;
    for (uint32_t i = 0; i < eNvFlowGridTextureChannelCount; ++i) {
        auto &target = m_targets[i];
        if (target.m_state == 3 && target.m_pushID >= maxPushID) {
            maxPushID = target.m_pushID;
            maxTargetIdx = i;
        }
    }
    m_frontTargetIdx = maxTargetIdx;

    for (uint32_t i = 0; i < eNvFlowGridTextureChannelCount; ++i) {
        auto &target = m_targets[i];
        if (target.m_state == 3 && target.m_pushID < maxPushID && i != m_frontTargetIdx) {
            auto gridExport =
                NvFlowGridImportGetGridExport(target.m_gridImport, renderContext);
            auto exportHandle = NvFlowGridExportGetHandle(gridExport, renderContext,
                                                          eNvFlowGridTextureChannelDensity);
            NvFlowGridExportLayeredView exportLayeredView = {};
            NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);
            for (uint32_t layerIdx = 0; layerIdx < exportHandle.numLayerViews; ++layerIdx) {
                NvFlowGridExportLayerView resource = {};
                NvFlowGridExportGetLayerView(exportHandle, layerIdx, &resource);
                NvFlowContextTransitionToCommonState(renderContext, resource.data);
                NvFlowContextTransitionToCommonState(renderContext,
                                                     resource.mapping.blockTable);

                NvFlowContextTransitionToCommonState(renderContext,
                                                     resource.mapping.blockList);
            }

            uint64_t uid = target.m_eventID++;
            NvFlowContextEventQueuePush(renderContext, target.m_gridEventQueue, uid);
            target.m_state = 4;
        }
    }

    for (uint32_t i = 0; i < eNvFlowGridTextureChannelCount; ++i) {
        auto &target = m_targets[i];
        if (target.m_state == 4) {
            uint64_t eventID = 0;
            if (NvFlowContextEventQueuePop(renderContext, target.m_gridEventQueue,
                                           &eventID) == eNvFlowSuccess)
                target.m_state = 0;
        }
    }

    return NvFlowGridImportGetGridExport(m_targets[m_frontTargetIdx].m_gridImport,
                                         renderContext);
}

GridProxyInterQueueCommonMemory::GridProxyInterQueueCommonMemory(
    const NvFlowGridProxyDesc *desc)
    : m_currentPushID{0},
      m_frontTargetIdx{0},
      m_targets{},
      m_gridProxyInterQueueCS{0},
      m_gridConstantBuffer{0} {
    auto gridContext = desc->gridContext;
    auto renderContext = desc->renderContext;

    for (auto &target : m_targets) {
        NvFlowGridImportDesc importDesc = {};
        importDesc.gridExport = desc->gridExport;
        target.m_gridImport = NvFlowCreateGridImport(renderContext, &importDesc);
        target.m_gridImportStateCPU = NvFlowCreateGridImportStateCPU(target.m_gridImport);
        target.m_gridEventQueue = NvFlowCreateContextEventQueue(gridContext);
        target.m_renderEventQueue = NvFlowCreateContextEventQueue(renderContext);
    }

    m_targets[0].m_state = 3;

    m_frontTargetIdx = 0;

    auto createShader = [](NvFlowContext *context, const BYTE *cs, uint64_t cs_length,
                           const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_gridProxyInterQueueCS =
        createShader(gridContext, NVFLOW_CREATE_SHADER_ARGS(gridProxyInterQueueCS));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(GridProxyInterQueueShaderParams);
    cbDesc.uploadAccess = 1;
    m_gridConstantBuffer = NvFlowCreateConstantBuffer(gridContext, &cbDesc);
}

GridProxyInterQueueCommonMemory::~GridProxyInterQueueCommonMemory() {
    for (auto &target : m_targets) {
        SafeRelease(target.m_gridImport);
        SafeRelease(target.m_gridImportStateCPU);
        SafeRelease(target.m_gridEventQueue);
        SafeRelease(target.m_renderEventQueue);
        for (auto &perLayer : target.perLayer) {
            SafeRelease(perLayer.m_renderBlockList);
            SafeRelease(perLayer.m_renderBlockTable);
            SafeRelease(perLayer.m_renderData);
        }
    }
    SafeRelease(m_gridProxyInterQueueCS);
    SafeRelease(m_gridConstantBuffer);
}

uint64_t GridProxyMultiGPU::getGPUBytesUsed() {
    return 0;
}

void GridProxyMultiGPU::push(NvFlowGridExport *gridExport,
                             const NvFlowGridProxyFlushParams *params) {}

void GridProxyMultiGPU::flush(const NvFlowGridProxyFlushParams *params) {}

NvFlowGridExport *GridProxyMultiGPU::getGridExport(NvFlowContext *renderContext) {
    return nullptr;
}

GridProxyMultiGPU::GridProxyMultiGPU(const NvFlowGridProxyDesc *desc)
    : m_headerWidthBits{0},
      m_headerWidth{0},
      m_headerHeight{0},
      m_dataWidthBits{0},
      m_dataWidth{0},
      m_dataHeight{0},
      m_currentPushID{0},
      m_targets{},
      m_pipes{},
      m_gridImport{0},
      m_serializeCS{0},
      m_serializeHeaderCS{0},
      m_gridConstantBuffer{0},
      m_deserializeCS{0},
      m_deserializeHeaderCS{0},
      m_blockTableClearCS{0},
      m_renderConstantBuffer{0},
      m_vBlockIdxToBlockID{0} {
    NvFlowDim blockTableDim;
    auto exportHandle = NvFlowGridExportGetHandle(desc->gridExport, desc->gridContext,
                                                  eNvFlowGridTextureChannelDensity);
    NvFlowGridExportLayeredView exportLayeredView = {};
    NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

    blockTableDim = make_dim(exportLayeredView.mapping.shaderParams.gridDim);
    float rt = sqrt(float(exportLayeredView.mapping.maxBlocks));
    uint32_t rtlog2 = ceil(log2(rt));

    m_headerWidthBits = rtlog2;
    m_headerWidth = 1 << m_headerWidthBits;
    m_headerHeight =
        (exportLayeredView.mapping.maxBlocks + m_headerWidth - 1) / m_headerWidth;

    uint32_t blockCells = exportLayeredView.mapping.shaderParams.blockDim.z *
                          exportLayeredView.mapping.shaderParams.blockDim.y *
                          exportLayeredView.mapping.shaderParams.blockDim.x;
    uint32_t capacity = blockCells * exportLayeredView.mapping.maxBlocks;
    float x = sqrt(float(capacity));
    m_dataWidthBits = ceil(log2(x));
    m_dataWidth = 1 << m_dataWidthBits;
    m_dataHeight = (capacity + m_dataWidth - 1) / m_dataWidth;

    NvFlowContext *gridContext = desc->gridContext;
    NvFlowContext *gridCopyContext = desc->gridCopyContext;
    NvFlowContext *renderCopyContext = desc->renderCopyContext;
    NvFlowContext *renderContext = desc->renderContext;

    NvFlowTexture2DDesc headerTexDesc = {};
    headerTexDesc.format = eNvFlowFormat_r32g32_float;
    headerTexDesc.width = m_headerWidth;
    headerTexDesc.height = m_headerHeight;

    for (uint32_t pipeIdx = 0; pipeIdx < 4; ++pipeIdx) {
        auto &pipe = m_pipes[pipeIdx];
        pipe.m_gridTextureHeader = NvFlowCreateTexture2D(gridContext, &headerTexDesc);
        pipe.m_gridCopyTextureHeader =
            NvFlowShareTexture2D(gridCopyContext, pipe.m_gridTextureHeader);
        pipe.m_gridCrossAdapterHeader =
            NvFlowCreateTexture2DCrossAdapter(gridCopyContext, &headerTexDesc);
        pipe.m_renderCrossAdapterHeader = NvFlowShareTexture2DCrossAdapter(
            renderCopyContext, pipe.m_gridCrossAdapterHeader);
    }

    for (uint32_t targetIdx = 0; targetIdx < 8; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        NvFlowContextAPI renderAPIType = NvFlowContextGetContextType(renderContext);
        NvFlowContextAPI renderCopyAPIType = NvFlowContextGetContextType(renderCopyContext);
        if (renderAPIType == renderCopyAPIType) {
            target.m_renderTextureHeader =
                NvFlowCreateTexture2D(renderContext, &headerTexDesc);
            target.m_renderCopyTextureHeader =
                NvFlowShareTexture2D(renderCopyContext, target.m_renderTextureHeader);
        } else {
            target.m_renderTextureHeader =
                NvFlowCreateTexture2DCrossAPI(renderContext, &headerTexDesc);
            target.m_renderCopyTextureHeader = NvFlowShareTexture2DCrossAPI(
                renderCopyContext, target.m_renderTextureHeader);
        }
    }

    NvFlowTexture2DDesc dataTexDesc = {};
    dataTexDesc.format = eNvFlowFormat_r16g16_float;
    dataTexDesc.width = m_dataWidth;
    dataTexDesc.height = m_dataHeight;
    for (uint32_t pipeIdx = 0; pipeIdx < 4; ++pipeIdx) {
        auto &pipe = m_pipes[pipeIdx];
        pipe.m_gridTextureData = NvFlowCreateTexture2D(gridContext, &dataTexDesc);
        pipe.m_gridCopyTextureData =
            NvFlowShareTexture2D(gridCopyContext, pipe.m_gridTextureData);
        pipe.m_gridCrossAdapterData =
            NvFlowCreateTexture2DCrossAdapter(gridCopyContext, &dataTexDesc);
        pipe.m_renderCrossAdapterData = NvFlowShareTexture2DCrossAdapter(
            renderCopyContext, pipe.m_gridCrossAdapterData);
    }

    for (uint32_t targetIdx = 0; targetIdx < 8; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        NvFlowContextAPI renderAPIType = NvFlowContextGetContextType(renderContext);
        NvFlowContextAPI renderCopyAPITye = NvFlowContextGetContextType(renderCopyContext);
        if (renderAPIType == renderCopyAPITye) {
            target.m_renderTextureData = NvFlowCreateTexture2D(renderContext, &dataTexDesc);
            target.m_renderCopyTextureData =
                NvFlowShareTexture2D(renderCopyContext, target.m_renderTextureData);
        } else {
            target.m_renderTextureData =
                NvFlowCreateTexture2DCrossAPI(renderContext, &dataTexDesc);
            target.m_renderCopyTextureData =
                NvFlowShareTexture2DCrossAPI(gridContext, target.m_renderTextureData);
        }
    }

    for (uint32_t pipeIdx = 0; pipeIdx < 4; ++pipeIdx) {
        auto &pipe = m_pipes[pipeIdx];
        pipe.m_gridEventQueue = NvFlowCreateContextEventQueue(gridContext);
        pipe.m_gridCopyEventQueue = NvFlowCreateContextEventQueue(gridCopyContext);
        pipe.m_renderCopyEventQueue = NvFlowCreateContextEventQueue(renderCopyContext);

        NvFlowFenceDesc fenceDesc = {};
        fenceDesc.crossAdapterShared = 0;
        pipe.m_gridFence = NvFlowCreateFence(gridContext, &fenceDesc);
        fenceDesc.crossAdapterShared = 1;
        pipe.m_gridCopyFence = NvFlowCreateFence(gridContext, &fenceDesc);
        pipe.m_renderCopyFence = NvFlowShareFence(renderCopyContext, pipe.m_gridCopyFence);
    }

    for (uint32_t targetIdx = 0; targetIdx < 8; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        target.m_renderEventQueue = NvFlowCreateContextEventQueue(renderContext);
    }

    auto createShader = [](NvFlowContext *context, const BYTE *cs, uint64_t cs_length,
                           const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_serializeCS =
        createShader(gridContext, NVFLOW_CREATE_SHADER_ARGS(gridViewSerializeCS));
    m_serializeHeaderCS =
        createShader(gridContext, NVFLOW_CREATE_SHADER_ARGS(gridViewSerializeHeaderCS));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(SerializeShaderParams);
    cbDesc.uploadAccess = 1;
    m_gridConstantBuffer = NvFlowCreateConstantBuffer(gridContext, &cbDesc);

    m_deserializeCS =
        createShader(renderContext, NVFLOW_CREATE_SHADER_ARGS(gridViewDeserializeCS));
    m_deserializeHeaderCS =
        createShader(renderContext, NVFLOW_CREATE_SHADER_ARGS(gridViewDeserializeHeaderCS));
    m_blockTableClearCS =
        createShader(renderContext, NVFLOW_CREATE_SHADER_ARGS(sparseClearCS));

    ZeroMemory(&cbDesc, sizeof(cbDesc));
    cbDesc.sizeInBytes = sizeof(SerializeShaderParams);
    cbDesc.uploadAccess = 1;
    m_renderConstantBuffer = NvFlowCreateConstantBuffer(renderContext, &cbDesc);

    NvFlowTexture3DDesc blockTableDesc = {};
    blockTableDesc.format = eNvFlowFormat_r32_uint;
    blockTableDesc.dim = blockTableDim;
    blockTableDesc.uploadAccess = 0;
    blockTableDesc.downloadAccess = 0;
    m_vBlockIdxToBlockID = NvFlowCreateTexture3D(renderContext, &blockTableDesc);

    NvFlowGridImportDesc importDesc = {};
    importDesc.gridExport = desc->gridExport;
    m_gridImport = NvFlowCreateGridImport(renderContext, &importDesc);

    for (uint32_t targetIdx = 0; targetIdx < 8; ++targetIdx) {
        auto &target = m_targets[targetIdx];
        target.m_gridImportStateCPU = NvFlowCreateGridImportStateCPU(m_gridImport);
    }
}

GridProxyMultiGPU::~GridProxyMultiGPU() {
    for (auto &target : m_targets) {
        SafeRelease(target.m_renderCopyTextureHeader);
        SafeRelease(target.m_renderCopyTextureData);
        SafeRelease(target.m_renderTextureHeader);
        SafeRelease(target.m_renderTextureData);
        SafeRelease(target.m_renderEventQueue);
        SafeRelease(target.m_gridImportStateCPU);
    }

    for (auto &pipe : m_pipes) {
        SafeRelease(pipe.m_gridTextureHeader);
        SafeRelease(pipe.m_gridCopyTextureHeader);
        SafeRelease(pipe.m_gridCrossAdapterHeader);
        SafeRelease(pipe.m_renderCrossAdapterHeader);
        SafeRelease(pipe.m_gridTextureData);
        SafeRelease(pipe.m_gridCopyTextureData);
        SafeRelease(pipe.m_gridCrossAdapterData);
        SafeRelease(pipe.m_renderCrossAdapterData);
        SafeRelease(pipe.m_gridEventQueue);
        SafeRelease(pipe.m_gridCopyEventQueue);
        SafeRelease(pipe.m_renderCopyEventQueue);
        SafeRelease(pipe.m_gridFence);
        SafeRelease(pipe.m_gridCopyFence);
        SafeRelease(pipe.m_renderCopyFence);
    }

    SafeRelease(m_serializeCS);
    SafeRelease(m_serializeHeaderCS);
    SafeRelease(m_gridConstantBuffer);
    SafeRelease(m_deserializeCS);
    SafeRelease(m_deserializeHeaderCS);
    SafeRelease(m_blockTableClearCS);
    SafeRelease(m_renderConstantBuffer);
    SafeRelease(m_vBlockIdxToBlockID);
    SafeRelease(m_gridImport);
}

GridProxyMultiGPU::Target::Target() {
    m_pushID = 0;
    m_state = 0;
    m_eventID = 0;
    m_minHeaderHeight = 0;
    m_minDataHeight = 0;
    pipeIdx = -1;
    m_gridImportStateCPU = 0;
    m_renderCopyTextureHeader = 0;
    m_renderCopyTextureData = 0;
    m_renderTextureHeader = 0;
    m_renderTextureData = 0;
    m_renderEventQueue = 0;
}

GridProxyMultiGPU::Pipe::Pipe() {
    m_state = 0;
    m_eventID = 0;
    m_gridTextureHeader = 0;
    m_gridCopyTextureHeader = 0;
    m_gridCrossAdapterHeader = 0;
    m_renderCrossAdapterHeader = 0;
    m_gridTextureData = 0;
    m_gridCopyTextureData = 0;
    m_gridCrossAdapterData = 0;
    m_renderCrossAdapterData = 0;
    m_gridEventQueue = 0;
    m_gridCopyEventQueue = 0;
    m_renderCopyEventQueue = 0;
    m_gridFence = 0;
    m_gridCopyFence = 0;
    m_renderCopyFence = 0;
    m_gridFenceValue = 0;
    m_sharedFenceValue = 0;
}

NvFlowGridProxy *FlowCreateGridProxy(const NvFlowGridProxyDesc *desc) {
    if (desc->proxyType == eNvFlowGridProxyTypeMultiGPU) {
        return new GridProxyMultiGPU(desc);
    } else if (desc->proxyType == eNvFlowGridProxyTypeInterQueue) {
        auto gridAPI = NvFlowContextGetContextType(desc->gridContext);
        auto renderAPI = NvFlowContextGetContextType(desc->renderContext);
        if (gridAPI == renderAPI) {
            return new GridProxyInterQueueCommonMemory(desc);
        } else {
            return new GridProxyInterQueueGPU(desc);
        }
    } else {
        return new GridProxySingleGPU(desc);
    }
}

}  // namespace NvFlow