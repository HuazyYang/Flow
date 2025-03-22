#include "GridImport.h"
#include "Object.h"
#include "ClientHelper.h"
#include "NvFlowContextImpl.h"

namespace NvFlow {

struct GridImportStateCPU : Object, NvFlowGridImportStateCPU {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void updateStateCPU(NvFlowContext *context, NvFlowGridExport *gridExport) override;

    // Details

    struct PerChannel {
        NvFlowGridExportHandle exportHandle;
        NvFlowGridExportLayeredView exportLayeredView;
        VectorCached<NvFlowGridExportLayerView, 8> exportLayerViews;
        VectorCached<NvFlowUint2, 1> layeredBlockList;
    };

    GridImportStateCPU();
    ~GridImportStateCPU();

    PerChannel m_channels[eNvFlowGridTextureChannelCount];
    NvFlowGridExportDebugVisView m_debugVis;
};

uint64_t GridImportStateCPU::getGPUBytesUsed() {
    return 0;
}

void GridImportStateCPU::updateStateCPU(NvFlowContext *context,
                                        NvFlowGridExport *gridExport) {
    for (uint32_t channelIdx = 0; channelIdx < eNvFlowGridTextureChannelCount;
         ++channelIdx) {
        auto &perChannel = m_channels[channelIdx];
        perChannel.exportLayerViews.clear();
        perChannel.exportHandle = NvFlowGridExportGetHandle(
            gridExport, context, (NvFlowGridTextureChannel)channelIdx);
        perChannel.exportLayeredView = {};
        NvFlowGridExportGetLayeredView(perChannel.exportHandle,
                                       &perChannel.exportLayeredView);

        perChannel.exportLayerViews.clear();
        for (uint32_t layerIdx = 0; layerIdx < perChannel.exportHandle.numLayerViews;
             ++layerIdx) {
            uint32_t allocIdx = perChannel.exportLayerViews.allocateBack();
            auto &layerView = perChannel.exportLayerViews[allocIdx];
            NvFlowGridExportGetLayerView(perChannel.exportHandle, layerIdx, &layerView);
        }

        perChannel.layeredBlockList.resize(perChannel.exportLayeredView.mapping.maxBlocks);
        for (uint32_t idx = 0; idx < perChannel.layeredBlockList.size(); ++idx) {
            perChannel.layeredBlockList[idx] =
                perChannel.exportLayeredView.mapping.layeredBlockListCPU[idx];
        }
        perChannel.exportLayeredView.mapping.layeredBlockListCPU =
            perChannel.layeredBlockList.data();
    }

    gridExport->getDebugVisView(&m_debugVis);
    m_debugVis.bounds = 0;
    m_debugVis.numBounds = 0;
    m_debugVis.spheres = 0;
    m_debugVis.numSpheres = 0;
    m_debugVis.capsules = 0;
    m_debugVis.numCapsules = 0;
    m_debugVis.boxes = 0;
    m_debugVis.numBoxes = 0;
}

GridImportStateCPU::GridImportStateCPU() {}

GridImportStateCPU::~GridImportStateCPU() {}

#include "updateLinearCS.hlsl.h"
#include "NvFlowShader.h"

struct UpdateLinearShaderParams {
    NvFlowShaderLinearParams params;
};

void GridImport::allocateResources(PerChannel *perChannel, NvFlowContext *context,
                                   uint32_t exportNumLayers,
                                   NvFlowGridExportLayeredView *exportLayeredView) {
    if (perChannel->importMode == eNvFlowGridImportModePoint && !perChannel->pointTexture) {
        NvFlowTexture3DDesc valueLinearDesc = {};
        valueLinearDesc.format = eNvFlowFormat_r16g16b16a16_float;
        valueLinearDesc.dim = computeTextureDim(&exportLayeredView->mapping.shaderParams);
        valueLinearDesc.uploadAccess = 0;
        valueLinearDesc.downloadAccess = 0;
        perChannel->pointTexture = NvFlowCreateTexture3D(context, &valueLinearDesc);
    }

    if (perChannel->importMode <= eNvFlowGridImportModeLinear &&
        !perChannel->linearTexture) {
        NvFlowTexture3DDesc desc = {};
        desc.format = eNvFlowFormat_r16g16b16a16_float;
        desc.dim = computeTextureDim(&exportLayeredView->mapping.shaderParams);
        desc.uploadAccess = 0;
        desc.downloadAccess = 0;
        perChannel->linearTexture = NvFlowCreateTexture3D(context, &desc);
    }

    if (perChannel->standaloneMode) {
        while (perChannel->perLayer.size() < exportNumLayers) {
            uint32_t layerIdx = perChannel->perLayer.allocateBack();
            auto &layer = perChannel->perLayer[layerIdx];
            layer.standaloneBlockTable = 0;
            layer.standaloneBlockList = 0;
        }

        for (uint32_t idx = 0; idx < exportNumLayers; ++idx) {
            auto &perLayer = perChannel->perLayer[idx];
            if (!perLayer.standaloneBlockTable) {
                NvFlowTexture3DDesc blockTableDesc = {};
                blockTableDesc.format = eNvFlowFormat_r32_uint;
                blockTableDesc.dim =
                    (const NvFlowDim &)(exportLayeredView->mapping.shaderParams.gridDim);
                blockTableDesc.uploadAccess = 0;
                blockTableDesc.downloadAccess = 0;
                perLayer.standaloneBlockTable =
                    NvFlowCreateTexture3D(context, &blockTableDesc);
            }

            if (!perLayer.standaloneBlockList) {
                NvFlowBufferDesc blockListDesc = {};
                blockListDesc.format = eNvFlowFormat_r32_uint;
                blockListDesc.dim = exportLayeredView->mapping.maxBlocks;
                blockListDesc.uploadAccess = 0;
                blockListDesc.downloadAccess = 0;
                perLayer.standaloneBlockList = NvFlowCreateBuffer(context, &blockListDesc);
            }
        }
    }
}

NvFlowDim GridImport::computeTextureDim(const NvFlowShaderLinearParams *shaderParams) {
    NvFlowDim texDim;
    texDim = (const NvFlowDim &)shaderParams->poolGridDim *
             ((const NvFlowDim &)shaderParams->blockDim + 2);
    return texDim;
}

uint32_t GridImport::addRef() {
    return Object::addRef();
}

uint32_t GridImport::release() {
    return Object::release();
}

uint64_t GridImport::getGPUBytesUsed() {
    return 0;
}

NvFlowGridImportHandle GridImport::getHandle(NvFlowContext *context,
                                             const NvFlowGridImportParams *params) {
    NvFlowGridImportHandle importHandle;
    importHandle.gridImport = 0;
    importHandle.channel = eNvFlowGridTextureChannelVelocity;
    importHandle.numLayerViews = 0;

    if (params->channel < (uint32_t)eNvFlowGridTextureChannelCount) {
        auto &perChannel = m_channels[params->channel];
        perChannel.enabled = 1;
        perChannel.standaloneMode = 0;
        perChannel.linearDirty = 1;
        perChannel.importMode = params->importMode;

        auto exportHandle =
            NvFlowGridExportGetHandle(params->gridExport, context, params->channel);
        NvFlowGridExportLayeredView exportLayeredView = {};
        NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);

        if (exportHandle.numLayerViews)
            allocateResources(&perChannel, context, exportHandle.numLayerViews,
                              &exportLayeredView);

        perChannel.importHandle.gridImport = this;
        perChannel.importHandle.channel = params->channel;
        perChannel.importHandle.numLayerViews = exportHandle.numLayerViews;
        CopyMemory(&perChannel.importLayeredView, &exportLayeredView,
                   sizeof(perChannel.importLayeredView));

        while (exportHandle.numLayerViews > perChannel.importLayerViews.size())
            perChannel.importLayerViews.allocateBack();

        for (uint32_t layerIdx = 0; layerIdx < exportHandle.numLayerViews; ++layerIdx) {
            NvFlowGridExportLayerView exportLayerView = {};
            NvFlowGridExportGetLayerView(exportHandle, layerIdx, &exportLayerView);
            auto &layer = perChannel.importLayerViews[layerIdx];
            layer.mapping = exportLayerView.mapping;
            if (params->importMode == eNvFlowGridImportModeLinear)
                layer.dataRW = NvFlowTexture3DGetResourceRW(perChannel.linearTexture);
            else
                layer.dataRW = NvFlowTexture3DGetResourceRW(perChannel.pointTexture);
        }

        overrideSST(&perChannel.importLayeredView.mapping.shaderParams);
        importHandle = perChannel.importHandle;
    }

    params->gridExport->getDebugVisView(&m_gridExport.m_debugVis);

    return importHandle;
}

void GridImport::getLayerView(NvFlowGridImportHandle importHandle, uint32_t layerIdx,
                              NvFlowGridImportLayerView *layerView) {
    if (importHandle.channel < (uint32_t)eNvFlowGridTextureChannelCount) {
        auto &perChannel = m_channels[importHandle.channel];
        if (layerIdx < perChannel.importLayerViews.size())
            *layerView = perChannel.importLayerViews[layerIdx];
    }
}

void GridImport::getLayeredView(NvFlowGridImportHandle handle,
                                NvFlowGridImportLayeredView *view) {
    if (handle.channel < (uint32_t)eNvFlowGridTextureChannelCount)
        *view = m_channels[handle.channel].importLayeredView;
}

void GridImport::releaseChannel(NvFlowContext *context, NvFlowGridTextureChannel channel) {
    if (channel < (uint32_t)eNvFlowGridTextureChannelCount)
        m_channels[channel].enabled = 0;
}

NvFlowGridExport *GridImport::getGridExport(NvFlowContext *context) {
    for (int channelIdx = 0; channelIdx < eNvFlowGridTextureChannelCount; ++channelIdx) {
        auto &perChannel = m_channels[channelIdx];
        if (perChannel.enabled && perChannel.importMode == eNvFlowGridImportModePoint &&
            perChannel.linearDirty) {
            auto &layered = perChannel.importLayeredView;

            auto params = (UpdateLinearShaderParams *)NvFlowConstantBufferMap(
                context, m_constantBuffer);
            params->params = layered.mapping.shaderParams;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);

            for (uint32_t layerIdx = 0; layerIdx < perChannel.importHandle.numLayerViews;
                 ++layerIdx) {
                auto &layer = perChannel.importLayerViews[layerIdx];

                NvFlowDispatchParams dparams = {};
                dparams.shader = m_updateLinearCS;
                dparams.gridDim.x =
                    (layered.mapping.shaderParams.linearBlockDim.w + 127) / 0x80;
                dparams.gridDim.y = layer.mapping.numBlocks;
                dparams.gridDim.z = 1;
                dparams.rootConstantBuffer = m_constantBuffer;
                dparams.readOnly[0] = layer.mapping.blockList;
                dparams.readOnly[1] = NvFlowTexture3DGetResource(perChannel.pointTexture);
                dparams.readOnly[2] = layer.mapping.blockTable;
                dparams.readWrite[0] =
                    NvFlowTexture3DGetResourceRW(perChannel.linearTexture);
                NvFlowContextDispatch(context, &dparams);
            }
            perChannel.linearDirty = 0;
        }
    }

    return &m_gridExport;
}

NvFlowGridImportStateCPU *GridImport::createImportStateCPU() {
    return new GridImportStateCPU();
}

NvFlowGridImportHandle GridImport::stateCPUGetHandle(
    NvFlowContext *context, const NvFlowGridImportStateCPUParams *params) {
    NvFlowGridImportHandle importHandle;
    importHandle.gridImport = 0;
    importHandle.channel = eNvFlowGridTextureChannelVelocity;
    importHandle.numLayerViews = 0;

    GridImportStateCPU *stateCPU = implCast<GridImportStateCPU>(params->stateCPU);

    if (params->channel < (uint32_t)eNvFlowGridTextureChannelCount) {
        auto &perChannel = m_channels[params->channel];
        perChannel.enabled = 1;
        perChannel.standaloneMode = 1;
        perChannel.linearDirty = 1;
        perChannel.importMode = params->importMode;

        auto &exportHandle = stateCPU->m_channels[params->channel].exportHandle;
        auto &exportLayeredView = stateCPU->m_channels[params->channel].exportLayeredView;
        if (exportHandle.numLayerViews)
            allocateResources(&perChannel, context, exportHandle.numLayerViews,
                              &exportLayeredView);

        perChannel.importHandle.gridImport = this;
        perChannel.importHandle.channel = params->channel;
        perChannel.importHandle.numLayerViews = exportHandle.numLayerViews;
        perChannel.importLayeredView.mapping = exportLayeredView.mapping;

        while (exportHandle.numLayerViews > perChannel.importLayerViews.size())
            perChannel.importLayerViews.allocateBack();

        for (uint32_t layerIdx = 0; layerIdx < exportHandle.numLayerViews; ++layerIdx) {
            auto &exportLayerView =
                stateCPU->m_channels[params->channel].exportLayerViews[layerIdx];
            auto &layer = perChannel.importLayerViews[layerIdx];
            layer.mapping = exportLayerView.mapping;

            auto &perLayer = perChannel.perLayer[layerIdx];
            layer.mapping.blockTable =
                NvFlowTexture3DGetResource(perLayer.standaloneBlockTable);
            layer.mapping.blockList = NvFlowBufferGetResource(perLayer.standaloneBlockList);
            layer.blockTableRW =
                NvFlowTexture3DGetResourceRW(perLayer.standaloneBlockTable);
            layer.blockListRW = NvFlowBufferGetResourceRW(perLayer.standaloneBlockList);
            if (params->importMode == eNvFlowGridImportModeLinear)
                layer.dataRW = NvFlowTexture3DGetResourceRW(perChannel.linearTexture);
            else
                layer.dataRW = NvFlowTexture3DGetResourceRW(perChannel.pointTexture);
        }

        overrideSST(&perChannel.importLayeredView.mapping.shaderParams);
        importHandle = perChannel.importHandle;
    }

    m_gridExport.m_debugVis = stateCPU->m_debugVis;

    return importHandle;
}

void GridImport::overrideSST(NvFlowShaderLinearParams *shaderParams) {
    NvFlowDim texDim = computeTextureDim(shaderParams);
    shaderParams->linearBlockDim.x = shaderParams->blockDim.x + 2;
    shaderParams->linearBlockDim.y = shaderParams->blockDim.y + 2;
    shaderParams->linearBlockDim.z = shaderParams->blockDim.z + 2;
    shaderParams->linearBlockDim.w = shaderParams->linearBlockDim.z *
                                     shaderParams->linearBlockDim.y *
                                     shaderParams->linearBlockDim.x;

    shaderParams->linearBlockOffset = make_uint4(1, 1, 1, 0);
    shaderParams->isVTR = make_uint4(0);
    shaderParams->dimInv.x = 1.f / texDim.x;
    shaderParams->dimInv.y = 1.f / texDim.y;
    shaderParams->dimInv.z = 1.f / texDim.z;
    shaderParams->dimInv.w =
        shaderParams->dimInv.x * shaderParams->dimInv.y * shaderParams->dimInv.z;
}

GridImport::GridImport(NvFlowContext *context, const NvFlowGridImportDesc *desc)
    : m_gridExport(this), m_channels{}, m_constantBuffer(0), m_updateLinearCS(0) {
    NvFlowComputeShaderDesc shaderDesc = {};
    shaderDesc.cs = g_updateLinearCS;
    shaderDesc.cs_length = sizeof(g_updateLinearCS);
    shaderDesc.label = L"updateLinearCS";
    m_updateLinearCS = NvFlowCreateComputeShader(context, &shaderDesc);

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(UpdateLinearShaderParams);
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);

    for (uint32_t channelIdx = 0; channelIdx < eNvFlowGridTextureChannelCount;
         ++channelIdx) {
        auto &perChannel = m_channels[channelIdx];
        auto exportHandle = NvFlowGridExportGetHandle(desc->gridExport, context,
                                                      (NvFlowGridTextureChannel)channelIdx);
        NvFlowGridExportLayeredView exportLayeredView = {};
        NvFlowGridExportGetLayeredView(exportHandle, &exportLayeredView);
        CopyMemory(&perChannel.importLayeredView, &exportLayeredView,
                   sizeof(perChannel.importLayeredView));
    }
}

GridImport::~GridImport() {
    for (uint32_t channelIdx = 0; channelIdx < eNvFlowGridTextureChannelCount;
         ++channelIdx) {
        auto &perChannel = m_channels[channelIdx];
        SafeRelease(perChannel.pointTexture);
        SafeRelease(perChannel.linearTexture);
        for (uint32_t layerIdx = 0; layerIdx < perChannel.perLayer.size(); ++layerIdx) {
            auto &layer = perChannel.perLayer[layerIdx];
            SafeRelease(layer.standaloneBlockTable);
            SafeRelease(layer.standaloneBlockList);
        }
    }

    SafeRelease(m_constantBuffer);
    SafeRelease(m_updateLinearCS);
}

NvFlowGridImport *FlowCreateGridImport(NvFlowContext *context,
                                       const NvFlowGridImportDesc *desc) {
    return new GridImport(context, desc);
}

}  // namespace NvFlow