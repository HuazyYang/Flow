#include "GridExport.h"
#include "Grid.h"
#include "GridImport.h"
#include "NvFlowContextImpl.h"

namespace NvFlow {

///
/// GridExportGrid implement
///

NvFlowGridExportHandle GridExportGrid::getHandle(NvFlowContext* context,
                                                 NvFlowGridTextureChannel channel) {
    NvFlowGridExportHandle exportHandle;
    exportHandle.gridExport = 0;
    exportHandle.channel = eNvFlowGridTextureChannelVelocity;
    exportHandle.numLayerViews = 0;
    updateChannel(context, channel);

    if (channel < eNvFlowGridTextureChannelCount) {
        exportHandle = m_channels[channel].exportHandle;
    }
    return exportHandle;
}

void GridExportGrid::getLayerView(NvFlowGridExportHandle handle, uint32_t layerIdx,
                                  NvFlowGridExportLayerView* view) {
    if (handle.channel < eNvFlowGridTextureChannelCount) {
        auto& perChannel = m_channels[handle.channel];
        if (layerIdx < perChannel.exportLayerViews.size())
            *view = perChannel.exportLayerViews[layerIdx];
    }
}

void GridExportGrid::getLayeredView(NvFlowGridExportHandle handle,
                                    NvFlowGridExportLayeredView* layeredView) {
    if (handle.channel < eNvFlowGridTextureChannelCount) {
        *layeredView = m_channels[handle.channel].exportLayeredView;
    }
}

void GridExportGrid::getDebugVisView(NvFlowGridExportDebugVisView* debugView) {
    memcpy(debugView, &m_debugVis, sizeof(m_debugVis));
}

GridExportGrid::GridExportGrid() : m_channels{}, m_debugVis{} {}

uint32_t GridExportGridImpl::addRef() {
    return 1;
}

uint32_t GridExportGridImpl::release() {
    return 1;
}

uint64_t GridExportGridImpl::getGPUBytesUsed() {
    return 0;
}

void GridExportGridImpl::updateChannel(NvFlowContext* context,
                                       NvFlowGridTextureChannel channel) {
    if (channel < eNvFlowGridTextureChannelCount) {
        auto sparseTexture = &m_grid->m_velocity;
        if (channel == eNvFlowGridTextureChannelDensity)
            sparseTexture = &m_grid->m_density;
        else if (channel == eNvFlowGridTextureChannelDensityCoarse)
            sparseTexture = &m_grid->m_densityCoarse;

        auto readHandle = sparseTexture->front.readLinearHandle(context);
        auto readLayeredView = readHandle.layeredView();
        auto& layeredBlockList = readLayeredView.layeredBlockList;
        auto& perChannel = m_channels[channel];
        perChannel.exportHandle.gridExport = this;
        perChannel.exportHandle.channel = channel;
        perChannel.exportHandle.numLayerViews = readHandle.numLayers;
        perChannel.exportLayeredView.mapping.shaderParams = readLayeredView.params;

        auto config = sparseTexture->getConfig();
        perChannel.exportLayeredView.mapping.maxBlocks = config.maxBlocks;
        perChannel.exportLayeredView.mapping.layeredBlockListCPU = layeredBlockList.dataCPU;
        perChannel.exportLayeredView.mapping.layeredNumBlocks = layeredBlockList.numBlocks;
        perChannel.exportLayeredView.mapping.modelMatrix = m_grid->m_currentModelMatrix;

        while (readHandle.numLayers > perChannel.exportLayerViews.size())
            perChannel.exportLayerViews.allocateBack();

        for (uint32_t layerIdx = 0; layerIdx < readHandle.numLayers; ++layerIdx) {
            auto& layer = perChannel.exportLayerViews[layerIdx];
            auto readLayerView = readHandle.layerView(layerIdx);
            layer.data = readLayerView.data;
            auto& MaterialHandleFromLayerIdx =
                m_grid->getMaterialHandleFromLayerIdx(layerIdx);
            layer.mapping.material = MaterialHandleFromLayerIdx;
            layer.mapping.blockTable = readLayerView.mapping.blockTable;
            layer.mapping.blockList = readLayerView.mapping.blockList;
            layer.mapping.numBlocks = readLayerView.mapping.numBlocks;
        }

        auto& src = m_grid->m_debugVis;
        m_debugVis.bounds = src.m_bounds.data();
        m_debugVis.numBounds = src.m_bounds.size();
        m_debugVis.spheres = src.m_spheres.data();
        m_debugVis.numSpheres = src.m_spheres.size();
        m_debugVis.capsules = src.m_capsules.data();
        m_debugVis.numCapsules = src.m_capsules.size();
        m_debugVis.boxes = src.m_boxes.data();
        m_debugVis.numBoxes = src.m_boxes.size();
    }
}

GridExportGridImpl::GridExportGridImpl(Grid* grid) {
    m_grid = grid;
}

GridExportGridImpl::~GridExportGridImpl() {}

uint32_t GridExportGridImport::addRef() {
    return 1;
}

uint32_t GridExportGridImport::release() {
    return 1;
}

uint64_t GridExportGridImport::getGPUBytesUsed() {
    return 0;
}

void GridExportGridImport::updateChannel(NvFlowContext* context,
                                         NvFlowGridTextureChannel channel) {
    if (channel < eNvFlowGridTextureChannelCount) {
        auto& importPerChannel = gridImport->m_channels[channel];
        auto& perChannel = m_channels[channel];
        if (importPerChannel.enabled) {
            perChannel.exportHandle.gridExport = this;
            perChannel.exportHandle.channel = channel;
            perChannel.exportHandle.numLayerViews =
                importPerChannel.importHandle.numLayerViews;

            CopyMemory(&perChannel.exportLayeredView, &importPerChannel.importLayeredView,
                       sizeof(perChannel.exportLayeredView));
            while (perChannel.exportHandle.numLayerViews >
                   perChannel.exportLayerViews.size())
                perChannel.exportLayerViews.allocateBack();

            for (uint32_t layerIdx = 0; layerIdx < perChannel.exportHandle.numLayerViews;
                 ++layerIdx) {
                auto& layer = perChannel.exportLayerViews[layerIdx];
                auto& importLayer = importPerChannel.importLayerViews[layerIdx];
                layer.mapping = importLayer.mapping;
                if (importPerChannel.standaloneMode) {
                    if (layerIdx >= importPerChannel.perLayer.size()) {
                        layer.mapping.blockList = 0;
                        layer.mapping.blockTable = 0;
                    } else {
                        auto& perLayer = importPerChannel.perLayer[layerIdx];
                        layer.mapping.blockTable =
                            NvFlowTexture3DGetResource(perLayer.standaloneBlockTable);
                        layer.mapping.blockList =
                            NvFlowBufferGetResource(perLayer.standaloneBlockList);
                    }
                }
                layer.data = NvFlowTexture3DGetResource(importPerChannel.linearTexture);
            }
        } else {
            perChannel.exportHandle.gridExport = this;
            perChannel.exportHandle.channel = channel;
            perChannel.exportHandle.numLayerViews = 0;
            CopyMemory(&perChannel.exportLayeredView, &importPerChannel.importLayeredView,
                       sizeof(perChannel.exportLayeredView));
            perChannel.exportLayeredView.mapping.layeredBlockListCPU = 0;
            perChannel.exportLayeredView.mapping.layeredNumBlocks = 0;
        }
    }
}

GridExportGridImport::GridExportGridImport(GridImport* gridImport)
    : gridImport{gridImport} {}

GridExportGridImport::~GridExportGridImport() {}

}  // namespace NvFlow