#include "ParticleSurfaceExport.h"
#include "NvFlowContextImpl.h"
#include "ParticleSurface.h"

namespace NvFlow {

void ParticleSurfaceExport::getDebugVisView(NvFlowGridExportDebugVisView* view) {
    if (view) {
        view->debugVisFlags = eNvFlowGridDebugVisBlocks;
        view->bounds = 0;
        view->numBounds = 0;
        view->spheres = 0;
        view->numSpheres = 0;
        view->capsules = 0;
        view->numCapsules = 0;
        view->boxes = 0;
        view->numBoxes = 0;
    }
}

ParticleSurfaceExport::ParticleSurfaceExport(ParticleSurface* particleSurface)
    : m_particleSurface{particleSurface} {}

ParticleSurfaceExport::~ParticleSurfaceExport() {}

uint32_t ParticleSurfaceExport::addRef() {
    return 1;
}

uint32_t ParticleSurfaceExport::release() {
    return 1;
}

uint64_t ParticleSurfaceExport::getGPUBytesUsed() {
    return 0;
}

NvFlowGridExportHandle ParticleSurfaceExport::getHandle(NvFlowContext* context,
                                                        NvFlowGridTextureChannel channel) {
    NvFlowGridExportHandle result;
    result.gridExport = this;
    result.channel = eNvFlowGridTextureChannelDensity;
    result.numLayerViews = 1;
    return result;
}

void ParticleSurfaceExport::getLayerView(NvFlowGridExportHandle handle, uint32_t layerIdx,
                                         NvFlowGridExportLayerView* layerView) {
    if (layerView) {
        layerView->data = NvFlowTexture3DGetResource(m_particleSurface->m_debugVisTex);
        ZeroMemory(&layerView->mapping, sizeof(layerView->mapping));
        layerView->mapping.blockTable =
            NvFlowTexture3DGetResource(m_particleSurface->m_blockTable);
        layerView->mapping.blockList =
            NvFlowBufferGetResource(m_particleSurface->m_blockList);
        layerView->mapping.numBlocks = m_particleSurface->m_blockConfig.maxBlocks;
    }
}

void ParticleSurfaceExport::getLayeredView(NvFlowGridExportHandle handle,
                                           NvFlowGridExportLayeredView* layeredView) {
    if (layeredView) {
        layeredView->mapping.shaderParams = m_particleSurface->m_linearParams;
        layeredView->mapping.maxBlocks = m_particleSurface->m_blockConfig.maxBlocks;
        layeredView->mapping.layeredBlockListCPU = 0;
        layeredView->mapping.layeredNumBlocks = 0;
        layeredView->mapping.modelMatrix = m_particleSurface->m_modelMatrix;
    }
}

}  // namespace NvFlow