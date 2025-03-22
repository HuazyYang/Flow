#ifndef NVFLOW_PARTICLESURFACEEXPORT_H
#define NVFLOW_PARTICLESURFACEEXPORT_H
#include "GridExport.h"

namespace NvFlow {

struct ParticleSurface;

struct ParticleSurfaceExport : NvFlowGridExport {
    uint32_t addRef() override;

    uint32_t release() override;

    uint64_t getGPUBytesUsed() override;

    NvFlowGridExportHandle getHandle(NvFlowContext *context,
                                     NvFlowGridTextureChannel channel) override;
    void getLayerView(NvFlowGridExportHandle handle, uint32_t layerIdx,
                      NvFlowGridExportLayerView *layerView) override;
    void getLayeredView(NvFlowGridExportHandle handle,
                        NvFlowGridExportLayeredView *layeredView) override;

    void getDebugVisView(NvFlowGridExportDebugVisView *view) override;

    // Details
    ParticleSurfaceExport(ParticleSurface *particleSurface);
    ~ParticleSurfaceExport();

    ParticleSurface *m_particleSurface;
};

}  // namespace NvFlow

#endif /* NVFLOW_PARTICLESURFACEEXPORT_H */
