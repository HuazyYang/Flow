#ifndef NVFLOW_GRIDIMPORT_H
#define NVFLOW_GRIDIMPORT_H
#include "GridExport.h"
#include "Object.h"
#include "NvFlowContextImpl.h"

struct NvFlowGridImportStateCPU : NvFlowObject {
    virtual void updateStateCPU(NvFlowContext *context, NvFlowGridExport *gridExport) = 0;
};

struct NvFlowGridImport : NvFlowObject {
    virtual NvFlowGridImportHandle getHandle(NvFlowContext *context,
                                             const NvFlowGridImportParams *params) = 0;

    virtual void getLayerView(NvFlowGridImportHandle importHandle, uint32_t layerIdx,
                              NvFlowGridImportLayerView *layerView) = 0;

    virtual void getLayeredView(NvFlowGridImportHandle handle,
                                NvFlowGridImportLayeredView *view) = 0;

    virtual void releaseChannel(NvFlowContext *context,
                                NvFlowGridTextureChannel channel) = 0;

    virtual NvFlowGridExport *getGridExport(NvFlowContext *context) = 0;

    virtual NvFlowGridImportStateCPU *createImportStateCPU() = 0;

    virtual NvFlowGridImportHandle stateCPUGetHandle(
        NvFlowContext *context, const NvFlowGridImportStateCPUParams *params) = 0;
};

namespace NvFlow {

NvFlowGridImport *FlowCreateGridImport(NvFlowContext *context,
                                       const NvFlowGridImportDesc *desc);

struct GridImport : Object, NvFlowGridImport {
    uint32_t addRef() override;
    uint32_t release() override;

    uint64_t getGPUBytesUsed() override;

    NvFlowGridImportHandle getHandle(NvFlowContext *context,
                                     const NvFlowGridImportParams *params) override;

    void getLayerView(NvFlowGridImportHandle importHandle, uint32_t layerIdx,
                      NvFlowGridImportLayerView *layerView) override;

    void getLayeredView(NvFlowGridImportHandle handle,
                        NvFlowGridImportLayeredView *view) override;

    void releaseChannel(NvFlowContext *context, NvFlowGridTextureChannel channel) override;

    NvFlowGridExport *getGridExport(NvFlowContext *context) override;

    NvFlowGridImportStateCPU *createImportStateCPU() override;

    NvFlowGridImportHandle stateCPUGetHandle(
        NvFlowContext *context, const NvFlowGridImportStateCPUParams *params) override;

    // Details
    struct PerChannel {
        struct PerLayer {
            NvFlowTexture3D *standaloneBlockTable;
            NvFlowBuffer *standaloneBlockList;
        };

        bool enabled;
        bool standaloneMode;
        bool linearDirty;
        NvFlowGridImportMode importMode;
        NvFlowTexture3D *pointTexture;
        NvFlowTexture3D *linearTexture;
        VectorCached<PerLayer, 8> perLayer;
        NvFlowGridImportHandle importHandle;
        NvFlowGridImportLayeredView importLayeredView;
        VectorCached<NvFlowGridImportLayerView, 8> importLayerViews;
    };

    void allocateResources(PerChannel *perChannel, NvFlowContext *context,
                           uint32_t exportNumLayers,
                           NvFlowGridExportLayeredView *exportLayeredView);

    NvFlowDim computeTextureDim(const NvFlowShaderLinearParams *shaderParams);

    void overrideSST(NvFlowShaderLinearParams *shaderParams);

    GridImport(NvFlowContext *context, const NvFlowGridImportDesc *desc);
    ~GridImport();

    GridExportGridImport m_gridExport;
    PerChannel m_channels[eNvFlowGridTextureChannelCount];
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowComputeShader *m_updateLinearCS;
};
}  // namespace NvFlow

#endif /* NVFLOW_GRIDIMPORT_H */
