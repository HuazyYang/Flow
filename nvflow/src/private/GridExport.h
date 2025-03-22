#ifndef NVFLOW_GRIDEXPORTGRID_H
#define NVFLOW_GRIDEXPORTGRID_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"
#include "VectorCached.h"

struct NvFlowGridExport : NvFlowObject {
    virtual NvFlowGridExportHandle getHandle(NvFlowContext *context,
                                             NvFlowGridTextureChannel channel) = 0;
    virtual void getLayerView(NvFlowGridExportHandle handle, uint32_t layerIdx,
                              NvFlowGridExportLayerView *view) = 0;
    virtual void getLayeredView(NvFlowGridExportHandle handle,
                                NvFlowGridExportLayeredView *layeredView) = 0;

    virtual void getDebugVisView(NvFlowGridExportDebugVisView *debugView) = 0;
};

namespace NvFlow {

struct GridExportGrid : NvFlowGridExport {
    NvFlowGridExportHandle getHandle(NvFlowContext *context,
                                     NvFlowGridTextureChannel channel) override;
    void getLayerView(NvFlowGridExportHandle handle, uint32_t layerIdx,
                      NvFlowGridExportLayerView *view) override;
    void getLayeredView(NvFlowGridExportHandle handle,
                        NvFlowGridExportLayeredView *layeredView) override;

    void getDebugVisView(NvFlowGridExportDebugVisView *debugView) override;

    virtual void updateChannel(NvFlowContext *context,
                               NvFlowGridTextureChannel texChannel) = 0;

    GridExportGrid();

    struct PerChannel {
        NvFlowGridExportHandle exportHandle;
        NvFlowGridExportLayeredView exportLayeredView;
        VectorCached<NvFlowGridExportLayerView, 8> exportLayerViews;
    };

    PerChannel m_channels[3];
    NvFlowGridExportDebugVisView m_debugVis;
};

struct Grid;

///
/// GridExportGridImpl
///
struct GridExportGridImpl : GridExportGrid {
    uint32_t addRef() override;

    uint32_t release() override;

    uint64_t getGPUBytesUsed() override;

    void updateChannel(NvFlowContext *context,
                       NvFlowGridTextureChannel texChannel) override;

    // Details
    GridExportGridImpl(Grid *grid);
    ~GridExportGridImpl();

    Grid *m_grid;
};

struct GridImport;

///
/// GridExportGridImport
///
struct GridExportGridImport : GridExportGrid {
    uint32_t addRef() override;
    uint32_t release() override;
    uint64_t getGPUBytesUsed() override;

    void updateChannel(NvFlowContext *context,
                       NvFlowGridTextureChannel texChannel) override;

    // Details
    GridExportGridImport(GridImport *gridImport);
    ~GridExportGridImport();

    GridImport *gridImport;
};

}  // namespace NvFlow

#endif /* NVFLOW_GRIDEXPORTGRID_H */
