#ifndef NVFLOW_GRID_H
#define NVFLOW_GRID_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"
#include "Advect.h"
#include "VorticityConfinement.h"
#include "Emitter.h"
#include "Pressure.h"
#include "GridExport.h"

struct NvFlowGrid : NvFlowObject {
    virtual void reset(const NvFlowGridResetDesc *desc) = 0;

    virtual NvFlowResult querySupport(NvFlowContext *context, NvFlowSupport *support) = 0;

    virtual NvFlowResult queryTime(NvFlowQueryTime *gpuTime, NvFlowQueryTime *cpuTime) = 0;

    virtual void GPUMemUsage(uint64_t *result) = 0;

    virtual void update(NvFlowContext *context, float dt) = 0;

    virtual void setTargetLocation(NvFlowFloat3 newLoc) = 0;

    virtual void setParams(const NvFlowGridParams *params) = 0;

    virtual NvFlowGridMaterialHandle getDefaultMaterial() = 0;

    virtual NvFlowGridMaterialHandle createMaterial(
        const NvFlowGridMaterialParams *materialParams) = 0;

    virtual void releaseMaterial(NvFlowGridMaterialHandle handle) = 0;

    virtual void setMaterialParams(NvFlowGridMaterialHandle handle,
                                   const NvFlowGridMaterialParams *materialParams) = 0;

    virtual void emit(const NvFlowShapeDesc *shapes, uint32_t numShapes,
                      const NvFlowGridEmitParams *params, uint32_t numParams) = 0;

    virtual void updateEmitMaterials(NvFlowGridMaterialHandle *materials,
                                     uint32_t numMaterials) = 0;

    virtual void updateEmitSDFs(NvFlowShapeSDF *const *sdfs, uint32_t numSdfs) = 0;

    virtual void emitCustomRegisterAllocFunc(NvFlowGridEmitCustomAllocFunc func,
                                             void *userdata) = 0;

    virtual void emitCustomRegisterEmitFunc(NvFlowGridTextureChannel channel,
                                            NvFlowGridEmitCustomEmitFunc func,
                                            void *userdata) = 0;

    virtual void emitCustomGetLayerParams(
        const NvFlowGridEmitCustomEmitParams *emitParams, uint32_t layerIdx,
        NvFlowGridEmitCustomEmitLayerParams *emitLayerParams) = 0;

    virtual NvFlowGridExport *getGridExport(NvFlowContext *context) = 0;
};

namespace NvFlow {

struct GridViewExportDebugVis {
    VectorCached<NvFlowFloat4x4, 16> m_bounds;
    VectorCached<NvFlowGridExportSimpleShape, 16> m_spheres;
    VectorCached<NvFlowGridExportSimpleShape, 16> m_capsules;
    VectorCached<NvFlowGridExportSimpleShape, 16> m_boxes;
};

struct BlockManager;
struct Advect;
struct VorticityConfinement;
struct Pressure;
struct Emitter;

struct Grid : Object, NvFlowGrid {
    uint32_t addRef() override;

    uint32_t release() override;

    uint64_t getGPUBytesUsed() override;

    void reset(const NvFlowGridResetDesc *desc) override;

    NvFlowResult querySupport(NvFlowContext *context, NvFlowSupport *support) override;

    NvFlowResult queryTime(NvFlowQueryTime *gpuTime, NvFlowQueryTime *cpuTime) override;

    void GPUMemUsage(uint64_t *numBytes) override;

    void update(NvFlowContext *context, float dt) override;

    void setTargetLocation(NvFlowFloat3 newLoc) override;

    void setParams(const NvFlowGridParams *params) override;

    NvFlowGridMaterialHandle getDefaultMaterial() override;

    NvFlowGridMaterialHandle createMaterial(
        const NvFlowGridMaterialParams *materialParams) override;

    void releaseMaterial(NvFlowGridMaterialHandle handle) override;

    void setMaterialParams(NvFlowGridMaterialHandle handle,
                           const NvFlowGridMaterialParams *materialParams) override;

    void emit(const NvFlowShapeDesc *shapes, uint32_t numShapes,
              const NvFlowGridEmitParams *params, uint32_t numParams) override;

    void updateEmitMaterials(NvFlowGridMaterialHandle *materials,
                             uint32_t numMaterials) override;

    void updateEmitSDFs(NvFlowShapeSDF *const *sdfs, uint32_t numSdfs) override;

    void emitCustomRegisterAllocFunc(NvFlowGridEmitCustomAllocFunc func,
                                     void *userdata) override;

    void emitCustomRegisterEmitFunc(NvFlowGridTextureChannel channel,
                                    NvFlowGridEmitCustomEmitFunc func,
                                    void *userdata) override;

    void emitCustomGetLayerParams(
        const NvFlowGridEmitCustomEmitParams *emitParams, uint32_t layerIdx,
        NvFlowGridEmitCustomEmitLayerParams *emitLayerParams) override;

    NvFlowGridExport *getGridExport(NvFlowContext *context) override;

    // Details
    struct PerMaterial {
        PerMaterial() : valid{}, emitterAllocRefCount{}, materialParams{} {}
        bool valid;
        unsigned int emitterAllocRefCount;
        NvFlowGridMaterialParams materialParams;
    };

    struct PerLayer {
        PerLayer() {
            materialValid = 0;
            materialIdx = 0;
            velocityNumBlocks = 0;
            velocityNumBlocksOld = 0;
            densityNumBlocks = 0;
            densityNumBlocksOld = 0;
        }
        bool materialValid;
        unsigned int materialIdx;
        unsigned int velocityNumBlocks;
        unsigned int velocityNumBlocksOld;
        unsigned int densityNumBlocks;
        unsigned int densityNumBlocksOld;
    };

    struct EmitCustomAllocCallback {
        void (*func)(void *, const NvFlowGridEmitCustomAllocParams *);
        void *userdata;
    };

    struct EmitCustomEmitCallback {
        void (*func)(void *, unsigned int *, const NvFlowGridEmitCustomEmitParams *);
        void *userdata;
    };

    Grid(NvFlowContext *context, const NvFlowGridDesc *desc);
    ~Grid();

    void advectCombustGetPerLayerImpl(AdvectPerLayerParams *params, uint32_t layerIdx);

    static void advectDensityGetPerLayer(AdvectPerLayerParams *params, Grid *userdata,
                                         uint32_t layerIdx);

    void advectDensityGetPerLayerImpl(AdvectPerLayerParams *params, uint32_t layerIdx);

    static void advectVelocityGetPerLayer(AdvectPerLayerParams *params, Grid *userdata,
                                          uint32_t layerIdx);

    void advectVelocityPerLayerImpl(AdvectPerLayerParams *params, uint32_t layerIdx);

    static void blockManagerPerLayer(BlockManagerPerLayerParams *params, Grid *userdata,
                                     uint32_t layerIdx);

    void blockManagerPerLayerImpl(BlockManagerPerLayerParams *params, uint32_t layerIdx);

    void doUpdate(NvFlowContext *context, float dt);

    void emitCustomAlloc(NvFlowContext *context);

    void emitCustomEmit(NvFlowContext *context);

    void emitDebugVis();

    void emitDebugVisSimpleShapes(VectorCached<NvFlowGridExportSimpleShape, 16> &arr,
                                  NvFlowShapeType shapeType);

    NvFlowGridMaterialHandle emitMaterialIndexToMaterial(uint32_t emitMaterialIndex);

    static void emitterPerLayer(EmitterPerLayerParams *params, Grid *userdata,
                                uint32_t layerIdx);

    void emitPerLayerImpl(EmitterPerLayerParams *params, uint32_t layerIdx);

    NvFlowGridMaterialHandle getMaterialHandleFromLayerIdx(uint32_t layerIdx);

    uint32_t getMaterialIdxFromLayerIdx(uint32_t layerIdx);

    PerMaterial *getPerMaterialFromLayerIdx(uint32_t layerIdx);

    PerMaterial *handleToPerMaterial(const NvFlowGridMaterialHandle &handle);

    static void reportDensityLayerNumBlocks(Grid *userdata, uint32_t numBlocks,
                                            uint32_t layerIdx);

    void reportDensityLayerNumBlocksImpl(uint32_t numBlocks, uint32_t layerIdx);

    static void reportSummaryUpdate(Grid *userdata);

    void reportSummaryUpdateImpl();

    static void reportVelocityLayerNumBlocks(Grid *userdata, uint32_t numBlocks,
                                             uint32_t layerIdx);

    void reportVelocityLayerNumBlocksImpl(uint32_t numBlocks, uint32_t layerIdx);

    void updateLayers();

    void updateModelMatrix();

    static void vorticityConfinementPerLayer(VorticityConfinementPerLayerParams *params,
                                             Grid *userdata, uint32_t layerIdx);

    void vorticityConfinementPerLayerImpl(VorticityConfinementPerLayerParams *params,
                                          uint32_t layerIdx);
    NvFlowGridDesc m_desc;
    NvFlowGridParams m_params;
    bool m_is_VTR_supported;
    bool m_resetRequested;
    bool m_resetMode;
    NvFlowGridResetDesc m_resetDesc;
    NvFlowFloat3 m_currentLocation;
    NvFlowFloat3 m_currentHalfSize;
    NvFlowFloat4x4 m_currentModelMatrix;
    NvFlowFloat3 m_oldLocation;
    NvFlowFloat3 m_targetLocation;
    BlockManager *m_blockManager;
    unsigned int m_summaryCount;
    SparseTextureFront m_velocity;
    SparseTextureFront m_densityCoarse;
    SparseTexturePool *m_velocityPool;
    SparseTextureMemoryPool *m_velocityMemoryPool;
    SparseTextureFront m_pressure;
    SparseTexturePool *m_pressurePool;
    SparseTextureMemoryPool *m_pressureMemoryPool;
    SparseTextureFront m_density;
    SparseTexturePool *m_densityPool;
    SparseTextureMemoryPool *m_densityMemoryPool;
    Advect *m_advect;
    VorticityConfinement *m_vorticityConfinement;
    Emitter *m_emitter;
    Pressure *m_pressureOp;
    GridExportGridImpl m_gridExport;
    GridViewExportDebugVis m_debugVis;
    VectorCached<NvFlowGridEmitParams, 16> m_emitters;
    VectorCached<NvFlowShapeDesc, 16> m_shapes;
    unsigned int m_shapeRefs;
    VectorCached<NvFlowGridMaterialHandle, 16> m_emitMaterials;
    VectorCached<NvFlowShapeSDF *, 16> m_shapeSDFs;
    VectorCached<Grid::PerMaterial, 16> m_materials;
    VectorCached<Grid::PerLayer, 16> m_layers;
    Grid::EmitCustomAllocCallback m_emitCustomAlloc;
    Grid::EmitCustomEmitCallback m_emitCustomEmit[3];
    NvFlowContextTimer *m_timer;
    float m_simTimeGPU;
    float m_simTimeCPU;
};


Grid *FlowCreateGrid(NvFlowContext *context, const NvFlowGridDesc *desc);

}  // namespace NvFlow

#endif /* NVFLOW_GRID_H */
