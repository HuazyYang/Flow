#ifndef NVFLOW_VOLUMERENDER_H
#define NVFLOW_VOLUMERENDER_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"

struct NvFlowVolumeRender : NvFlowObject {
    virtual NvFlowGridExport *lightGridExport(NvFlowContext *context,
                                              NvFlowGridExport *gridExport,
                                              const NvFlowVolumeLightingParams *params) = 0;

    virtual void renderGridExport(NvFlowContext *context, NvFlowGridExport *gridExport,
                                  const NvFlowVolumeRenderParams *params) = 0;

    virtual void renderTexture3D(NvFlowContext *context, NvFlowTexture3D *density,
                                 const NvFlowVolumeRenderParams *params) = 0;
};

namespace NvFlow {

NvFlowVolumeRender *FlowCreateVolumeRender(NvFlowContext *context,
                                           const NvFlowVolumeRenderDesc *desc);
}

#include "Object.h"
#include "GridExport.h"
#include "ClientHelper.h"
#include "NvFlowContextImpl.h"
#include "RadixSort.h"
#include "GridImport.h"

namespace NvFlow {

struct VolumeRender : Object, NvFlowVolumeRender {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    NvFlowGridExport *lightGridExport(NvFlowContext *context, NvFlowGridExport *gridExport,
                                      const NvFlowVolumeLightingParams *params) override;

    void renderGridExport(NvFlowContext *context, NvFlowGridExport *gridExport,
                          const NvFlowVolumeRenderParams *params) override;

    void renderTexture3D(NvFlowContext *context, NvFlowTexture3D *density,
                         const NvFlowVolumeRenderParams *params) override;

    // Details

    enum VolumeRenderDepthDir {
        eDepthDirForwardZ = 0x0,
        eDepthDirReverseZ = 0x1,
        eDepthDirMax = 0x2,
    };

    enum DepthDownLMS {
        eLMSDisabled = 0x0,
        eLMSEnabled = 0x1,
        eLMSMax = 0x2,
    };

    enum DepthDownsampleDepthMask {
        eDepthMaskDisabled = 0x0,
        eDepthMaskEnabled = 0x1,
        eDepthMaskMax = 0x2,
    };

    enum VolumeRenderNumLayers {
        eVolumeRenderNumLayers1 = 0x0,
        eVolumeRenderNumLayers2 = 0x1,
        eVolumeRenderNumLayers3 = 0x2,
        eVolumeRenderNumLayers4 = 0x3,
        eVolumeRenderNumLayersCount = 0x4,
    };

    enum VolumeRenderShader {
        eVolumeRender = 0x0,
        eVolumeRender_colormap = 0x1,
        eVolumeRender_debug = 0x2,
        eVolumeRender_raw = 0x3,
        eVolumeRenderMax = 0x4,
    };

    enum DepthTestShader {
        eDepthTestDisabled = 0x0,
        eDepthTestEnabled = 0x1,
        eDepthTestMax = 0x2,
    };

    struct BlockListRange {
        unsigned int layerListStart;
        unsigned int layerListCount;
        unsigned int blockListStart;
        unsigned int blockListCount;
        NvFlowResource *blockList;
    };

    struct BlockListRangeList {
        unsigned int *layerLists;
        unsigned int numLayerLists;
        BlockListRange *ranges;
        unsigned int numRanges;
        NvFlowResource *allBlockList;
        unsigned int allNumBlocks;
    };

    struct DebugUploadBuffer {
        NvFlowBuffer *m_buffer;
        unsigned int m_bufferSize;

        void reserve(NvFlowContext *context, NvFlowFormat format, uint32_t numElements);

        void release() { SafeRelease(m_buffer); }
    };

    struct DebugSimpleShapeMeshes {
        NvFlowVertexBuffer *m_vertexBuffer;
        NvFlowIndexBuffer *m_indexBuffer;

        static const float pts[944];

        static uint32_t indices[480];

        void release() {
            SafeRelease(m_vertexBuffer);
            SafeRelease(m_indexBuffer);
        }

        void drawMesh(NvFlowContext *context, NvFlowShapeType shapeType,
                      uint32_t numInstances, const NvFlowDrawParams *drawParams);
    };

    struct OffscreenBuffer {
        NvFlowColorBuffer *m_depthMaxBuffer;
        NvFlowColorBuffer *m_depthMinBuffer;
        NvFlowColorBuffer *m_colorBuffer;
        NvFlowDepthBuffer *m_rayMarchMask;
        unsigned int m_width;
        unsigned int m_height;
        NvFlowViewport m_viewport;
        float m_tmax;
        float screenPercentX;
        float screenPercentY;

        void release() {
            SafeRelease(m_depthMaxBuffer);
            SafeRelease(m_depthMinBuffer);
            SafeRelease(m_colorBuffer);
            SafeRelease(m_rayMarchMask);
        }
    };

    void composite(NvFlowContext *context, NvFlowRenderTarget *rt, NvFlowDepthStencil *ds,
                   NvFlowResource *dsvResource, const NvFlowFloat4 &depthInvTransform,
                   float tlimitMin, float tlimitMax,
                   const NvFlowVolumeRenderParams *params);

    void compositeDepthDebug(NvFlowContext *context, NvFlowRenderTarget *rt,
                             NvFlowDepthStencil *ds, NvFlowResource *dsvResource,
                             const NvFlowFloat4 &depthInvTransform, float tlimitMin,
                             float tlimitMax, const NvFlowVolumeRenderParams *params);

    void compositeDepthEstimate(NvFlowContext *context, NvFlowDepthStencil *ds,
                                NvFlowDepthStencilView *dsv,
                                const NvFlowFloat4 &depthInvTransform, float tlimitMin,
                                float tlimitMax, const NvFlowVolumeRenderParams *params);

    void debugRender(NvFlowContext *context, NvFlowRenderTarget *rt, NvFlowDepthStencil *ds,
                     const BlockListRangeList *blockListRangeList,
                     NvFlowGridExport *gridExport, const NvFlowFloat4x4 &viewProj,
                     const NvFlowFloat4x4 &modelViewProj, const NvFlowFloat4 &vGridDimInv,
                     const NvFlowVolumeRenderParams *params);

    void downsampleDepth(NvFlowContext *context, NvFlowResource *dsvResource,
                         NvFlowDepthStencil *ds,
                         const BlockListRangeList *blockListRangeList,
                         const NvFlowFloat4x4 &modelViewProjT,
                         const NvFlowFloat4 &depthInvTransform,
                         const NvFlowFloat4 &vGridDimInv,
                         const NvFlowVolumeRenderParams *params);

    void extractRenderTargets(NvFlowRenderTargetView **rtv, NvFlowDepthStencilView **dsv,
                              NvFlowRenderTarget **rt, NvFlowDepthStencil **ds,
                              unsigned int *rtv_width, unsigned int *rtv_height,
                              const NvFlowVolumeRenderParams *params);

    uint32_t genBlockDistKey(uint32_t val, const NvFlowUint4 &blockDim,
                             const NvFlowFloat4 &rayOriginVirtual);

    uint32_t genBlockDistKeyCoarse(uint32_t val, const NvFlowUint4 &blockDim,
                                   const NvFlowFloat4 &rayOriginVirtual);

    uint32_t genBlockIdxKeyCoarse(uint32_t val, const NvFlowUint4 &blockDim);

    BlockListRangeList generateBlockListRangeListLayered(
        NvFlowContext *context, NvFlowUint2 *layeredBlockListCPU, uint32_t layeredNumBlocks,
        uint32_t numLayerViews, const NvFlowUint4 &blockDim,
        const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual);

    BlockListRangeList generateBlockListRangeListSingle(
        NvFlowContext *context, unsigned int layerIdx, unsigned int numBlocks,
        NvFlowResource *blockList, const NvFlowUint4 &blockDim,
        const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual);

    void generateDefaultMesh(NvFlowContext *context);

    void generateMultiResMesh(NvFlowContext *context,
                              const NvFlowVolumeRenderMultiResParams *params);

    void rayMarch(NvFlowContext *context, NvFlowRenderTarget *dstTarget,
                  NvFlowDepthStencil *rayMarchMask, bool clearRenderTarget,
                  NvFlowResource *depthMaxBuffer, NvFlowResource *depthMinBuffer,
                  const NvFlowViewport &offscreenViewport, float screenPercentX,
                  float screenPercentY, const BlockListRangeList *blockListRangeList,
                  const NvFlowGridExportHandle &exportHandle,
                  const NvFlowFloat4x4 &modelViewProjT,
                  const NvFlowFloat4 &depthMaxInvTransform,
                  const NvFlowFloat4 &depthMinInvTransform,
                  const NvFlowFloat4 &rayOriginVirtual,
                  const NvFlowFloat4 &rayForwardDirVirtual,
                  const NvFlowShaderLinearParams *linearParams, const NvFlowDim &gridDim,
                  const bool enableVTR, const NvFlowVolumeRenderParams *params);

    void rayMarchDepthEstimate(
        NvFlowContext *context, NvFlowRenderTarget *dstTarget,
        const NvFlowViewport &offscreenViewport, float screenPercentX, float screenPercentY,
        const NvFlowGridExportHandle &exportHandle, const NvFlowFloat4x4 &modelViewProjT,
        const NvFlowFloat4 &rayOriginVirtual, const NvFlowFloat4 &rayForwardDirVirtual,
        const NvFlowShaderLinearParams *linearParams, const NvFlowDim &gridDim,
        bool enableVTR, const NvFlowVolumeRenderParams *params);

    void rayMarchMultiRes(
        NvFlowContext *context, const BlockListRangeList *blockListRangeList,
        const NvFlowGridExportHandle &exportHandle, const NvFlowFloat4x4 &modelViewProjT,
        const NvFlowFloat4 &depthInvTransform, const NvFlowFloat4 &rayOriginVirtual,
        const NvFlowFloat4 &rayForwardDirVirtual,
        const NvFlowShaderLinearParams *linearParams, const NvFlowDim &gridDim,
        bool enableVTR, const NvFlowFloat4x4 &projectionMatrixInv,
        const NvFlowVolumeRenderParams *params);

    void resizeDepth(NvFlowContext *context, unsigned int rtv_width,
                     unsigned int rtv_height, const NvFlowVolumeRenderParams *params);

    void resizeTargets(NvFlowContext *context, unsigned int rtv_width,
                       unsigned int rtv_height, const NvFlowVolumeRenderParams *params);

    VolumeRender(NvFlowContext *context, const NvFlowVolumeRenderDesc *desc);
    ~VolumeRender();

    NvFlowVolumeRenderDesc m_desc;
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowVertexBuffer *m_vertexBuffer;
    NvFlowIndexBuffer *m_indexBuffer;
    NvFlowVertexBuffer *m_compositeVertexBufferRect;
    NvFlowIndexBuffer *m_compositeIndexBufferRect;
    NvFlowVertexBuffer *m_compositeVertexBufferMultiRes;
    NvFlowIndexBuffer *m_compositeIndexBufferMultiRes;
    DebugUploadBuffer m_debugEmitBoundsBuffer;
    DebugUploadBuffer m_debugSphereBuffer;
    DebugUploadBuffer m_debugCapsuleBuffer;
    DebugUploadBuffer m_debugBoxBuffer;
    DebugSimpleShapeMeshes m_debugSimpleShapeMeshes;
    NvFlowGraphicsShader *m_compositeShader;
    NvFlowGraphicsShader *m_compositeShader_LMS;
    NvFlowGraphicsShader *m_compositeSmoothShader;
    NvFlowGraphicsShader *m_compositeSmoothShader_LMS;
    NvFlowGraphicsShader *m_compositeDepthEstimate[2];
    NvFlowGraphicsShader *m_compositeDepthEstimate_LMS[2];
    NvFlowGraphicsShader *m_compositeDepthDebug;
    NvFlowGraphicsShader *m_compositeDepthDebug_LMS;
    NvFlowGraphicsShader *m_depthDownsampleShader[2][2];
    NvFlowGraphicsShader *m_volumeRender[2][4][4];
    NvFlowGraphicsShader *m_volumeRenderDepthEstimate;
    NvFlowGraphicsShader *m_volumeRenderBox;
    NvFlowGraphicsShader *m_volumeRenderDebug;
    NvFlowGraphicsShader *m_volumeRenderDebugEmitBounds;
    NvFlowGraphicsShader *m_volumeRenderDebugShapesSimple;
    NvFlowComputeShader *m_sortShader;
    NvFlowGraphicsShader *m_multiResColorUpsampleShader;
    NvFlowGraphicsShader *m_multiResDepthDownsampleShader;
    NvFlowGraphicsShader *m_multiResDepthUpsampleShader;
    NvFlowGraphicsShader *m_volumeRenderDepth[2];
    NvFlowGraphicsShader *m_rayMarchMask;
    VectorCached<OffscreenBuffer, 8> m_offscreenBuffers;
    NvFlowDepthBuffer *m_depthMask;
    NvFlowColorBuffer *m_depthEstimate;
    NvFlowViewport m_depthEstimateViewport;
    float m_depthEstimateScreenPercentX;
    float m_depthEstimateScreenPercentY;
    RadixSort *m_sort;
    RadixSortCPU *m_sortCPU;
    NvFlowBuffer *m_blockListUpload;
    VectorCached<unsigned int, 1> m_blockListLayers;
    VectorCached<unsigned int, 1> m_blockListSorted;
    VectorCached<BlockListRange, 1> m_blockListRanges;
    VectorCached<unsigned int, 1> m_layerCounters;
    VectorCached<NvFlowRenderMaterialHandle, 1> m_drawRenderMaterials;
    VectorCached<unsigned int, 1> m_drawLayerIndices;
    VectorCached<NvFlowRenderMaterialHandle, 1> m_depthEstimateRenderMaterials;
    NvFlowGridImport *m_gridImport;
    NvFlowComputeShader *m_lightingShader;
    NvFlowComputeShader *m_lightingShader_SST;
    NvFlowComputeShader *m_lightingShader_VTR;
};

struct VolumeRenderShaderParams {
#include "volumeRenderShaderParams.h"
};

struct VolumeRender2ShaderParams {
    NvFlowFloat4x4 modelViewProj;
    NvFlowFloat4x4 modelViewProjInv;
    NvFlowFloat4 viewportInvScale;
    NvFlowFloat4 viewportInvOffset;
    NvFlowFloat4 dimInv;
    NvFlowFloat4 minCoord;
    NvFlowFloat4 maxCoord;
    NvFlowFloat4 rayOrigin;
    NvFlowFloat4 rayForwardDir;
    NvFlowUint4 renderMode;
    NvFlowFloat4 alphaScale;
};

};  // namespace NvFlow

#endif /* NVFLOW_VOLUMERENDER_H */
