#include "GridSummary.h"
#include "Object.h"
#include "ClientHelper.h"
#include "NvFlowContextImpl.h"
#include "GridExport.h"

namespace NvFlow {

struct GridSummary : Object, NvFlowGridSummary {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    NvFlowGridSummaryStateCPU *createStateCPU() override;
    void update(NvFlowContext *context,
                const NvFlowGridSummaryUpdateParams *params) override;
    void debugRender(NvFlowContext *context,
                     const NvFlowGridSummaryDebugRenderParams *debugParams) override;

    // Details
    struct PerLayer {
        NvFlowGridMaterialHandle gridMaterial;
        uint32_t layerStart;
        uint32_t resultCount;
    };

    enum CaptureState {
        eCaptureStateIdle = 0x0,
        eCaptureStateDownloading = 0x1,
        eCaptureStateRead = 0x2,
    };

    struct Capture {
        CaptureState m_captureState;
        uint64_t m_captureVersion;
        NvFlowBuffer *m_downloadBuffer;
        VectorCached<PerLayer, 16> m_perLayers;
    };

    struct DebugVisResources {
        bool m_allocated;
        NvFlowBuffer *m_uploadBuffer;
        NvFlowVertexBuffer *m_vertexBuffer;
        NvFlowIndexBuffer *m_indexBuffer;
        NvFlowConstantBuffer *m_renderConstantBuffer;
        NvFlowGraphicsShader *m_gridSummaryDebugVis;
    };

    void allocateDebugVisResources(NvFlowContext *context,
                                   const NvFlowGridSummaryDebugRenderParams *params);

    GridSummary(NvFlowContext *context, const NvFlowGridSummaryDesc *desc);
    ~GridSummary();

    uint32_t m_maxBufferDim;
    NvFlowUint4 m_subBlockDimBits;
    NvFlowFloat4 m_gridWorldHalfSize;
    NvFlowFloat4 m_gridWorldLocation;
    uint64_t m_captureVersion;
    VectorCached<Capture, 4> m_captures;
    uint32_t m_frontCapture;
    NvFlowComputeShader *m_gridSummaryCS;
    NvFlowConstantBuffer *m_constantBuffer;
    DebugVisResources m_debugVisResources;
};

struct GridSummaryDebugVisShaderParams {
    NvFlowFloat4x4 modelViewProj;
};

struct GridSummaryShaderParams {
    NvFlowShaderLinearParams velocityParams;
    NvFlowShaderLinearParams densityParams;
    NvFlowUint4 subBlockDimBits;
    NvFlowFloat4 gridWorldHalfSize;
    NvFlowFloat4 gridWorldLocation;
    NvFlowUint4 blockIdxOffset;
    NvFlowFloat4 velocityScale;
    NvFlowFloat4 densityScale;
};

struct GridSummaryStateCPU : Object, NvFlowGridSummaryStateCPU {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    uint32_t getNumLayers() override;
    NvFlowGridMaterialHandle getLayerMaterial(uint32_t layerIdx) override;
    void getSummaries(const NvFlowGridSummaryResult **results, uint32_t *numResults,
                      uint32_t layerIdx) override;

    // Details

    struct PerLayer {
        NvFlowGridMaterialHandle gridMaterial;
        uint32_t layerStart;
        uint32_t resultCount;
    };

    GridSummaryStateCPU(GridSummary *gridSummary);

    VectorCached<NvFlowGridSummaryResult, 1> m_results;
    VectorCached<PerLayer, 16> m_perLayers;
};

#include "gridSummaryCS.hlsl.h"
#include "gridSummaryDebugVisVS.hlsl.h"
#include "gridSummaryDebugVisPS.hlsl.h"

uint64_t GridSummary::getGPUBytesUsed() {
    return 0;
}

NvFlowGridSummaryStateCPU *GridSummary::createStateCPU() {
    return new GridSummaryStateCPU(this);
}

void GridSummary::update(NvFlowContext *context,
                         const NvFlowGridSummaryUpdateParams *params) {
    for (uint32_t captureIdx = 0; captureIdx < m_captures.size(); ++captureIdx) {
        auto &capture = m_captures[captureIdx];
        if (capture.m_captureState == eCaptureStateDownloading) {
            if (NvFlowBufferMapDownload(context, capture.m_downloadBuffer)) {
                capture.m_captureState = eCaptureStateRead;
                NvFlowBufferUnmapDownload(context, capture.m_downloadBuffer);
            }
        }
    }

    uint64_t newestCaptureVersion = 0;
    uint64_t newestCaptureIdx = 0;
    for (uint32_t idx = 0; idx < m_captures.size(); ++idx) {
        auto &capture = m_captures[idx];
        if (capture.m_captureState == eCaptureStateRead) {
            if (capture.m_captureVersion > newestCaptureVersion) {
                newestCaptureVersion = capture.m_captureVersion;
                newestCaptureIdx = idx;
            }
        }
    }

    for (uint32_t j = 0; j < m_captures.size(); ++j) {
        auto &capture = m_captures[j];
        if (capture.m_captureState == eCaptureStateRead) {
            if (capture.m_captureVersion != newestCaptureVersion)
                capture.m_captureState = eCaptureStateIdle;
        }
    }

    if (params->stateCPU) {
        auto stateCPU = implCast<GridSummaryStateCPU>(params->stateCPU);
        if (newestCaptureVersion) {
            auto &capture = m_captures[newestCaptureIdx];
            auto results = (NvFlowGridSummaryResult *)NvFlowBufferMapDownload(
                context, capture.m_downloadBuffer);
            if (results) {
                uint32_t totalResults = 0;
                stateCPU->m_perLayers.clear();
                stateCPU->m_results.clear();
                for (uint32_t layerIdx = 0; layerIdx < capture.m_perLayers.size();
                     ++layerIdx) {
                    auto &perLayer = capture.m_perLayers[layerIdx];

                    uint32_t allocIdx = stateCPU->m_perLayers.allocateBack();
                    auto &layer = stateCPU->m_perLayers[allocIdx];

                    layer.gridMaterial = perLayer.gridMaterial;
                    layer.layerStart = perLayer.layerStart;
                    layer.resultCount = perLayer.resultCount;

                    totalResults += perLayer.resultCount;
                }

                for (uint32_t resultIdx = 0; resultIdx < totalResults; ++resultIdx) {
                    uint32_t Back = stateCPU->m_results.allocateBack();
                    auto &dstResult = stateCPU->m_results[Back];
                    dstResult = results[resultIdx];
                }

                NvFlowBufferUnmapDownload(context, capture.m_downloadBuffer);
            }
        } else {
            stateCPU->m_perLayers.clear();
            stateCPU->m_results.clear();
        }
    }

    if (params->gridExport) {
        bool captureValid = 0;
        uint32_t k;
        for (k = 0; k < m_captures.size(); ++k) {
            if (m_captures[k].m_captureState == eCaptureStateIdle) {
                captureValid = 1;
                break;
            }
        }

        if (!captureValid && m_captures.size() < 4) {
            k = m_captures.allocateBack();
            captureValid = 1;
        }

        if (captureValid) {
            auto &capture = m_captures[k];
            ++m_captureVersion;
            capture.m_captureState = eCaptureStateDownloading;
            capture.m_captureVersion = m_captureVersion;

            auto velocityHandle =
                params->gridExport->getHandle(context, eNvFlowGridTextureChannelVelocity);
            auto densityHandle = params->gridExport->getHandle(
                context, eNvFlowGridTextureChannelDensityCoarse);
            NvFlowGridExportLayeredView velocityLayeredView = {};
            NvFlowGridExportLayeredView densityLayeredView = {};
            velocityHandle.gridExport->getLayeredView(velocityHandle, &velocityLayeredView);
            densityHandle.gridExport->getLayeredView(densityHandle, &densityLayeredView);
            auto mat = densityLayeredView.mapping.modelMatrix;
            NvFlowFloat4 gridLocation = densityLayeredView.mapping.modelMatrix.w;
            NvFlowFloat4 gridHalfSize = transform4(make_float4(1.f, 1.f, 1.f, 0.f), mat);
            gridHalfSize.w = 0.f;
            NvFlowUint4 subBlockDimBits;
            (NvFlowUint3 &)subBlockDimBits =
                (const NvFlowUint3 &)densityLayeredView.mapping.shaderParams.blockDimBits -
                3;
            subBlockDimBits.w = densityLayeredView.mapping.shaderParams.blockDimBits.w - 9;

            uint32_t resultsPerBlock =
                1 << (uint8_t(densityLayeredView.mapping.shaderParams.blockDimBits.w) - 9);
            if (!capture.m_downloadBuffer) {
                NvFlowBufferDesc bufDesc = {};
                bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
                bufDesc.dim = resultsPerBlock * 4 * densityLayeredView.mapping.maxBlocks;
                bufDesc.uploadAccess = 0;
                bufDesc.downloadAccess = 1;
                m_maxBufferDim = bufDesc.dim;
                capture.m_downloadBuffer = NvFlowCreateBuffer(context, &bufDesc);
            }

            NvFlowUint4 blockIdxOffset = make_uint4(0);
            capture.m_perLayers.clear();

            const float velScale = 0.001953125f;
            const NvFlowFloat4 velocityScale = make_float4(velScale);
            const NvFlowFloat4 densityScale = make_float4(velScale);

            for (uint32_t m = 0; m < densityHandle.numLayerViews; ++m) {
                NvFlowGridExportLayerView velocityLayerView = {}, densityLayerView = {};
                densityHandle.gridExport->getLayerView(densityHandle, m, &densityLayerView);
                velocityHandle.gridExport->getLayerView(velocityHandle, m,
                                                        &velocityLayerView);

                auto mapped = (GridSummaryShaderParams *)NvFlowConstantBufferMap(
                    context, m_constantBuffer);
                if (mapped) {
                    mapped->velocityParams = velocityLayeredView.mapping.shaderParams;
                    mapped->densityParams = densityLayeredView.mapping.shaderParams;
                    mapped->subBlockDimBits = subBlockDimBits;
                    mapped->gridWorldHalfSize = gridHalfSize;
                    mapped->gridWorldLocation = gridLocation;
                    mapped->blockIdxOffset = blockIdxOffset;
                    mapped->velocityScale = velocityScale;
                    mapped->densityScale = densityScale;
                    NvFlowConstantBufferUnmap(context, m_constantBuffer);
                }

                NvFlowDim gridDim;
                gridDim.x = (densityLayeredView.mapping.shaderParams.blockDim.x *
                                 densityLayerView.mapping.numBlocks +
                             7) /
                            8;
                gridDim.y = (densityLayeredView.mapping.shaderParams.blockDim.y + 7) / 8;
                gridDim.z = (densityLayeredView.mapping.shaderParams.blockDim.z + 7) / 8;
                NvFlowDispatchParams dparams = {};
                dparams.shader = m_gridSummaryCS;
                dparams.gridDim = gridDim;
                dparams.rootConstantBuffer = m_constantBuffer;
                dparams.readOnly[0] = densityLayerView.mapping.blockList;
                dparams.readOnly[1] = velocityLayerView.data;
                dparams.readOnly[2] = velocityLayerView.mapping.blockTable;
                dparams.readOnly[3] = densityLayerView.data;
                dparams.readOnly[4] = densityLayerView.mapping.blockTable;
                dparams.readWrite[0] = NvFlowBufferGetResourceRW(capture.m_downloadBuffer);
                NvFlowContextDispatch(context, &dparams);

                uint32_t perLayerIdx = capture.m_perLayers.allocateBack();
                auto &perLayer = capture.m_perLayers[perLayerIdx];
                perLayer.gridMaterial = densityLayerView.mapping.material;
                perLayer.layerStart = resultsPerBlock * blockIdxOffset.x;
                perLayer.resultCount = resultsPerBlock * densityLayerView.mapping.numBlocks;
                blockIdxOffset.x += densityLayerView.mapping.numBlocks;
            }
            NvFlowBufferDownload(context, capture.m_downloadBuffer);
        }
    }
}

void GridSummary::debugRender(NvFlowContext *context,
                              const NvFlowGridSummaryDebugRenderParams *params) {
    auto rtv = params->renderTargetView;
    if (params->stateCPU) {
        auto stateCPU = implCast<GridSummaryStateCPU>(params->stateCPU);
        if (stateCPU->m_results.size()) {
            if (m_maxBufferDim) {
                if (!m_debugVisResources.m_allocated) {
                    m_debugVisResources.m_allocated = 1;
                    allocateDebugVisResources(context, params);
                }
                uint32_t numResults = stateCPU->m_results.size();
                auto modelViewProj = params->viewMatrix * params->projectionMatrix;
                auto modelViewProjT = transpose(modelViewProj);
                auto mapped = (GridSummaryDebugVisShaderParams *)NvFlowConstantBufferMap(
                    context, m_debugVisResources.m_renderConstantBuffer);
                if (mapped) {
                    mapped->modelViewProj = modelViewProjT;
                    NvFlowConstantBufferUnmap(context,
                                              m_debugVisResources.m_renderConstantBuffer);
                }

                auto results = (NvFlowGridSummaryResult *)NvFlowBufferMap(
                    context, m_debugVisResources.m_uploadBuffer);
                if (results) {
                    auto src = stateCPU->m_results.data();
                    for (uint32_t i = 0; i < numResults; ++i)
                        results[i] = src[i];
                    NvFlowBufferUnmap(context, m_debugVisResources.m_uploadBuffer);
                }

                auto RenderTarget = NvFlowRenderTargetViewGetRenderTarget(rtv);
                NvFlowRenderTargetDesc rtv_desc;
                NvFlowRenderTargetGetDesc(RenderTarget, &rtv_desc);
                NvFlowGraphicsShaderSetFormats(context,
                                               m_debugVisResources.m_gridSummaryDebugVis,
                                               rtv_desc.rt_format, eNvFlowFormat_d32_float);

                NvFlowDrawParams drawParams = {};
                drawParams.shader = m_debugVisResources.m_gridSummaryDebugVis;
                drawParams.rootConstantBuffer = m_debugVisResources.m_renderConstantBuffer;
                drawParams.vs_readOnly[0] =
                    NvFlowBufferGetResource(m_debugVisResources.m_uploadBuffer);
                NvFlowContextSetVertexBuffer(context, m_debugVisResources.m_vertexBuffer,
                                             sizeof(NvFlowFloat4), 0);
                NvFlowContextSetIndexBuffer(context, m_debugVisResources.m_indexBuffer, 0);
                NvFlowContextDrawIndexedInstanced(context, 2, numResults, &drawParams);
            }
        }
    }
}

void GridSummary::allocateDebugVisResources(
    NvFlowContext *context, const NvFlowGridSummaryDebugRenderParams *params) {
    NvFlowBufferDesc bufDesc = {};
    bufDesc.format = eNvFlowFormat_r32g32b32a32_float;
    bufDesc.dim = m_maxBufferDim;
    bufDesc.uploadAccess = 1;
    bufDesc.downloadAccess = 0;
    m_debugVisResources.m_uploadBuffer = NvFlowCreateBuffer(context, &bufDesc);

    NvFlowFloat4 pts[8] = {{-1.f, -1.f, -1.f, 1.f}, {1.f, -1.f, -1.f, 1.f},
                           {1.f, 1.f, -1.f, 1.f},   {-1.f, 1.f, -1.f, 1.f},
                           {-1.f, 1.f, 1.f, 1.f},   {1.f, 1.f, 1.f, 1.f},
                           {1.f, -1.f, 1.f, 1.f},   {-1.f, -1.f, 1.f, 1.f}};

    NvFlowUint indices[66] = {0, 1, 3, 3, 1, 2, 5, 6, 4, 4, 6, 7, 1, 0, 6, 6, 0,
                              7, 2, 5, 3, 3, 5, 4, 3, 4, 0, 0, 4, 7, 1, 6, 2, 2,
                              6, 5, 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7,
                              4, 0, 7, 1, 6, 2, 5, 3, 4, 0, 3, 1, 1, 3, 2};

    NvFlowVertexBufferDesc vbufDesc = {};
    vbufDesc.data = pts;
    vbufDesc.sizeInBytes = sizeof(pts);
    m_debugVisResources.m_vertexBuffer = NvFlowCreateVertexBuffer(context, &vbufDesc);

    NvFlowIndexBufferDesc ibufDesc = {};
    ibufDesc.data = indices;
    ibufDesc.format = eNvFlowFormat_r32_uint;
    ibufDesc.sizeInBytes = sizeof(indices);
    m_debugVisResources.m_indexBuffer = NvFlowCreateIndexBuffer(context, &ibufDesc);

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(GridSummaryDebugVisShaderParams);
    cbDesc.uploadAccess = 1;
    m_debugVisResources.m_renderConstantBuffer =
        NvFlowCreateConstantBuffer(context, &cbDesc);

    NvFlowInputElementDesc elementDescs[1];
    elementDescs[0].format = eNvFlowFormat_r32g32b32a32_float;
    elementDescs[0].semanticName = "POSITION";

    NvFlowGraphicsShaderDesc shaderDesc = {};
    shaderDesc.numInputElements = countof(elementDescs);
    shaderDesc.inputElementDescs = elementDescs;
    shaderDesc.blendState.enable = 0;
    shaderDesc.depthState.depthEnable = 0;
    shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
    shaderDesc.depthState.depthFunc = eNvFlowComparison_LessEqual;
    shaderDesc.uavTarget = 0;
    shaderDesc.depthClipEnable = 1;
    shaderDesc.numRenderTargets = 1;
    shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r8g8b8a8_unorm;
    shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
    shaderDesc.vs = g_gridSummaryDebugVisVS;
    shaderDesc.vs_length = sizeof(g_gridSummaryDebugVisVS);
    shaderDesc.ps = g_gridSummaryDebugVisPS;
    shaderDesc.ps_length = sizeof(g_gridSummaryDebugVisPS);
    shaderDesc.label = L"gridSummaryDebugVis";
    shaderDesc.lineList = 1;
    m_debugVisResources.m_gridSummaryDebugVis =
        NvFlowCreateGraphicsShader(context, &shaderDesc);
}

GridSummary::GridSummary(NvFlowContext *context, const NvFlowGridSummaryDesc *desc)
    : m_maxBufferDim(0),
      m_subBlockDimBits{0, 0, 0, 0},
      m_gridWorldHalfSize{0.f, 0.f, 0.f, 0.f},
      m_gridWorldLocation{0.f, 0.f, 0.f, 0.f},
      m_captureVersion{0},
      m_frontCapture{0},
      m_gridSummaryCS{0},
      m_constantBuffer{0},
      m_debugVisResources{} {
    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_gridSummaryCS = createShader(NVFLOW_CREATE_SHADER_ARGS(gridSummaryCS));

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(GridSummaryShaderParams);
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);
}

GridSummary::~GridSummary() {
    for (auto &capture : m_captures)
        SafeRelease(capture.m_downloadBuffer);

    SafeRelease(m_gridSummaryCS);
    SafeRelease(m_constantBuffer);

    SafeRelease(m_debugVisResources.m_uploadBuffer);
    SafeRelease(m_debugVisResources.m_vertexBuffer);
    SafeRelease(m_debugVisResources.m_indexBuffer);
    SafeRelease(m_debugVisResources.m_renderConstantBuffer);
    SafeRelease(m_debugVisResources.m_gridSummaryDebugVis);
}

uint64_t GridSummaryStateCPU::getGPUBytesUsed() {
    return 0;
}

uint32_t GridSummaryStateCPU::getNumLayers() {
    return m_perLayers.size();
}

NvFlowGridMaterialHandle GridSummaryStateCPU::getLayerMaterial(uint32_t layerIdx) {
    if (layerIdx < m_perLayers.size()) {
        return m_perLayers[layerIdx].gridMaterial;
    } else {
        NvFlowGridMaterialHandle result;
        ZeroMemory(&result, sizeof(result));
        return result;
    }
}

void GridSummaryStateCPU::getSummaries(const NvFlowGridSummaryResult **results,
                                       uint32_t *numResults, uint32_t layerIdx) {
    if (layerIdx >= m_perLayers.size()) {
        if (results) *results = 0;
        if (numResults) *numResults = 0;
    } else {
        if (results) {
            auto &perLayer = m_perLayers[layerIdx];
            *results = &m_results[perLayer.layerStart];
        }
        if (numResults) {
            *numResults = m_perLayers[layerIdx].resultCount;
        }
    }
}

GridSummaryStateCPU::GridSummaryStateCPU(GridSummary *gridSummary) {}

NvFlowGridSummary *FlowCreateGridSummary(NvFlowContext *context,
                                         const NvFlowGridSummaryDesc *desc) {
    return new GridSummary(context, desc);
}

}  // namespace NvFlow