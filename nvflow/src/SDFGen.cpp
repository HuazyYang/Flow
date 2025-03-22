#include "SDFGen.h"
#include "NvFlowContextImpl.h"
#include "VectorCached.h"
#include "Object.h"
#include "ClientHelper.h"

namespace NvFlow {

struct SDFGen : Object, NvFlowSDFGen {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void reset(NvFlowContext *context) override;
    void voxelize(NvFlowContext *context, const NvFlowSDFGenMeshParams *params) override;
    void update(NvFlowContext *context) override;
    NvFlowTexture3D *shape(NvFlowContext *context) override;

    // Details
    SDFGen(NvFlowContext *context, const NvFlowSDFGenDesc *desc);
    ~SDFGen();

    int stateUpdate(NvFlowContext *context);

    struct Level {
        NvFlowTexture3D *m_normalField;
        NvFlowTexture3D *m_signFieldFront;
        NvFlowTexture3D *m_signFieldBack;
        NvFlowDim m_dim;

        void release() {
            SafeRelease(m_normalField);
            SafeRelease(m_signFieldFront);
            SafeRelease(m_signFieldBack);
        }
    };

    NvFlowComputeShader *m_clearTextureCS;
    NvFlowComputeShader *m_downsampleCS;
    NvFlowComputeShader *m_smoothCS;
    NvFlowComputeShader *m_upsampleCS;
    NvFlowComputeShader *m_blockCompactCS;
    NvFlowComputeShader *m_blockUnsignedDistanceCS;
    NvFlowComputeShader *m_compactCS;
    NvFlowComputeShader *m_unsignedDistanceCS;
    NvFlowGraphicsShader *m_meshNormalField;
    NvFlowConstantBuffer *m_constantBuffer;
    VectorCached<NvFlowIndexBuffer *, 8> m_indexBuffers;
    VectorCached<NvFlowVertexBuffer *, 8> m_vertexBuffers;
    NvFlowDim m_rootDim;
    unsigned int m_numLevels;
    Level m_levels[4];
    NvFlowBuffer *m_atomic;
    NvFlowConstantBuffer *m_atomicConst;
    NvFlowBuffer *m_blockList;
    NvFlowTexture3D *m_rangeList;
    NvFlowBuffer *m_cellList;
    NvFlowTexture3D *m_blockDistanceField;
    NvFlowTexture3D *m_distanceField;
    int m_state;
    int m_stateLevel;
    int m_substate;
};

struct SmoothShaderParams {
    NvFlowUint4 signFieldDim;
};

struct UnsignedDistanceShaderParams {
    NvFlowUint4 blockFieldDim;
    NvFlowFloat4 scale;
};

struct MeshNormalFieldShaderParams {
    NvFlowFloat4x4 projection;
    NvFlowFloat4 normalFieldDim;
    NvFlowUint4 swizzleMode;
};

#include "clearTextureCS.hlsl.h"
#include "downsampleCS.hlsl.h"
#include "smoothCS.hlsl.h"
#include "upsampleCS.hlsl.h"
#include "blockCompactCS.hlsl.h"
#include "blockUnsignedDistanceCS.hlsl.h"
#include "compactCS.hlsl.h"
#include "unsignedDistanceCS.hlsl.h"
#include "meshNormalFieldVS.hlsl.h"
#include "meshNormalFieldPS.hlsl.h"

uint64_t SDFGen::getGPUBytesUsed() {
    return 0;
}

void SDFGen::reset(NvFlowContext *context) {
    NvFlowDispatchParams dparams = {};
    dparams.shader = m_clearTextureCS;
    dparams.gridDim = (m_rootDim + 7) >> 3;
    dparams.rootConstantBuffer = 0;
    dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_levels[0].m_normalField);
    NvFlowContextDispatch(context, &dparams);
}

void SDFGen::voxelize(NvFlowContext *context, const NvFlowSDFGenMeshParams *params) {
    auto rtv = params->renderTargetView;
    auto dsv = params->depthStencilView;
    auto rt = NvFlowRenderTargetViewGetRenderTarget(rtv);
    auto ds = NvFlowDepthStencilViewGetDepthStencil(dsv);
    uint32_t allocIdx = m_indexBuffers.allocateBack();
    auto &indexBuffer = m_indexBuffers[allocIdx];
    allocIdx = m_vertexBuffers.allocateBack();
    auto &vertexBuffer = m_vertexBuffers[allocIdx];

    NvFlowIndexBufferDesc indexBufDesc = {};
    indexBufDesc.data = params->indices;
    indexBufDesc.sizeInBytes = sizeof(NvFlowUint) * params->numIndices;
    indexBufDesc.format = eNvFlowFormat_r32_uint;
    indexBuffer = NvFlowCreateIndexBuffer(context, &indexBufDesc);

    float *verts = (float *)Allocable::allocate(8 * params->numVertices * sizeof(float));
    int posStride = params->positionStride / sizeof(float);
    int normStride = params->normalStride / sizeof(float);
    for (uint32_t i = 0; i < params->numVertices; ++i) {
        verts[8 * i] = params->positions[i * posStride];
        verts[8 * i + 1] = params->positions[i * posStride + 1];
        verts[8 * i + 2] = params->positions[i * posStride + 2];
        verts[8 * i + 3] = 1.f;
        verts[8 * i + 4] = params->normals[i * normStride];
        verts[8 * i + 5] = params->normals[i * normStride + 1];
        verts[8 * i + 6] = params->normals[i * normStride + 2];
        verts[8 * i + 7] = 0.f;
    }

    NvFlowVertexBufferDesc vertexBufDesc = {};
    vertexBufDesc.data = verts;
    vertexBufDesc.sizeInBytes = sizeof(float) * 8 * params->numVertices;
    vertexBuffer = NvFlowCreateVertexBuffer(context, &vertexBufDesc);

    Allocable::deallocate(verts);

    NvFlowFloat4x4 modelViewT = transpose(params->modelMatrix);
    NvFlowDim normalDim = m_rootDim;

    NvFlowDrawParams dparams = {};
    dparams.shader = m_meshNormalField;
    dparams.rootConstantBuffer = m_constantBuffer;
    dparams.ps_readWrite[0] = NvFlowTexture3DGetResourceRW(m_levels[0].m_normalField);

    for (int swizzleID = 0; swizzleID < 3; ++swizzleID) {
        auto mapped = (MeshNormalFieldShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->projection = modelViewT;
            mapped->normalFieldDim = make_float4(normalDim, 0);
            mapped->swizzleMode = make_uint4(swizzleID);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowViewport viewport = {};
        switch (swizzleID) {
            case 0:
                viewport.width = normalDim.x;
                viewport.height = normalDim.y;
                break;
            case 1:
                viewport.width = normalDim.y;
                viewport.height = normalDim.z;
                break;
            case 2:
                viewport.width = normalDim.z;
                viewport.height = normalDim.x;
                break;
        }
        NvFlowContextSetViewport(context, &viewport);
        NvFlowContextSetIndexBuffer(context, indexBuffer, 0);
        NvFlowContextSetVertexBuffer(context, vertexBuffer, 8 * sizeof(float), 0);
        NvFlowContextDrawIndexedInstanced(context, params->numIndices, 1, &dparams);
    }
    NvFlowContextSetRenderTarget(context, rt, ds);
}

void SDFGen::update(NvFlowContext *context) {
    while (stateUpdate(context))
        ;
}

NvFlowTexture3D *SDFGen::shape(NvFlowContext *context) {
    return m_distanceField;
}

SDFGen::SDFGen(NvFlowContext *context, const NvFlowSDFGenDesc *desc)
    : m_clearTextureCS(0),
      m_downsampleCS(0),
      m_smoothCS(0),
      m_upsampleCS(0),
      m_blockCompactCS(0),
      m_blockUnsignedDistanceCS(0),
      m_compactCS(0),
      m_unsignedDistanceCS(0),
      m_meshNormalField(0),
      m_constantBuffer(0),
      m_numLevels(4),
      m_levels{},
      m_atomic(0),
      m_atomicConst(0),
      m_blockList(0),
      m_rangeList(0),
      m_cellList(0),
      m_blockDistanceField(0),
      m_distanceField(0),
      m_state(0),
      m_stateLevel(0),
      m_substate(0) {
    m_rootDim = desc->resolution;

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_clearTextureCS = createShader(NVFLOW_CREATE_SHADER_ARGS(clearTextureCS));
    m_downsampleCS = createShader(NVFLOW_CREATE_SHADER_ARGS(downsampleCS));
    m_smoothCS = createShader(NVFLOW_CREATE_SHADER_ARGS(smoothCS));
    m_upsampleCS = createShader(NVFLOW_CREATE_SHADER_ARGS(upsampleCS));
    m_blockCompactCS = createShader(NVFLOW_CREATE_SHADER_ARGS(blockCompactCS));
    m_blockUnsignedDistanceCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(blockUnsignedDistanceCS));
    m_compactCS = createShader(NVFLOW_CREATE_SHADER_ARGS(compactCS));
    m_unsignedDistanceCS = createShader(NVFLOW_CREATE_SHADER_ARGS(unsignedDistanceCS));

    {
        NvFlowInputElementDesc elementDesc[2];
        elementDesc[0].semanticName = "POSITION";
        elementDesc[0].format = eNvFlowFormat_r32g32b32a32_float;
        elementDesc[1].semanticName = "NORMAL";
        elementDesc[1].format = eNvFlowFormat_r32g32b32a32_float;
        NvFlowGraphicsShaderDesc shaderDesc = {};
        shaderDesc.inputElementDescs = elementDesc;
        shaderDesc.numInputElements = countof(elementDesc);
        shaderDesc.vs = g_meshNormalFieldVS;
        shaderDesc.vs_length = sizeof(g_meshNormalFieldVS);
        shaderDesc.ps = g_meshNormalFieldPS;
        shaderDesc.ps_length = sizeof(g_meshNormalFieldPS);
        shaderDesc.label = L"meshNormalFieldPS";
        shaderDesc.blendState.enable = false;
        shaderDesc.depthState.depthEnable = false;
        shaderDesc.depthState.depthWriteMask = eNvFlowDepthWriteMask_All;
        shaderDesc.numRenderTargets = 0;
        shaderDesc.renderTargetFormat[0] = eNvFlowFormat_r16g16b16a16_unorm;
        shaderDesc.depthStencilFormat = eNvFlowFormat_d32_float;
        shaderDesc.uavTarget = 1;
        shaderDesc.depthClipEnable = 1;
        m_meshNormalField = NvFlowCreateGraphicsShader(context, &shaderDesc);
    }

    {
        NvFlowConstantBufferDesc cbDesc = {};
        cbDesc.sizeInBytes = 256;
        cbDesc.uploadAccess = 1;
        m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);
    }

    NvFlowTexture3DDesc texDesc = {};
    texDesc.format = eNvFlowFormat_r32g32b32a32_float;
    texDesc.dim = m_rootDim;
    texDesc.uploadAccess = 0;
    texDesc.downloadAccess = 0;
    for (int i = 0; i < countof(m_levels); ++i) {
        auto &l = m_levels[i];
        l.m_dim = m_rootDim >> i;
        texDesc.dim = l.m_dim;
        texDesc.format = eNvFlowFormat_r8g8b8a8_snorm;
        l.m_normalField = NvFlowCreateTexture3D(context, &texDesc);
        texDesc.format = eNvFlowFormat_r32_float;
        l.m_signFieldFront = NvFlowCreateTexture3D(context, &texDesc);
        l.m_signFieldBack = NvFlowCreateTexture3D(context, &texDesc);
    }

    NvFlowBufferDesc atomicDesc = {};
    atomicDesc.format = eNvFlowFormat_r32_uint;
    atomicDesc.dim = 64;
    atomicDesc.downloadAccess = 0;
    atomicDesc.uploadAccess = 1;
    m_atomic = NvFlowCreateBuffer(context, &atomicDesc);
    atomicDesc.uploadAccess = 0;

    NvFlowConstantBufferDesc atomicConstDesc = {};
    atomicConstDesc.sizeInBytes = 256;
    atomicConstDesc.uploadAccess = 0;
    m_atomicConst = NvFlowCreateConstantBuffer(context, &atomicConstDesc);

    atomicDesc.dim = m_levels[3].m_dim.z * m_levels[3].m_dim.y * m_levels[3].m_dim.x;
    m_blockList = NvFlowCreateBuffer(context, &atomicDesc);

    texDesc.format = eNvFlowFormat_r32g32_uint;
    texDesc.dim = m_levels[3].m_dim;
    m_rangeList = NvFlowCreateTexture3D(context, &texDesc);

    atomicDesc.dim = m_rootDim.z * m_rootDim.y * m_rootDim.x;
    m_cellList = NvFlowCreateBuffer(context, &atomicDesc);

    texDesc.format = eNvFlowFormat_r32_float;
    texDesc.dim = m_levels[3].m_dim;
    m_blockDistanceField = NvFlowCreateTexture3D(context, &texDesc);

    texDesc.dim = m_levels[0].m_dim;
    m_distanceField = NvFlowCreateTexture3D(context, &texDesc);
}

SDFGen::~SDFGen() {
    SafeRelease(m_clearTextureCS);
    SafeRelease(m_downsampleCS);
    SafeRelease(m_smoothCS);
    SafeRelease(m_upsampleCS);
    SafeRelease(m_blockCompactCS);
    SafeRelease(m_blockUnsignedDistanceCS);
    SafeRelease(m_compactCS);
    SafeRelease(m_unsignedDistanceCS);
    SafeRelease(m_meshNormalField);
    SafeRelease(m_constantBuffer);

    SafeRelease(m_indexBuffers);
    SafeRelease(m_vertexBuffers);

    for (auto &l : m_levels)
        l.release();

    SafeRelease(m_atomic);
    SafeRelease(m_atomicConst);
    SafeRelease(m_blockList);
    SafeRelease(m_rangeList);
    SafeRelease(m_cellList);
    SafeRelease(m_blockDistanceField);
    SafeRelease(m_distanceField);
}

int SDFGen::stateUpdate(NvFlowContext *context) {
    if (!m_state) {
        NvFlowDispatchParams dparams = {};
        dparams.shader = m_downsampleCS;
        dparams.gridDim = (m_rootDim + 7) >> 3;
        dparams.rootConstantBuffer = 0;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_levels[0].m_normalField);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_levels[1].m_normalField);
        dparams.readWrite[1] = NvFlowTexture3DGetResourceRW(m_levels[2].m_normalField);
        dparams.readWrite[2] = NvFlowTexture3DGetResourceRW(m_levels[3].m_normalField);
        dparams.readWrite[3] = NvFlowTexture3DGetResourceRW(m_levels[3].m_signFieldFront);
        NvFlowContextDispatch(context, &dparams);
        m_state = 1;
        m_stateLevel = m_numLevels - 1;
    }

    if (m_state == 1) {
        auto &level = m_levels[m_stateLevel];
        auto mapped =
            (SmoothShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
        if (mapped) {
            mapped->signFieldDim = make_uint4(level.m_dim, 0);
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }
        NvFlowDispatchParams dst = {};
        dst.shader = m_smoothCS;
        dst.gridDim = (level.m_dim + 7) >> 3;
        dst.rootConstantBuffer = m_constantBuffer;
        dst.readOnly[0] = NvFlowTexture3DGetResource(level.m_signFieldFront);
        dst.readOnly[1] = NvFlowTexture3DGetResource(level.m_normalField);
        dst.readWrite[0] = NvFlowTexture3DGetResourceRW(level.m_signFieldBack);
        NvFlowContextDispatch(context, &dst);

        swap(level.m_signFieldFront, level.m_signFieldBack);
        ++m_substate;

        int iterations = 2;
        if (m_stateLevel == m_numLevels - 1)
            iterations = level.m_dim.z + level.m_dim.y + level.m_dim.x;
        if (m_substate >= iterations) {
            m_substate = 0;
            --m_stateLevel;
            m_state = 2;
            if (m_stateLevel == -1) {
                m_state = 3;
                m_stateLevel = 0;
            }
        }
    }

    if (m_state == 2) {
        auto &fineLevel = m_levels[m_stateLevel];
        auto &coarseLevel = m_levels[m_stateLevel + 1];
        NvFlowDispatchParams params = {};
        params.shader = m_upsampleCS;
        params.gridDim = (fineLevel.m_dim + 7) >> 3;
        params.rootConstantBuffer = 0;
        params.readOnly[0] = NvFlowTexture3DGetResource(coarseLevel.m_signFieldFront);
        params.readWrite[0] = NvFlowTexture3DGetResourceRW(fineLevel.m_signFieldBack);
        NvFlowContextDispatch(context, &params);

        swap(fineLevel.m_signFieldBack, fineLevel.m_signFieldFront);
        m_state = 1;
        m_substate = 0;
    }

    if (m_state == 3) {
        auto &blockLevel = m_levels[3];
        auto data = (UnsignedDistanceShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (data) {
            data->blockFieldDim = make_uint4((m_rootDim + 7) >> 3, 0);
            float x = m_rootDim.x;
            data->scale = make_float4(4.f) / x;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        int clearVal[64];
        ZeroMemory(clearVal, sizeof(clearVal));
        auto atomicData = NvFlowBufferMap(context, m_atomic);
        if (atomicData) {
            CopyMemory(atomicData, clearVal, sizeof(clearVal));
            NvFlowBufferUnmap(context, m_atomic);
        }

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_blockCompactCS;
        dparams.gridDim = (blockLevel.m_dim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(blockLevel.m_normalField);
        dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_blockList);
        dparams.readWrite[1] = NvFlowBufferGetResourceRW(m_atomic);
        NvFlowContextDispatch(context, &dparams);

        NvFlowContextCopyConstantBuffer(context, m_atomicConst, m_atomic);

        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_blockUnsignedDistanceCS;
        dparams.gridDim = (m_rootDim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.secondConstantBuffer = m_atomicConst;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_levels[0].m_normalField);
        dparams.readOnly[1] = NvFlowTexture3DGetResource(blockLevel.m_normalField);
        dparams.readOnly[2] = NvFlowBufferGetResource(m_blockList);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_blockDistanceField);
        NvFlowContextDispatch(context, &dparams);

        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_compactCS;
        dparams.gridDim = (m_rootDim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_levels[0].m_normalField);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_rangeList);
        dparams.readWrite[1] = NvFlowBufferGetResourceRW(m_cellList);
        dparams.readWrite[2] = NvFlowBufferGetResourceRW(m_atomic);
        NvFlowContextDispatch(context, &dparams);

        ZeroMemory(&dparams, sizeof(dparams));
        dparams.shader = m_unsignedDistanceCS;
        dparams.gridDim = (m_rootDim + 7) >> 3;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.secondConstantBuffer = m_atomicConst;
        dparams.readOnly[0] = NvFlowTexture3DGetResource(m_levels[0].m_normalField);
        dparams.readOnly[1] = NvFlowTexture3DGetResource(blockLevel.m_normalField);
        dparams.readOnly[2] = NvFlowTexture3DGetResource(m_blockDistanceField);
        dparams.readOnly[3] = NvFlowTexture3DGetResource(m_rangeList);
        dparams.readOnly[4] = NvFlowBufferGetResource(m_cellList);
        dparams.readOnly[5] = NvFlowTexture3DGetResource(m_levels[0].m_signFieldFront);
        dparams.readOnly[6] = NvFlowBufferGetResource(m_blockList);
        dparams.readWrite[0] = NvFlowTexture3DGetResourceRW(m_distanceField);
        NvFlowContextDispatch(context, &dparams);

        m_state = 0;
        m_substate = 0;
    }

    return m_state;
}

NvFlowSDFGen *FlowCreateSDFGen(NvFlowContext *context, const NvFlowSDFGenDesc *desc) {
    return new SDFGen(context, desc);
}

}  // namespace NvFlow
