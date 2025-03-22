#include "VorticityConfinement.h"
#include "SparseTexturePool.h"
#include "Object.h"
#include "ClientHelper.h"

namespace NvFlow {

struct VorticityConfinementShaderParams {
    NvFlowShaderPointParams velocityParams;
    float scale;
    float velocityMask;
    float temperatureMask;
    float smokeMask;
    float fuelMask;
    float constantMask;
    float pad1;
    float pad2;
    NvFlowShaderLinearParams densityParams;
};

struct VorticityConfinementImpl : Object, VorticityConfinement {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    void execute(NvFlowContext *context, SparseTextureFront *velocity,
                 SparseTextureFront *coarseDensity,
                 const VorticityConfinementParams *params) override;

    // Details

    VorticityConfinementImpl(NvFlowContext *context, const VorticityConfinementDesc *desc);
    ~VorticityConfinementImpl();

    NvFlow::VorticityConfinementDesc m_desc;
    NvFlowComputeShader *m_vorticityConfinementCS;
    NvFlowComputeShader *m_vorticityConfinementCS_noDensity;
    NvFlowConstantBuffer *m_constantBuffer;
};

uint64_t VorticityConfinementImpl::getGPUBytesUsed() {
    return 0;
}

void VorticityConfinementImpl::execute(NvFlowContext *context, SparseTextureFront *velocity,
                                       SparseTextureFront *coarseDensity,
                                       const VorticityConfinementParams *params) {
    auto outTexture = velocity->acquireTexture(context);
    auto densityReadOnly = coarseDensity->front.readLinearHandle(context);
    auto densityReadOnlyLayeredView = densityReadOnly.layeredView();
    auto velocityOutHandle = outTexture.writePointHandle(context);
    auto velocityInHandle = velocity->front.readPointHandle(context);
    auto velocityOutLayeredView = velocityOutHandle.layeredView();
    auto velocityInLayeredView = velocityInHandle.layeredView();

    for (uint32_t layerIdx = 0; layerIdx < velocityOutHandle.numLayers; ++layerIdx) {
        VorticityConfinementPerLayerParams layerParams;
        params->getPerLayer(&layerParams, params->userdata, layerIdx);
        auto velocityOutLayerView = velocityOutHandle.layerView(layerIdx);
        auto velocityInLayerView = velocityInHandle.layerView(layerIdx);
        auto densityReadOnlyLayerView = densityReadOnly.layerView(layerIdx);

        auto mapped = (VorticityConfinementShaderParams *)NvFlowConstantBufferMap(
            context, m_constantBuffer);
        if (mapped) {
            mapped->velocityParams = velocityInLayeredView.pointParams;
            mapped->scale = params->deltaTime * layerParams.forceScale;
            mapped->velocityMask = layerParams.velocityMask;
            mapped->temperatureMask = layerParams.temperatureMask;
            mapped->smokeMask = layerParams.smokeMask;
            mapped->fuelMask = layerParams.fuelMask;
            mapped->constantMask = layerParams.constantMask;
            mapped->pad1 = 0;
            mapped->pad2 = 0;
            mapped->densityParams = densityReadOnlyLayeredView.params;
            NvFlowConstantBufferUnmap(context, m_constantBuffer);
        }

        NvFlowDim gridDim;
        gridDim.x = (velocityOutLayeredView.params.blockDim.x *
                         velocityOutLayerView.mapping.numBlocks +
                     7) >>
                    3;
        gridDim.y = (velocityOutLayeredView.params.blockDim.y + 7) >> 3;
        gridDim.z = (velocityOutLayeredView.params.blockDim.z + 7) >> 3;

        NvFlowDispatchParams dparams = {};
        dparams.shader = m_vorticityConfinementCS;
        if (layerParams.temperatureMask == 0.f && layerParams.smokeMask == 0.f &&
            layerParams.fuelMask == 0.f) {
            dparams.shader = m_vorticityConfinementCS_noDensity;
        }
        dparams.gridDim = gridDim;
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = velocityOutLayerView.mapping.blockList;
        dparams.readOnly[1] = velocityOutLayerView.mapping.blockTable;
        dparams.readOnly[2] = velocityInLayerView.data;
        dparams.readOnly[3] = densityReadOnlyLayerView.mapping.blockTable;
        dparams.readOnly[4] = densityReadOnlyLayerView.data;
        dparams.readWrite[0] = velocityOutLayerView.data;
        NvFlowContextDispatch(context, &dparams);
    }

    velocity->swap(outTexture);
}

#include "vorticityConfinementCS.hlsl.h"
#include "vorticityConfinementCS_noDensity.hlsl.h"

VorticityConfinementImpl::VorticityConfinementImpl(NvFlowContext *context,
                                                   const VorticityConfinementDesc *desc)
    : m_vorticityConfinementCS(0),
      m_vorticityConfinementCS_noDensity(0),
      m_constantBuffer(0) {
    m_desc = *desc;

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_vorticityConfinementCS =
        createShader(NVFLOW_CREATE_SHADER_ARGS(vorticityConfinementCS));
    m_vorticityConfinementCS_noDensity =
        createShader(NVFLOW_CREATE_SHADER_ARGS(vorticityConfinementCS_noDensity));

    NvFlowConstantBufferDesc bufDesc;
    bufDesc.sizeInBytes = sizeof(VorticityConfinementShaderParams);
    bufDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &bufDesc);
}

VorticityConfinementImpl::~VorticityConfinementImpl() {
    SafeRelease(m_vorticityConfinementCS);
    SafeRelease(m_vorticityConfinementCS_noDensity);
    SafeRelease(m_constantBuffer);
}

VorticityConfinement *createVorticityConfinement(NvFlowContext *context,
                                                 const VorticityConfinementDesc *desc) {
    return new VorticityConfinementImpl(context, desc);
}

}  // namespace NvFlow