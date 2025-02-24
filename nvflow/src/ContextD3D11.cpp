#include "ContextD3D11.h"
#include "Types.h"

namespace NvFlow {

struct BlendEnumTableD3D11 {
    BlendEnumTableD3D11() {
        this->table[1] = D3D11_BLEND_ZERO;
        this->table[2] = D3D11_BLEND_ONE;
        this->table[3] = D3D11_BLEND_SRC_ALPHA;
        this->table[4] = D3D11_BLEND_INV_SRC_ALPHA;
        this->table[5] = D3D11_BLEND_DEST_ALPHA;
        this->table[6] = D3D11_BLEND_INV_DEST_ALPHA;
    }

    D3D11_BLEND table[7];
} const g_blendEnumTableD3D11;

struct BlendOpEnumTableD3D11 {
    BlendOpEnumTableD3D11() {
        this->table[1] = D3D11_BLEND_OP_ADD;
        this->table[2] = D3D11_BLEND_OP_SUBTRACT;
        this->table[3] = D3D11_BLEND_OP_REV_SUBTRACT;
        this->table[4] = D3D11_BLEND_OP_MIN;
        this->table[5] = D3D11_BLEND_OP_MAX;
    }

    D3D11_BLEND_OP table[6];
} const g_blendOpEnumTableD3D11;

struct ComparisonTableD3D11 {
    ComparisonTableD3D11() {
        this->table[1] = D3D11_COMPARISON_NEVER;
        this->table[2] = D3D11_COMPARISON_LESS;
        this->table[3] = D3D11_COMPARISON_EQUAL;
        this->table[4] = D3D11_COMPARISON_LESS_EQUAL;
        this->table[5] = D3D11_COMPARISON_GREATER;
        this->table[6] = D3D11_COMPARISON_NOT_EQUAL;
        this->table[7] = D3D11_COMPARISON_GREATER_EQUAL;
        this->table[8] = D3D11_COMPARISON_ALWAYS;
    }

    D3D11_COMPARISON_FUNC table[9];
} g_comparisonTableD3D11;

NvFlowDim ContextD3D11::extractDim(ID3D11Resource *resource) {
    D3D11_RESOURCE_DIMENSION type;
    resource->GetType(&type);
    switch (type) {
        case D3D11_RESOURCE_DIMENSION_BUFFER: {
            ComPtr<ID3D11Buffer> buffer;
            resource->QueryInterface(IID_PPV_ARGS(&buffer));
            D3D11_BUFFER_DESC buffDesc;
            buffer->GetDesc(&buffDesc);
            return {buffDesc.ByteWidth, 0, 0};
        }
        case D3D11_RESOURCE_DIMENSION_TEXTURE1D: {
            ComPtr<ID3D11Texture1D> texture;
            resource->QueryInterface(IID_PPV_ARGS(&texture));
            D3D11_TEXTURE1D_DESC texDesc;
            texture->GetDesc(&texDesc);
            return {texDesc.Width, texDesc.ArraySize, 0};
        }
        case D3D11_RESOURCE_DIMENSION_TEXTURE2D: {
            ComPtr<ID3D11Texture2D> texture;
            resource->QueryInterface(IID_PPV_ARGS(&texture));
            D3D11_TEXTURE2D_DESC texDesc;
            texture->GetDesc(&texDesc);
            return {texDesc.Width, texDesc.Height, texDesc.ArraySize};
        }
        case D3D11_RESOURCE_DIMENSION_TEXTURE3D: {
            ComPtr<ID3D11Texture3D> texture;
            resource->QueryInterface(IID_PPV_ARGS(&texture));
            D3D11_TEXTURE3D_DESC texDesc;
            texture->GetDesc(&texDesc);
            return {texDesc.Width, texDesc.Height, texDesc.Depth};
        }

        default:
            return {0, 0, 0};
    }
}

NvFlowDim ContextD3D11::extractDim(ID3D11View *srv) {
    ComPtr<ID3D11Resource> resource;
    srv->GetResource(&resource);
    return extractDim(resource.Get());
}

ContextD3D11::ContextD3D11(const NvFlowContextDescD3D11 *pdesc) {
    updateContext(pdesc);

    auto createSampler = [this](D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE mode =
                                                         D3D11_TEXTURE_ADDRESS_CLAMP) {
        D3D11_SAMPLER_DESC samplerDesc = {filter,
                                          mode,
                                          mode,
                                          mode,
                                          0.f,
                                          0,
                                          D3D11_COMPARISON_NEVER,
                                          {0.f, 0.f, 0.f, 0.f},
                                          0,
                                          D3D11_FLOAT32_MAX};
        ComPtr<ID3D11SamplerState> sampler;
        m_device->CreateSamplerState(&samplerDesc, &sampler);
        return sampler;
    };

    m_sampler0 =
        createSampler(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER);
    m_sampler1 =
        createSampler(D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER);
    m_sampler2 =
        createSampler(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_WRAP);
    m_sampler3 = createSampler(D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_WRAP);
    m_sampler4 =
        createSampler(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP);
    m_sampler5 = createSampler(D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP);
}

ID3D11Device *ContextD3D11::getDevice() {
    return m_device.Get();
}

ID3D11DeviceContext *ContextD3D11::getContext() {
    return m_deviceContext.Get();
}

NvFlowContextAPI ContextD3D11::getContextType() {
    return eNvFlowContextD3D11;
}

void ContextD3D11::clearDepthStencil(DepthStencil *dsv, float depth) {
    auto dsvd3d11 = implCast<DepthStencilD3D11>(dsv);
    m_deviceContext->ClearDepthStencilView(dsvd3d11->m_dsv.Get(), D3D11_CLEAR_DEPTH, depth,
                                           0);
}

void ContextD3D11::clearRenderTarget(RenderTarget *rtv, const float color[4]) {
    auto rtvd3d11 = implCast<RenderTargetD3D11>(rtv);
    m_deviceContext->ClearRenderTargetView(rtvd3d11->m_rtv.Get(), color);
}

void ContextD3D11::contextPush() {
    auto ctx = m_deviceContext.Get();
    ctx->IAGetIndexBuffer(&m_state.indexBuffer, &m_state.indexFormat, &m_state.indexOffset);
    ctx->IAGetInputLayout(&m_state.layout);
    ctx->IAGetPrimitiveTopology(&m_state.topology);

    m_state.vertexBuffers.resize(16);
    m_state.vertexStrides.resize(16);
    m_state.vertexOffsets.resize(16);

    ctx->IAGetVertexBuffers(0, 16, m_state.vertexBuffers.data(),
                            m_state.vertexStrides.data(), m_state.vertexOffsets.data());

    ctx->OMGetBlendState(&m_state.blendState, m_state.blendFactor,
                         &m_state.blendSampleMask);

    ctx->OMGetDepthStencilState(&m_state.depthStencilState, &m_state.stencilRef);

    m_state.renderTargetViews.resize(8);
    ctx->OMGetRenderTargets(8, m_state.renderTargetViews.data(), &m_state.depthStencilView);

    m_state.scissorRects.reserve(16);
    UINT numScissorRects;
    ctx->RSGetScissorRects(&numScissorRects, nullptr);
    m_state.scissorRects.resize(numScissorRects);
    ctx->RSGetScissorRects(&numScissorRects, m_state.scissorRects.data());

    ctx->RSGetState(&m_state.rasterizerState);

    m_state.viewports.reserve(16);
    UINT numViewports;
    ctx->RSGetViewports(&numViewports, nullptr);
    m_state.viewports.resize(numViewports);
    ctx->RSGetViewports(&numViewports, m_state.viewports.data());

    m_state.vsConstantBuffers.resize(8);
    ctx->VSGetConstantBuffers(0, 8, m_state.vsConstantBuffers.data());
    m_state.vsSamplers.resize(8);
    ctx->VSGetSamplers(0, 8, m_state.vsSamplers.data());
    m_state.vsSrvs.resize(16);
    ctx->VSGetShaderResources(0, 16, m_state.vsSrvs.data());

    m_state.psConstantBuffers.resize(8);
    ctx->PSGetConstantBuffers(0, 8, m_state.psConstantBuffers.data());
    m_state.psSamplers.resize(8);
    ctx->PSGetSamplers(0, 8, m_state.psSamplers.data());
    m_state.psSrvs.resize(16);
    ctx->PSGetShaderResources(0, 16, m_state.psSrvs.data());

    m_state.csConstantBuffers.resize(8);
    ctx->CSGetConstantBuffers(0, 8, m_state.csConstantBuffers.data());
    m_state.csSamplers.resize(8);
    ctx->CSGetSamplers(0, 8, m_state.csSamplers.data());
    m_state.csSrvs.resize(16);
    ctx->CSGetShaderResources(0, 16, m_state.csSrvs.data());
    m_state.csUavs.resize(8);
    ctx->CSGetUnorderedAccessViews(0, 8, m_state.csUavs.data());

    ctx->CSGetShader(&m_state.computeShader, nullptr, nullptr);
    ctx->VSGetShader(&m_state.vertexShader, nullptr, nullptr);
    ctx->PSGetShader(&m_state.pixelShader, nullptr, nullptr);
    ctx->GSGetShader(&m_state.geometryShader, nullptr, nullptr);
    ctx->HSGetShader(&m_state.hullShader, nullptr, nullptr);
    ctx->DSGetShader(&m_state.domainShader, nullptr, nullptr);

    //
    ctx->RSSetState(nullptr);
    ctx->OMSetBlendState(nullptr, nullptr, UINT(-1));
    ctx->OMSetDepthStencilState(nullptr, 0);

    ctx->CSSetShader(nullptr, nullptr, 0);
    ctx->VSSetShader(nullptr, nullptr, 0);
    ctx->PSSetShader(nullptr, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);
}

void ContextD3D11::contextPop() {
    auto ctx = m_deviceContext.Get();
    ctx->IASetInputLayout(m_state.layout);
    ctx->IASetPrimitiveTopology(m_state.topology);
    ctx->IASetVertexBuffers(0, m_state.vertexBuffers.size(), m_state.vertexBuffers.data(),
                            m_state.vertexStrides.data(), m_state.vertexOffsets.data());
    ctx->OMSetBlendState(m_state.blendState, m_state.blendFactor, m_state.blendSampleMask);
    ctx->OMSetDepthStencilState(m_state.depthStencilState, m_state.stencilRef);
    ctx->OMSetRenderTargets(8, m_state.renderTargetViews.data(), m_state.depthStencilView);
    ctx->RSSetScissorRects(m_state.scissorRects.size(), m_state.scissorRects.data());
    ctx->RSSetState(m_state.rasterizerState);
    ctx->RSSetViewports(m_state.viewports.size(), m_state.viewports.data());

    ctx->VSSetConstantBuffers(0, 8, m_state.vsConstantBuffers.data());
    ctx->VSSetSamplers(0, 8, m_state.vsSamplers.data());
    ctx->VSSetShaderResources(0, 16, m_state.vsSrvs.data());

    ctx->PSSetConstantBuffers(0, 8, m_state.psConstantBuffers.data());
    ctx->PSSetSamplers(0, 8, m_state.psSamplers.data());
    ctx->PSSetShaderResources(0, 16, m_state.psSrvs.data());

    ctx->CSSetConstantBuffers(0, 8, m_state.csConstantBuffers.data());
    ctx->CSSetSamplers(0, 8, m_state.csSamplers.data());
    ctx->CSSetShaderResources(0, 16, m_state.csSrvs.data());
    ctx->CSSetUnorderedAccessViews(0, 8, m_state.csUavs.data(), nullptr);

    ctx->CSSetShader(m_state.computeShader, nullptr, 0);
    ctx->VSSetShader(m_state.vertexShader, nullptr, 0);
    ctx->HSSetShader(m_state.hullShader, nullptr, 0);
    ctx->DSSetShader(m_state.domainShader, nullptr, 0);
    ctx->GSSetShader(m_state.geometryShader, nullptr, 0);
    ctx->PSSetShader(m_state.pixelShader, nullptr, 0);

    m_state.reset();
}

void ContextD3D11::copy(Buffer *dst, Buffer *src, uint32_t offset, uint32_t numBytes) {
    auto dstBufD3D11 = implCast<BufferD3D11>(dst);
    auto srcBufD3D11 = implCast<BufferD3D11>(src);
    D3D11_BOX box = {};
    box.left = offset;
    box.right = numBytes + offset;
    box.top = 0;
    box.bottom = 1;
    box.front = 0;
    box.back = 1;

    if (numBytes)
        m_deviceContext->CopySubresourceRegion(dstBufD3D11->m_buffer.Get(), 0, offset, 0, 0,
                                               srcBufD3D11->m_buffer.Get(), 0, &box);
}

void ContextD3D11::copy(Buffer *dst, Resource *src, uint32_t offset, uint32_t numBytes) {
    auto dstBufD3D11 = implCast<BufferD3D11>(dst);
    auto srcBufD3D11 = implCast<ResourceD3D11>(src);
    D3D11_BOX box = {};
    box.left = offset;
    box.right = offset + numBytes;
    box.top = 0;
    box.bottom = 1;
    box.front = 0;
    box.back = 1;

    ComPtr<ID3D11Resource> srcResource;
    srcBufD3D11->m_srv->GetResource(&srcResource);

    if (numBytes)
        m_deviceContext->CopySubresourceRegion(dstBufD3D11->m_buffer.Get(), 0, offset, 0, 0,
                                               srcResource.Get(), 0, &box);
}

void ContextD3D11::copy(ConstantBuffer *dstIn, Buffer *srcIn) {
    auto dst = implCast<ConstantBufferD3D11>(dstIn);
    auto src = implCast<BufferD3D11>(srcIn);
    m_deviceContext->CopyResource(dst->m_buffer.Get(), src->m_buffer.Get());
}

void ContextD3D11::copy(DepthStencil *dstIn, Resource *srcIn) {
    auto dst = implCast<DepthStencilD3D11>(dstIn);
    auto src = implCast<ResourceD3D11>(srcIn);
    ComPtr<ID3D11Resource> srcResource, dstResource;
    dst->m_dsv->GetResource(&dstResource);
    src->m_srv->GetResource(&srcResource);
    m_deviceContext->CopyResource(dstResource.Get(), srcResource.Get());
}

void ContextD3D11::copy(ResourceRW *dstIn, Resource *srcIn) {
    auto dst = implCast<ResourceRWD3D11>(dstIn);
    auto src = implCast<ResourceD3D11>(srcIn);
    ComPtr<ID3D11Resource> srcResource, dstResource;
    dst->m_srv->GetResource(&dstResource);
    src->m_srv->GetResource(&srcResource);
    m_deviceContext->CopyResource(dstResource.Get(), srcResource.Get());
}

void ContextD3D11::copy(Texture3D *dstIn, Texture3D *srcIn) {
    auto dst = implCast<Texture3DD3D11>(dstIn);
    auto src = implCast<Texture3DD3D11>(srcIn);
    m_deviceContext->CopyResource(dst->m_texture.Get(), src->m_texture.Get());
}

void ContextD3D11::copy(Texture3D *dstIn, Resource *srcIn) {
    auto dst = implCast<Texture3DD3D11>(dstIn);
    auto src = implCast<ResourceD3D11>(srcIn);
    ComPtr<ID3D11Resource> srcResource;
    src->m_srv->GetResource(&srcResource);
    m_deviceContext->CopyResource(dst->m_texture.Get(), srcResource.Get());
}

void ContextD3D11::copyFromShared(Texture2D *dstTexture,
                                  Texture2DCrossAdapter *sharedTexture, uint32_t height) {}

void ContextD3D11::copyToShared(Texture2DCrossAdapter *dstSharedTexture,
                                Texture2D *srcTexture, uint32_t height) {
    return;
}

Buffer *ContextD3D11::createBuffer(const NvFlowBufferDesc *desc) {
    return new BufferD3D11(this, desc);
}

Buffer *ContextD3D11::createBufferView(Buffer *buffer, const NvFlowBufferViewDesc *desc) {
    return new BufferD3D11(this, implCast<BufferD3D11>(buffer), desc);
}

ColorBuffer *ContextD3D11::createColorBuffer(const NvFlowColorBufferDesc *desc) {
    return new ColorBufferD3D11(this, desc);
}

ComputeShader *ContextD3D11::createComputeShader(const NvFlowComputeShaderDesc *desc) {
    return new ComputeShaderD3D11(this, desc);
}

ConstantBuffer *ContextD3D11::createConstantBuffer(const NvFlowConstantBufferDesc *desc) {
    return new ConstantBufferD3D11(this, desc);
}

DepthBuffer *ContextD3D11::createDepthBuffer(const NvFlowDepthBufferDesc *desc) {
    return new DepthBufferD3D11(this, desc);
}

EventQueue *ContextD3D11::createEventQueue() {
    return new EventQueueD3D11(this);
}

Fence *ContextD3D11::createFence(const NvFlowFenceDesc *desc) {
    return nullptr;
}

GraphicsShader *ContextD3D11::createGraphicsShader(const NvFlowGraphicsShaderDesc *desc) {
    return new GraphicsShaderD3D11(this, desc);
}

HeapVTR *ContextD3D11::createHeapVTR(const NvFlowHeapSparseDesc *desc) {
    return new HeapVTRD3D11(this, desc);
}

IndexBuffer *ContextD3D11::createIndexBuffer(const NvFlowIndexBufferDesc *desc) {
    return new IndexBufferD3D11(this, desc);
}

Texture1D *ContextD3D11::createTexture1D(const NvFlowTexture1DDesc *desc) {
    return new Texture1DD3D11(this, desc);
}

Texture2D *ContextD3D11::createTexture2D(const NvFlowTexture2DDesc *desc) {
    return new Texture2DD3D11(this, desc, false);
}

Texture2DCrossAdapter *ContextD3D11::createTexture2DCrossAdapter(
    const NvFlowTexture2DDesc *desc) {
    return nullptr;
}

Texture2D *ContextD3D11::createTexture2DShared(const NvFlowTexture2DDesc *desc) {
    return new Texture2DD3D11(this, desc, true);
}

Texture3D *ContextD3D11::createTexture3D(const NvFlowTexture3DDesc *desc) {
    return new Texture3DD3D11(this, desc);
}

Texture3DVTR *ContextD3D11::createTexture3DVTR(const NvFlowTexture3DSparseDesc *desc) {
    return new Texture3DVTRD3D11(this, desc);
}

ResourceReference *ContextD3D11::shareResourceReference(Resource *resource) {
    return new ResourceReferenceD3D11(this, implCast<ResourceD3D11>(resource));
}

Timer *ContextD3D11::createTimer() {
    return new TimerD3D11(this);
}

VertexBuffer *ContextD3D11::createVertexBuffer(const NvFlowVertexBufferDesc *desc) {
    return new VertexBufferD3D11(this, desc);
}

void ContextD3D11::dispatch(const NvFlowDispatchParams *params) {
    auto shader = implCast<ComputeShaderD3D11>(params->shader);
    profileItemBegin(shader->m_desc.label);

    if (params->gridDim.x && params->gridDim.y && params->gridDim.z) {
        constexpr UINT maxReadSlots = 16;
        constexpr UINT maxWriteSlots = 8;
        ID3D11ShaderResourceView *srvs[maxReadSlots];
        ID3D11UnorderedAccessView *uavs[maxWriteSlots];

        auto context = getContext();
        context->CSSetShader(shader->m_cs.Get(), nullptr, 0);

        ZeroMemory(srvs, sizeof(srvs));
        ZeroMemory(uavs, sizeof(uavs));
        for (int i = 0; i < maxReadSlots; ++i) {
            auto r = implCast<ResourceD3D11>(params->readOnly[i]);
            if (r) srvs[i] = r->m_srv.Get();
        }

        for (int i = 0; i < maxWriteSlots; ++i) {
            auto r = implCast<ResourceRWD3D11>(params->readWrite[i]);
            if (r) uavs[i] = r->m_uav.Get();
        }

        context->CSSetShaderResources(0, maxReadSlots, srvs);
        context->CSSetUnorderedAccessViews(0, maxWriteSlots, uavs, nullptr);

        if (params->rootConstantBuffer) {
            auto cbv =
                implCast<ConstantBufferD3D11>(params->rootConstantBuffer)->m_buffer.Get();
            context->CSSetConstantBuffers(0, 1, &cbv);
        }
        if (params->secondConstantBuffer) {
            auto cbv =
                implCast<ConstantBufferD3D11>(params->secondConstantBuffer)->m_buffer.Get();
            context->CSSetConstantBuffers(1, 1, &cbv);
        }

        context->CSSetSamplers(0, 6, m_sampler0.GetAddressOf());

        if (m_profiler) m_profiler->begin(shader->m_desc.label);
        context->Dispatch(params->gridDim.x, params->gridDim.y, params->gridDim.z);
        if (m_profiler) m_profiler->end();

        // Reset SRVS and UAVS
        ZeroMemory(srvs, sizeof(srvs));
        ZeroMemory(uavs, sizeof(uavs));
        context->CSSetShaderResources(0, maxReadSlots, srvs);
        context->CSSetUnorderedAccessViews(0, maxWriteSlots, uavs, nullptr);
    }
    profileItemEnd();
}

void ContextD3D11::download(Buffer *buffer) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);
    if (buffer->m_desc.downloadAccess)
        m_deviceContext->CopyResource(bufD3D11->m_downloadBuffer.Get(),
                                      bufD3D11->m_buffer.Get());
}

void ContextD3D11::download(Buffer *buffer, uint32_t offset, uint32_t numBytes) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);
    if (buffer->m_desc.downloadAccess) {
        D3D11_BOX box;
        box.left = offset;
        box.right = numBytes + offset;
        box.top = 0;
        box.bottom = 1;
        box.front = 0;
        box.back = 1;
        if (numBytes)
            m_deviceContext->CopySubresourceRegion(bufD3D11->m_downloadBuffer.Get(), 0,
                                                   offset, 0, 0, bufD3D11->m_buffer.Get(),
                                                   0, &box);
    }
}

void ContextD3D11::download(Texture3D *buffer) {
    auto texD3D11 = implCast<Texture3DD3D11>(buffer);
    if (buffer->m_desc.downloadAccess) {
        m_deviceContext->CopyResource(texD3D11->m_downloadTexture.Get(),
                                      texD3D11->m_texture.Get());
    }
}

void ContextD3D11::drawIndexedInstanced(uint32_t indicesPerInstance, uint32_t numInstances,
                                        const NvFlowDrawParams *params) {
    auto shader = implCast<GraphicsShaderD3D11>(params->shader);
    profileItemBegin(shader->m_desc.label);

    constexpr UINT maxSlots = 16;
    const UINT maxWriteSlots = 1;
    BOOL isTargetUAV = FALSE;
    auto context = getContext();
    auto vs = shader->m_vs.Get();
    auto ps = shader->m_ps.Get();
    context->IASetInputLayout(shader->m_inputLayout.Get());
    context->OMSetBlendState(shader->m_blenderState.Get(), nullptr, ~0);
    context->OMSetDepthStencilState(shader->m_depthStencilState.Get(), 0);
    if (params->frontCounterClockwise)
        context->RSSetState(shader->m_rasterizerStateRH.Get());
    else
        context->RSSetState(shader->m_rasterizerStateLH.Get());

    context->VSSetShader(vs, nullptr, 0);
    context->PSSetShader(ps, nullptr, 0);

    if (shader->m_desc.lineList)
        context->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
    else
        context->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    ID3D11ShaderResourceView *vs_srvs[maxSlots];
    ID3D11ShaderResourceView *ps_srvs[maxSlots];
    ID3D11UnorderedAccessView *ps_uavs[maxWriteSlots];

    ZeroMemory(vs_srvs, sizeof(vs_srvs));
    ZeroMemory(ps_srvs, sizeof(ps_srvs));

    for (int i = 0; i < maxSlots; ++i) {
        auto r = implCast<ResourceD3D11>(params->vs_readOnly[i]);
        if (r) vs_srvs[i] = r->m_srv.Get();
    }
    for (int i = 0; i < maxSlots; ++i) {
        auto r = implCast<ResourceD3D11>(params->ps_readOnly[i]);
        if (r) ps_srvs[i] = r->m_srv.Get();
    }
    for (int i = 0; i < maxWriteSlots; ++i) {
        auto r = implCast<ResourceRWD3D11>(params->ps_readWrite[i]);
        if (r) {
            ps_uavs[i] = r->m_uav.Get();
            isTargetUAV = TRUE;
        }
    }

    context->VSSetShaderResources(0, maxSlots, vs_srvs);
    context->PSSetShaderResources(0, maxSlots, ps_srvs);
    if (isTargetUAV)
        context->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, 0, 0, maxWriteSlots,
                                                           ps_uavs, nullptr);

    if (params->rootConstantBuffer) {
        auto rootConstantBuffer = implCast<ConstantBufferD3D11>(params->rootConstantBuffer);
        auto cbv = rootConstantBuffer->m_buffer.Get();
        context->VSSetConstantBuffers(0, 1, &cbv);
        context->PSSetConstantBuffers(0, 1, &cbv);
    }

    context->VSSetSamplers(0, 6, m_sampler0.GetAddressOf());
    context->PSSetSamplers(0, 6, m_sampler0.GetAddressOf());

    if (m_profiler) m_profiler->begin(shader->m_desc.label);
    context->DrawIndexedInstanced(indicesPerInstance, numInstances, 0, 0, 0);
    if (m_profiler) m_profiler->end();

    ZeroMemory(ps_srvs, sizeof(ps_srvs));
    ZeroMemory(ps_uavs, sizeof(ps_uavs));
    context->VSSetShaderResources(0, maxSlots, ps_srvs);
    context->PSSetShaderResources(0, maxSlots, ps_srvs);
    if (isTargetUAV)
        context->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, 0, 0, 1, ps_uavs,
                                                           nullptr);

    profileItemEnd();
}

void ContextD3D11::eventQueuePush(EventQueue *eventQueueIn, uint64_t uid) {
    auto eventQueue = implCast<EventQueueD3D11>(eventQueueIn);
    eventQueue->push(uid, this);
}

int ContextD3D11::eventQueuePop(EventQueue *eventQueueIn, uint64_t *pUid) {
    auto eventQueue = implCast<EventQueueD3D11>(eventQueueIn);
    return eventQueue->pop(pUid, this);
}

int ContextD3D11::is_VTR_supported() {
    auto checkVTR = [](ID3D11Device *device) {
        D3D11_FEATURE_DATA_D3D11_OPTIONS2 options;
        HRESULT hr = device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS2, &options,
                                                 sizeof(options));
        return SUCCEEDED(hr) && options.TiledResourcesTier >= D3D11_TILED_RESOURCES_TIER_3;
    };

    return checkVTR(getDevice());
}

NvFlowMappedData *ContextD3D11::map(NvFlowMappedData *result, Texture3D *buffer) {
    auto texD3D11 = implCast<Texture3DD3D11>(buffer);
    if (texD3D11->m_desc.uploadAccess) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        getContext()->Map(texD3D11->m_uploadTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
                          &mapped);
        memcpy(result, &mapped, sizeof(mapped));
    } else {
        ZeroMemory(result, sizeof(NvFlowMappedData));
    }
    return result;
}

void *ContextD3D11::map(Buffer *buffer) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);
    if (!bufD3D11->m_desc.uploadAccess) return nullptr;

    D3D11_MAPPED_SUBRESOURCE mapped;
    getContext()->Map(bufD3D11->m_uploadBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
                      &mapped);
    return mapped.pData;
}

void *ContextD3D11::map(ConstantBuffer *buffer) {
    auto bufD3D11 = implCast<ConstantBufferD3D11>(buffer);
    D3D11_MAPPED_SUBRESOURCE mapped;
    getContext()->Map(bufD3D11->m_buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    return mapped.pData;
}

void *ContextD3D11::map(IndexBuffer *buffer) {
    auto bufD3D11 = implCast<IndexBufferD3D11>(buffer);
    D3D11_MAPPED_SUBRESOURCE mapped;
    getContext()->Map(bufD3D11->m_uploadBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
                      &mapped);
    return mapped.pData;
}

void *ContextD3D11::map(Texture1D *buffer) {
    auto bufD3D11 = implCast<Texture1DD3D11>(buffer);
    if (!bufD3D11->m_desc.uploadAccess) return nullptr;

    D3D11_MAPPED_SUBRESOURCE mapped;
    getContext()->Map(bufD3D11->m_uploadTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
                      &mapped);
    return mapped.pData;
}

void *ContextD3D11::map(VertexBuffer *buffer) {
    auto bufD3D11 = implCast<VertexBufferD3D11>(buffer);
    D3D11_MAPPED_SUBRESOURCE mapped;
    getContext()->Map(bufD3D11->m_uploadBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
                      &mapped);
    return mapped.pData;
}

NvFlowMappedData *ContextD3D11::mapDownload(NvFlowMappedData *result, Texture3D *buffer) {
    auto texD3D11 = implCast<Texture3DD3D11>(buffer);
    if (texD3D11->m_desc.downloadAccess) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        getContext()->Map(texD3D11->m_downloadTexture.Get(), 0, D3D11_MAP_READ,
                          D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
        memcpy(result, &mapped, sizeof(mapped));
    } else {
        ZeroMemory(result, sizeof(NvFlowMappedData));
    }
    return result;
}

void *ContextD3D11::mapDownload(Buffer *buffer) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);

    if (!bufD3D11->m_desc.downloadAccess) return nullptr;

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(getContext()->Map(bufD3D11->m_downloadBuffer.Get(), 0, D3D11_MAP_READ,
                                    D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped))) {
        return mapped.pData;
    } else
        return nullptr;
}

void ContextD3D11::processFenceSignal(NvFlowContext *context) {}

void ContextD3D11::processFenceWait(NvFlowContext *context) {}

void ContextD3D11::restoreResourceState(Resource *resource) {}

void ContextD3D11::setFormats(GraphicsShader *graphicsShader,
                              NvFlowFormat renderTargetFormat,
                              NvFlowFormat depthStencilFormat) {
    auto shader = implCast<GraphicsShaderD3D11>(graphicsShader);
    shader->setFormats(renderTargetFormat, depthStencilFormat);
}

void ContextD3D11::setIndexBuffer(IndexBuffer *buffer, uint32_t offset) {
    auto bufD3D11 = implCast<IndexBufferD3D11>(buffer);
    getContext()->IASetIndexBuffer(bufD3D11->m_buffer.Get(),
                                   convertToDXGI(buffer->m_desc.format), offset);
}

void ContextD3D11::setVertexBuffer(VertexBuffer *buffer, uint32_t stride, uint32_t offset) {
    auto bufD3D11 = implCast<VertexBufferD3D11>(buffer);
    getContext()->IASetVertexBuffers(0, 1, bufD3D11->m_buffer.GetAddressOf(), &stride,
                                     &offset);
}

void ContextD3D11::setViewport(const NvFlowViewport *vp) {
    getContext()->RSSetViewports(1, (const D3D11_VIEWPORT *)vp);
}

Fence *ContextD3D11::shareFence(Fence *fence) {
    return nullptr;
}

Texture2D *ContextD3D11::shareTexture2D(Texture2D *sharedTexture) {
    return new Texture2DD3D11(this, sharedTexture, false);
}

Texture2DCrossAdapter *ContextD3D11::shareTexture2DCrossAdapter(
    Texture2DCrossAdapter *sharedTexture) {
    return nullptr;
}

Texture2D *ContextD3D11::shareTexture2DShared(Texture2D *sharedTexture) {
    return new Texture2DD3D11(this, sharedTexture, true);
}

void ContextD3D11::signalFence(Fence *fence, uint64_t fenceValue) {}

void ContextD3D11::timerBegin(Timer *timerIn) {
    auto timer = implCast<TimerD3D11>(timerIn);
    if (!timer->m_state) {
        auto context = getContext();
        QueryPerformanceFrequency(&timer->m_cpuFreq);
        QueryPerformanceFrequency(&timer->m_cpuBegin);
        context->Begin(timer->m_disjoint.Get());
        context->End(timer->m_begin.Get());
        timer->m_state = 1;
    }
}

void ContextD3D11::timerEnd(Timer *timerIn) {
    auto timer = implCast<TimerD3D11>(timerIn);
    if (timer->m_state == 1) {
        auto context = getContext();
        context->End(timer->m_end.Get());
        context->End(timer->m_disjoint.Get());
        QueryPerformanceCounter(&timer->m_cpuEnd);
        timer->m_state = 2;
    }
}

int ContextD3D11::timerGetResult(Timer *timerIn, float *timeGPU, float *timeCPU) {
    auto timer = implCast<TimerD3D11>(timerIn);
    if (timer->m_state != 2) return 1;

    HRESULT hr;
    auto context = getContext();
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT tsDisjoint;
    INT64 tsBegin, tsEnd;

    hr = context->GetData(timer->m_disjoint.Get(), &tsDisjoint, sizeof(tsDisjoint), 0);
    if (FAILED(hr)) return 1;

    if (tsDisjoint.Disjoint) return 1;

    hr = context->GetData(timer->m_begin.Get(), &tsBegin, sizeof(tsBegin), 0);
    if (FAILED(hr)) return 1;

    hr = context->GetData(timer->m_end.Get(), &tsEnd, sizeof(tsEnd), 0);
    if (FAILED(hr)) return 1;

    float tsDiff = float(tsEnd - tsBegin);
    constexpr float FltMax = 1.8446744e19f;  // 2^64
    if (tsEnd < tsBegin) tsDiff = tsDiff + FltMax;

    float Frequency_low = float(tsDisjoint.Frequency);
    if ((tsDisjoint.Frequency & 0x8000000000000000i64) != 0i64)
        Frequency_low = Frequency_low + FltMax;

    float msGPU = tsDiff / Frequency_low;

    float msCPU = (double)(int)(timer->m_cpuEnd.QuadPart - timer->m_cpuBegin.QuadPart) /
                  (double)(int)timer->m_cpuFreq.QuadPart;

    if (timeGPU) *timeGPU = msGPU;
    if (timeCPU) *timeCPU = msCPU;

    timer->m_state = 1;
    return 0;
}

void ContextD3D11::transitionToCommonState(Resource *resource) {}

void ContextD3D11::unmap(Buffer *buffer) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);
    if (buffer->m_desc.uploadAccess) {
        auto context = getContext();
        context->Unmap(bufD3D11->m_uploadBuffer.Get(), 0);
        context->CopyResource(bufD3D11->m_buffer.Get(), bufD3D11->m_uploadBuffer.Get());
    }
}

void ContextD3D11::unmap(Buffer *buffer, uint32_t offset, uint32_t numBytes) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);
    if (buffer->m_desc.uploadAccess) {
        auto context = getContext();
        context->Unmap(bufD3D11->m_uploadBuffer.Get(), 0);
        D3D11_BOX box;
        box.left = offset;
        box.right = offset + numBytes;
        box.top = 0;
        box.bottom = 1;
        box.front = 0;
        box.back = 1;
        if (numBytes)
            context->CopySubresourceRegion(bufD3D11->m_buffer.Get(), 0, offset, 0, 0,
                                           bufD3D11->m_uploadBuffer.Get(), 0, &box);
    }
}

void ContextD3D11::unmap(ConstantBuffer *buffer) {
    auto bufD3D11 = implCast<ConstantBufferD3D11>(buffer);
    getContext()->Unmap(bufD3D11->m_buffer.Get(), 0);
}

void ContextD3D11::unmap(IndexBuffer *buffer) {
    auto bufD3D11 = implCast<IndexBufferD3D11>(buffer);
    auto context = getContext();
    context->Unmap(bufD3D11->m_uploadBuffer.Get(), 0);
    context->CopyResource(bufD3D11->m_buffer.Get(), bufD3D11->m_uploadBuffer.Get());
}

void ContextD3D11::unmap(VertexBuffer *buffer) {
    auto bufD3D11 = implCast<VertexBufferD3D11>(buffer);
    auto context = getContext();
    context->Unmap(bufD3D11->m_uploadBuffer.Get(), 0);
    context->CopyResource(bufD3D11->m_buffer.Get(), bufD3D11->m_uploadBuffer.Get());
}

void ContextD3D11::unmap(Texture1D *buffer) {
    auto bufD3D11 = implCast<Texture1DD3D11>(buffer);
    if (bufD3D11->m_desc.uploadAccess) {
        auto context = getContext();
        context->Unmap(bufD3D11->m_uploadTexture.Get(), 0);
        context->CopyResource(bufD3D11->m_texture.Get(), bufD3D11->m_uploadTexture.Get());
    }
}

void ContextD3D11::unmap(Texture3D *texture) {
    auto texD3D11 = implCast<Texture3DD3D11>(texture);
    if (texD3D11->m_desc.uploadAccess) {
        auto context = getContext();
        context->Unmap(texD3D11->m_uploadTexture.Get(), 0);
        context->CopyResource(texD3D11->m_texture.Get(), texD3D11->m_uploadTexture.Get());
    }
}

void ContextD3D11::unmapDownload(Buffer *buffer) {
    auto bufD3D11 = implCast<BufferD3D11>(buffer);
    if (bufD3D11->m_desc.downloadAccess) {
        getContext()->Unmap(bufD3D11->m_downloadBuffer.Get(), 0);
    }
}

void ContextD3D11::unmapDownload(Texture3D *texture) {
    auto texD3D11 = implCast<Texture3DD3D11>(texture);
    if (texD3D11->m_desc.downloadAccess) {
        getContext()->Unmap(texD3D11->m_downloadTexture.Get(), 0);
    }
}

void ContextD3D11::updateVTRMapping(Texture3DVTR *textureIn, HeapVTR *heapIn,
                                    uint32_t *blockTableImage, uint32_t rowPitch,
                                    uint32_t depthPitch) {
    auto texture = implCast<Texture3DVTRD3D11>(textureIn);
    auto heap = implCast<HeapVTRD3D11>(heapIn);
    auto context = getContext();

    NvFlowDim gridDim = texture->m_gridDim;
    uint32_t maxBlocks = gridDim.x * gridDim.y * gridDim.z;

    m_tileCoords.resize(maxBlocks);
    m_tileRegionSize.resize(maxBlocks);
    m_rangeFlags.resize(maxBlocks);
    m_tilePoolCoords.resize(maxBlocks);
    m_tilePoolRangeSize.resize(maxBlocks);

    uint32_t tileID;
    uint32_t index = 0;
    auto currentImage = texture->m_blockTable;
    for (int k = 0; k < currentImage.dim().z; ++k)
        for (int j = 0; j < currentImage.dim().y; ++j)
            for (int i = 0; i < currentImage.dim().x; ++i) {
                auto newVal =
                    blockTableImage[i + (rowPitch * j + depthPitch * k) / sizeof(uint32_t)];
                auto &oldVal = currentImage(i, j, k);
                if (newVal != oldVal) {
                    tileID = ~newVal;
                    D3D11_TILED_RESOURCE_COORDINATE coords;
                    coords.X = i;
                    coords.Y = j;
                    coords.Z = k;
                    coords.Subresource = 0;
                    m_tileCoords[index] = coords;

                    D3D11_TILE_REGION_SIZE regionSize;
                    regionSize.NumTiles = 1;
                    regionSize.bUseBox = FALSE;
                    regionSize.Width = 1;
                    regionSize.Height = 65537;
                    m_tileRegionSize[index] = regionSize;

                    m_rangeFlags[index] =
                        newVal ? D3D11_TILE_RANGE_REUSE_SINGLE_TILE : D3D11_TILE_RANGE_NULL;
                    m_tilePoolCoords[index] = tileID;
                    m_tilePoolRangeSize[index] = 1;
                    oldVal = newVal;
                    ++index;
                }
            }

    if (index) {
        ComPtr<ID3D11DeviceContext2> deviceContext2;
        if (SUCCEEDED(m_deviceContext->QueryInterface(IID_PPV_ARGS(&deviceContext2)))) {
            deviceContext2->UpdateTileMappings(
                texture->m_texture.Get(), index, m_tileCoords.data(),
                m_tileRegionSize.data(), heap->m_heap.Get(), index,
                (const UINT *)m_rangeFlags.data(), m_tilePoolCoords.data(),
                m_tilePoolRangeSize.data(), D3D11_TILE_MAPPING_NO_OVERWRITE);
        }
    }
}

void ContextD3D11::waitOnFence(Fence *fence, uint64_t fenceValue) {}

DepthStencilView *ContextD3D11::createDepthStencilView(
    const NvFlowDepthStencilViewDescD3D11 *desc) {
    return new DepthStencilViewD3D11(this, desc);
}

RenderTargetView *ContextD3D11::createRenderTargetView(
    const NvFlowRenderTargetViewDescD3D11 *desc) {
    return new RenderTargetViewD3D11(this, desc);
}

void ContextD3D11::setRenderTarget(RenderTarget *rtvIn, DepthStencil *dsvIn) {
    auto rtv = implCast<RenderTargetD3D11>(rtvIn);
    auto dsv = implCast<DepthStencilD3D11>(dsvIn);
    ID3D11RenderTargetView *prtv = rtv ? rtv->m_rtv.Get() : nullptr;
    ID3D11DepthStencilView *pdsv = dsv ? dsv->m_dsv.Get() : nullptr;
    auto context = getContext();

    context->OMSetRenderTargets(1, &prtv, pdsv);

    if (rtv) {
        D3D11_VIEWPORT vp;
        memcpy(&vp, &rtv->m_viewport, sizeof(vp));
        context->RSSetViewports(1, &vp);
    } else if (dsv) {
        D3D11_VIEWPORT vp;
        memcpy(&vp, &dsv->m_viewport, sizeof(vp));
        context->RSSetViewports(1, &vp);
    }
}

ResourceReference *ContextD3D11::shareResourceReference(ResourceD3D11 *resource) {
    return new ResourceReferenceD3D11(this, resource);
}

void ContextD3D11::updateDepthStencilView(NvFlowDepthStencilView *view,
                                          const NvFlowDepthStencilViewDescD3D11 *desc) {
    auto viewD3D11 = implCast<DepthStencilViewD3D11>(view);
    viewD3D11->update(desc);
}

void ContextD3D11::updateRenderTargetView(NvFlowRenderTargetView *view,
                                          const NvFlowRenderTargetViewDescD3D11 *desc) {
    auto viewD3D11 = implCast<RenderTargetViewD3D11>(view);
    viewD3D11->update(desc);
}

void ContextD3D11::updateResourceRWViewDesc(ResourceRWD3D11 *resourceRWIn,
                                            NvFlowResourceRWViewDescD3D11 *desc) {
    desc->resourceView.srv = resourceRWIn->m_srv.Get();
    desc->uav = resourceRWIn->m_uav.Get();
}

void ContextD3D11::updateResourceViewDesc(ResourceD3D11 *resourceIn,
                                          NvFlowResourceViewDescD3D11 *desc) {
    desc->srv = resourceIn->m_srv.Get();
}

void ContextD3D11::updateContext(const NvFlowContextDescD3D11 *pdesc) {
    m_device = pdesc->device;
    m_deviceContext = pdesc->deviceContext;
    m_deviceContext->QueryInterface(IID_PPV_ARGS(&m_d3dAnnotation));
}

void ContextD3D11::updateContextDesc(NvFlowContextDescD3D11 *desc) {
    desc->device = m_device.Get();
    desc->deviceContext = m_deviceContext.Get();
}

BufferD3D11::BufferD3D11(ContextD3D11 *ctx, BufferD3D11 *buffer,
                         const NvFlowBufferViewDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = buffer->m_desc;
    uint32_t srcFormatSize = getFormatSizeInBytes(m_desc.format);
    m_desc.format = desc->format;
    uint32_t newFormatSize = getFormatSizeInBytes(m_desc.format);
    m_desc.dim = m_desc.dim * srcFormatSize / newFormatSize;
    m_buffer = buffer->m_buffer;

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = convertToDXGI(m_desc.format);
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = m_desc.dim;
    ctx->getDevice()->CreateShaderResourceView(m_buffer.Get(), &srvDesc, &m_srv);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = srvDesc.Format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = m_desc.dim;
    ctx->getDevice()->CreateUnorderedAccessView(m_buffer.Get(), &uavDesc, &m_uav);
}

BufferD3D11::BufferD3D11(ContextD3D11 *ctx, const NvFlowBufferDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;

    D3D11_BUFFER_DESC bufDesc = {};
    bufDesc.ByteWidth = getFormatSizeInBytes(m_desc.format) * m_desc.dim;
    bufDesc.Usage = D3D11_USAGE_DEFAULT;
    bufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bufDesc.CPUAccessFlags = 0;
    bufDesc.MiscFlags = 0;
    auto device = ctx->getDevice();
    device->CreateBuffer(&bufDesc, nullptr, &m_buffer);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = convertToDXGI(m_desc.format);
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = m_desc.dim;
    device->CreateShaderResourceView(m_buffer.Get(), &srvDesc, &m_srv);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
    uavDesc.Format = convertToDXGI(m_desc.format);
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = m_desc.dim;
    device->CreateUnorderedAccessView(m_buffer.Get(), &uavDesc, &m_uav);

    if (m_desc.uploadAccess) {
        bufDesc.Usage = D3D11_USAGE_DYNAMIC;
        bufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device->CreateBuffer(&bufDesc, nullptr, &m_uploadBuffer);
    }

    if (m_desc.downloadAccess) {
        bufDesc.Usage = D3D11_USAGE_STAGING;
        bufDesc.BindFlags = 0;
        bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device->CreateBuffer(&bufDesc, nullptr, &m_downloadBuffer);
    }
}

uint64_t BufferD3D11::getGPUBytesUsed() {
    return getFormatSizeInBytes(m_desc.format) * m_desc.dim;
}

ConstantBufferD3D11::ConstantBufferD3D11(ContextD3D11 *ctx,
                                         const NvFlowConstantBufferDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    UINT byteWidth = alignUp<16>(desc->sizeInBytes);

    D3D11_BUFFER_DESC bufDesc = {};
    bufDesc.ByteWidth = byteWidth;
    if (desc->uploadAccess) {
        bufDesc.Usage = D3D11_USAGE_DYNAMIC;
        bufDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    } else {
        bufDesc.Usage = D3D11_USAGE_DEFAULT;
        bufDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bufDesc.CPUAccessFlags = 0;
    }
    ctx->getDevice()->CreateBuffer(&bufDesc, nullptr, &m_buffer);
}

uint64_t ConstantBufferD3D11::getGPUBytesUsed() {
    return alignUp<16>(m_desc.sizeInBytes);
}

VertexBufferD3D11::VertexBufferD3D11(ContextD3D11 *ctx, const NvFlowVertexBufferDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;

    D3D11_BUFFER_DESC bufDesc = {};
    bufDesc.ByteWidth = desc->sizeInBytes;
    bufDesc.Usage = D3D11_USAGE_DEFAULT;
    bufDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufDesc.CPUAccessFlags = 0;
    bufDesc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = desc->data;
    initData.SysMemPitch = desc->sizeInBytes;
    ctx->getDevice()->CreateBuffer(&bufDesc, &initData, &m_buffer);

    bufDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bufDesc.MiscFlags = 0;
    ctx->getDevice()->CreateBuffer(&bufDesc, nullptr, &m_uploadBuffer);
}

uint64_t VertexBufferD3D11::getGPUBytesUsed() {
    return m_desc.sizeInBytes;
}

IndexBufferD3D11::IndexBufferD3D11(ContextD3D11 *ctx, const NvFlowIndexBufferDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    m_format = convertToDXGI(m_desc.format);

    D3D11_BUFFER_DESC bufDesc = {};
    bufDesc.ByteWidth = desc->sizeInBytes;
    bufDesc.Usage = D3D11_USAGE_DEFAULT;
    bufDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bufDesc.CPUAccessFlags = 0;
    bufDesc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = desc->data;
    initData.SysMemPitch = desc->sizeInBytes;
    ctx->getDevice()->CreateBuffer(&bufDesc, &initData, &m_buffer);

    bufDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bufDesc.MiscFlags = 0;
    ctx->getDevice()->CreateBuffer(&bufDesc, nullptr, &m_uploadBuffer);
}

uint64_t IndexBufferD3D11::getGPUBytesUsed() {
    return m_desc.sizeInBytes;
}

Texture1DD3D11::Texture1DD3D11(ContextD3D11 *ctx, const NvFlowTexture1DDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;

    D3D11_TEXTURE1D_DESC texDesc = {};
    texDesc.Format = convertToDXGI(m_desc.format);
    texDesc.Width = m_desc.dim;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;
    auto device = ctx->getDevice();
    device->CreateTexture1D(&texDesc, nullptr, &m_texture);

    device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_srv);
    device->CreateUnorderedAccessView(m_texture.Get(), nullptr, &m_uav);

    if (m_desc.uploadAccess) {
        texDesc.Usage = D3D11_USAGE_DYNAMIC;
        texDesc.BindFlags = 0;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device->CreateTexture1D(&texDesc, nullptr, &m_uploadTexture);
    }
}

uint64_t Texture1DD3D11::getGPUBytesUsed() {
    return getFormatSizeInBytes(m_desc.format) * m_desc.dim;
}

Texture2DD3D11::Texture2DD3D11(ContextD3D11 *ctx, const NvFlowTexture2DDesc *desc,
                               bool createShared)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;

    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = m_desc.width;
    texDesc.Height = m_desc.height;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = convertToDXGI(m_desc.format);
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = createShared ? D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX : 0;
    auto device = ctx->getDevice();
    device->CreateTexture2D(&texDesc, nullptr, &m_texture);

    device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_srv);
    device->CreateUnorderedAccessView(m_texture.Get(), nullptr, &m_uav);

    if (createShared) {
        ComPtr<IDXGIKeyedMutex> keyedMutex;
        device->QueryInterface(IID_PPV_ARGS(&keyedMutex));
        keyedMutex->AcquireSync(0, INFINITE);
    }
}

Texture2DD3D11::Texture2DD3D11(ContextD3D11 *ctx, Texture2D *sharedTexture, bool openShared)
    : Object(ctx->getDeferredRelease()) {
    m_desc = sharedTexture->m_desc;

    auto device = ctx->getDevice();

    if (openShared) {
        HANDLE sharedHandle;
        sharedTexture->openSharedHandle(&sharedHandle);
        device->OpenSharedResource(sharedHandle, IID_PPV_ARGS(&m_texture));
        sharedTexture->closeSharedHandle(sharedHandle);
    } else {
        if (sharedTexture) {
            auto texD3D11 = static_cast<Texture2DD3D11 *>(sharedTexture);
            m_texture = texD3D11->m_texture;
        }
    }

    if (m_texture) {
        device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_srv);
        device->CreateUnorderedAccessView(m_texture.Get(), nullptr, &m_uav);
    }
}

void Texture2DD3D11::openSharedHandle(HANDLE *handleIn) {
    ComPtr<IDXGIResource1> resource;
    m_texture->QueryInterface(IID_PPV_ARGS(&resource));
    resource->GetSharedHandle(handleIn);
}

void Texture2DD3D11::closeSharedHandle(HANDLE handle) {}

uint64_t Texture2DD3D11::getGPUBytesUsed() {
    return getFormatSizeInBytes(m_desc.format) * m_desc.width * m_desc.height;
}

Texture3DD3D11::Texture3DD3D11(ContextD3D11 *ctx, const NvFlowTexture3DDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    auto device = ctx->getDevice();

    D3D11_TEXTURE3D_DESC texDesc = {};
    texDesc.Width = m_desc.dim.x;
    texDesc.Height = m_desc.dim.y;
    texDesc.Depth = m_desc.dim.z;
    texDesc.MipLevels = 1;
    texDesc.Format = convertToDXGI(m_desc.format);
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;

    device->CreateTexture3D(&texDesc, nullptr, &m_texture);

    device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_srv);
    device->CreateUnorderedAccessView(m_texture.Get(), nullptr, &m_uav);

    if (m_desc.uploadAccess) {
        texDesc.Usage = D3D11_USAGE_DYNAMIC;
        texDesc.BindFlags = 0;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device->CreateTexture3D(&texDesc, nullptr, &m_uploadTexture);
    }

    if (m_desc.downloadAccess) {
        texDesc.Usage = D3D11_USAGE_STAGING;
        texDesc.BindFlags = 0;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device->CreateTexture3D(&texDesc, nullptr, &m_downloadTexture);
    }
}

uint64_t Texture3DD3D11::getGPUBytesUsed() {
    uint64_t sz = getFormatSizeInBytes(m_desc.format);
    return sz * m_desc.dim.x * m_desc.dim.y * m_desc.dim.z;
}

ResourceReferenceD3D11::ResourceReferenceD3D11(ContextD3D11 *ctx, ResourceD3D11 *resource)
    : Object(ctx->getDeferredRelease()) {
    resource->m_srv->GetResource(&m_resource);
}

uint64_t ResourceReferenceD3D11::getGPUBytesUsed() {
    return 0;
}

HeapVTRD3D11::HeapVTRD3D11(ContextD3D11 *ctx, const NvFlowHeapSparseDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;

    constexpr uint32_t tileSize = 0x10000;  // 64K
    m_numTiles = (m_desc.sizeInBytes + tileSize - 1) / tileSize;

    D3D11_BUFFER_DESC bufDesc = {};
    bufDesc.ByteWidth = m_numTiles * tileSize;
    bufDesc.Usage = D3D11_USAGE_DEFAULT;
    bufDesc.BindFlags = 0;
    bufDesc.CPUAccessFlags = 0;
    bufDesc.MiscFlags = D3D11_RESOURCE_MISC_TILE_POOL;
    bufDesc.StructureByteStride = 0;
    ctx->getDevice()->CreateBuffer(&bufDesc, nullptr, &m_heap);
}

uint64_t HeapVTRD3D11::getGPUBytesUsed() {
    return m_desc.sizeInBytes;
}

Texture3DVTRD3D11::Texture3DVTRD3D11(ContextD3D11 *ctx,
                                     const NvFlowTexture3DSparseDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    auto device = ctx->getDevice();

    D3D11_TEXTURE3D_DESC texDesc = {};
    texDesc.Width = m_desc.dim.x;
    texDesc.Height = m_desc.dim.y;
    texDesc.Height = m_desc.dim.z;
    texDesc.MipLevels = 1;
    texDesc.Format = convertToDXGI(m_desc.format);
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = D3D11_RESOURCE_MISC_TILED;

    device->CreateTexture3D(&texDesc, nullptr, &m_texture);
    device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_srv);
    device->CreateUnorderedAccessView(m_texture.Get(), nullptr, &m_uav);

    m_blockDim = getTileDim(m_desc.format);
    m_gridDim = m_desc.dim / m_blockDim;
    m_blockTable.resize(m_gridDim);
}

uint64_t Texture3DVTRD3D11::getGPUBytesUsed() {
    return 0;
}

ColorBufferD3D11::ColorBufferD3D11(ContextD3D11 *ctx, const NvFlowColorBufferDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    auto device = ctx->getDevice();

    m_rt_format = desc->format;
    m_viewport.topLeftX = 0.f;
    m_viewport.topLeftY = 0.f;
    m_viewport.width = (float)m_desc.width;
    m_viewport.height = (float)desc->height;
    m_viewport.minDepth = 0.f;
    m_viewport.maxDepth = FLOAT_1_0;

    m_desc = *desc;

    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = m_desc.width;
    texDesc.Height = m_desc.height;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = convertToDXGI(m_desc.format);
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags =
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;
    device->CreateTexture2D(&texDesc, nullptr, &m_texture);
    device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_srv);
    device->CreateUnorderedAccessView(m_texture.Get(), nullptr, &m_uav);
    device->CreateRenderTargetView(m_texture.Get(), nullptr, &m_rtv);
}

uint64_t ColorBufferD3D11::getGPUBytesUsed() {
    uint32_t sz = getFormatSizeInBytes(m_desc.format);
    sz *= m_desc.width * m_desc.height;
    return sz;
}

DepthBufferD3D11::DepthBufferD3D11(ContextD3D11 *ctx, const NvFlowDepthBufferDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    m_ds_format = m_desc.format_dsv;
    m_viewport.topLeftX = 0.f;
    m_viewport.topLeftY = 0.f;
    m_viewport.width = (float)m_desc.width;
    m_viewport.height = (float)m_desc.height;
    m_viewport.minDepth = 0.f;
    m_viewport.maxDepth = FLOAT_1_0;

    auto device = ctx->getDevice();

    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = m_desc.width;
    texDesc.Height = m_desc.height;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = convertToDXGI(m_desc.format_resource);
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;
    device->CreateTexture2D(&texDesc, nullptr, &m_texture);

    D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format = convertToDXGI(m_desc.format_dsv);
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Flags = 0;
    dsvDesc.Texture2D.MipSlice = 1;
    device->CreateDepthStencilView(m_texture.Get(), &dsvDesc, &m_dsv);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = convertToDXGI(m_desc.format_srv);
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(m_texture.Get(), &srvDesc, &m_srv);
}

uint64_t DepthBufferD3D11::getGPUBytesUsed() {
    uint64_t sz = getFormatSizeInBytes(m_desc.format_resource);
    sz *= m_desc.width * m_desc.height;
    return sz;
}

DepthStencilViewD3D11::DepthStencilViewD3D11(ContextD3D11 *ctx,
                                             const NvFlowDepthStencilViewDescD3D11 *desc)
    : Object(ctx->getDeferredRelease()) {
    update(desc);
}

uint64_t DepthStencilViewD3D11::getGPUBytesUsed() {
    return Object::getGPUBytesUsed();
}

void DepthStencilViewD3D11::update(const NvFlowDepthStencilViewDescD3D11 *desc) {
    m_desc = *desc;

    D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    m_desc.dsv->GetDesc(&dsvDesc);
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    m_desc.srv->GetDesc(&srvDesc);

    NvFlowDim resDim = ContextD3D11::extractDim(m_desc.srv);
    m_ds_format = convertToNvFlow(dsvDesc.Format);
    m_srv = m_desc.srv;
    m_dsv = m_desc.dsv;
    m_width = resDim.x;
    m_height = resDim.y;
    copyViewport(&m_viewport, &m_desc.viewport);
}

NvFlowDepthBufferDesc DepthStencilViewD3D11::getDepthBufferDesc() {
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    m_srv->GetDesc(&srvDesc);

    NvFlowFormat resourceFormat = eNvFlowFormat_unknown;
    {
        ComPtr<ID3D11Resource> resource;
        ComPtr<ID3D11Texture2D> texture;
        m_srv->GetResource(&resource);
        resource->QueryInterface(IID_PPV_ARGS(&texture));
        if (texture) {
            D3D11_TEXTURE2D_DESC texDesc;
            texture->GetDesc(&texDesc);
            resourceFormat = convertToNvFlow(texDesc.Format);
        }
    }

    NvFlowDepthBufferDesc bufDesc = {};
    bufDesc.format_resource = resourceFormat;
    bufDesc.format_dsv = m_ds_format;
    bufDesc.format_srv = convertToNvFlow(srvDesc.Format);
    bufDesc.width = m_viewport.width;
    bufDesc.height = m_viewport.height;
    return bufDesc;
}

void copyViewport(NvFlowViewport *dst, const D3D11_VIEWPORT *src) {
    *(D3D11_VIEWPORT *)dst = *src;
}

D3D11_BLEND convertToD3D11(NvFlowBlendEnum blendEnum) {
    return g_blendEnumTableD3D11.table[blendEnum];
}

D3D11_BLEND_OP convertToD3D11(NvFlowBlendOpEnum blendOpEnum) {
    return g_blendOpEnumTableD3D11.table[blendOpEnum];
}

D3D11_COMPARISON_FUNC convertToD3D11(NvFlowComparisonEnum comparison) {
    return g_comparisonTableD3D11.table[comparison];
}

int64_t FlowDeferredReleaseD3D11(float timeoutMS) {
    return 0;
}

RenderTargetViewD3D11::RenderTargetViewD3D11(ContextD3D11 *ctx,
                                             const NvFlowRenderTargetViewDescD3D11 *desc)
    : Object(ctx->getDeferredRelease()) {
    update(desc);
}

uint64_t RenderTargetViewD3D11::getGPUBytesUsed() {
    return 0;
}

void RenderTargetViewD3D11::update(const NvFlowRenderTargetViewDescD3D11 *desc) {
    m_desc = *desc;

    D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
    m_rtv->GetDesc(&rtvDesc);
    m_rt_format = convertToNvFlow(rtvDesc.Format);
    m_rtv = m_desc.rtv;
    copyViewport(&m_viewport, &m_desc.viewport);
}

ComputeShaderD3D11::ComputeShaderD3D11(ContextD3D11 *ctx,
                                       const NvFlowComputeShaderDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    if (!m_desc.label) m_desc.label = L"unlabeled";
    ctx->getDevice()->CreateComputeShader(m_desc.cs, m_desc.cs_length, nullptr, &m_cs);
}

uint64_t ComputeShaderD3D11::getGPUBytesUsed() {
    return Object::getGPUBytesUsed();
}

GraphicsShaderD3D11::GraphicsShaderD3D11(ContextD3D11 *ctx,
                                         const NvFlowGraphicsShaderDesc *desc)
    : Object(ctx->getDeferredRelease()) {
    m_desc = *desc;
    m_inputElementDescs.resize(m_desc.numInputElements);

    auto device = ctx->getDevice();

    for (int i = 0; i < m_desc.numInputElements; ++i) {
        m_inputElementDescs[i] = m_desc.inputElementDescs[i];
    }
    m_desc.inputElementDescs = m_inputElementDescs.data();

    if (!m_desc.label) m_desc.label = L"unlabeled";

    device->CreateVertexShader(m_desc.vs, m_desc.vs_length, nullptr, &m_vs);
    device->CreatePixelShader(m_desc.ps, m_desc.ps_length, nullptr, &m_ps);

    VectorCached<D3D11_INPUT_ELEMENT_DESC, 4> elementDesc;
    UINT alignedByteOffset = 0;

    elementDesc.resize(m_desc.numInputElements);
    for (int i = 0; i < m_desc.numInputElements; ++i) {
        auto &input = m_desc.inputElementDescs[i];
        auto &e = elementDesc[i];
        e.SemanticName = input.semanticName;
        e.SemanticIndex = 0;
        e.Format = convertToDXGI(input.format);
        e.InputSlot = 0;
        e.AlignedByteOffset = alignedByteOffset;
        e.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
        e.InstanceDataStepRate = 0;
        alignedByteOffset = -1;
    }

    device->CreateInputLayout(elementDesc.data(), m_desc.numInputElements, m_desc.vs,
                              m_desc.vs_length, &m_inputLayout);

    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = m_desc.blendState.enable;
    blendDesc.RenderTarget[0].SrcBlend = convertToD3D11(m_desc.blendState.srcBlendColor);
    blendDesc.RenderTarget[0].DestBlend = convertToD3D11(m_desc.blendState.dstBlendColor);
    blendDesc.RenderTarget[0].BlendOp = convertToD3D11(m_desc.blendState.blendOpColor);
    blendDesc.RenderTarget[0].SrcBlendAlpha =
        convertToD3D11(m_desc.blendState.srcBlendAlpha);
    blendDesc.RenderTarget[0].DestBlendAlpha =
        convertToD3D11(m_desc.blendState.dstBlendAlpha);
    blendDesc.RenderTarget[0].BlendOpAlpha = convertToD3D11(m_desc.blendState.blendOpAlpha);
    device->CreateBlendState(&blendDesc, &m_blenderState);

    D3D11_DEPTH_STENCIL_DESC depthDesc = {};
    depthDesc.DepthEnable = m_desc.depthState.depthEnable;
    depthDesc.DepthWriteMask = m_desc.depthState.depthWriteMask == eNvFlowDepthWriteMask_All
                                   ? D3D11_DEPTH_WRITE_MASK_ALL
                                   : D3D11_DEPTH_WRITE_MASK_ZERO;
    depthDesc.DepthFunc = convertToD3D11(m_desc.depthState.depthFunc);
    depthDesc.StencilEnable = FALSE;
    device->CreateDepthStencilState(&depthDesc, &m_depthStencilState);

    D3D11_RASTERIZER_DESC rastDesc = {};
    rastDesc.FillMode = D3D11_FILL_SOLID;
    rastDesc.CullMode = m_desc.uavTarget ? D3D11_CULL_NONE : D3D11_CULL_BACK;
    rastDesc.FrontCounterClockwise = FALSE;
    rastDesc.DepthBias = 0;
    rastDesc.DepthBiasClamp = 0.f;
    rastDesc.SlopeScaledDepthBias = 0.f;
    rastDesc.DepthClipEnable = m_desc.depthClipEnable;
    rastDesc.ScissorEnable = FALSE;
    rastDesc.MultisampleEnable = FALSE;
    rastDesc.AntialiasedLineEnable = FALSE;
    device->CreateRasterizerState(&rastDesc, &m_rasterizerStateLH);
    rastDesc.FrontCounterClockwise = TRUE;
    device->CreateRasterizerState(&rastDesc, &m_rasterizerStateRH);
}

uint64_t GraphicsShaderD3D11::getGPUBytesUsed() {
    return 0;
}

void GraphicsShaderD3D11::setFormats(NvFlowFormat rtFormat, NvFlowFormat dsFormat) {
    m_desc.renderTargetFormat[0] = rtFormat;
    m_desc.depthStencilFormat = dsFormat;
}

TimerD3D11::TimerD3D11(ContextD3D11 *ctx) : Object(ctx->getDeferredRelease()) {
    auto device = ctx->getDevice();

    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    queryDesc.MiscFlags = 0;
    device->CreateQuery(&queryDesc, &m_disjoint);

    queryDesc.Query = D3D11_QUERY_TIMESTAMP;
    device->CreateQuery(&queryDesc, &m_begin);
    device->CreateQuery(&queryDesc, &m_end);
}

uint64_t TimerD3D11::getGPUBytesUsed() {
    return 0;
}

EventQueueD3D11::EventQueueD3D11(ContextD3D11 *ctx)
    : Object(ctx->getDeferredRelease()), m_pushID(0) {}

uint64_t EventQueueD3D11::getGPUBytesUsed() {
    return 0;
}

void EventQueueD3D11::push(uint64_t uid, ContextD3D11 *ctx) {
    auto e = getNewEvent();
    e->state = eEventStateActive;
    e->uid = uid;
    e->pushID = ++m_pushID;

    if (!e->query) {
        D3D11_QUERY_DESC queryDesc = {};
        queryDesc.Query = D3D11_QUERY_EVENT;
        queryDesc.MiscFlags = 0;
        ctx->getDevice()->CreateQuery(&queryDesc, &e->query);
    }
    ctx->getContext()->End(e->query.Get());
}

int EventQueueD3D11::pop(uint64_t *pUid, ContextD3D11 *ctx) {
    HRESULT hr;
    int compute;
    bool valid = false;
    uint32_t minIdx = 0;
    uint64_t minPushID = 0;

    for (uint32_t i = 0; i < (uint32_t)m_events.size(); ++i) {
        auto &e = m_events[i];
        if (e.state == eEventStateActive) {
            if (valid) {
                if (e.pushID < minPushID) {
                    minIdx = i;
                    minPushID = e.pushID;
                }
            } else {
                minIdx = i;
                minPushID = e.pushID;
                valid = true;
            }
        }
    }

    if (!valid) return 1;

    auto &minEvent = m_events[minIdx];
    hr = ctx->getContext()->GetData(minEvent.query.Get(), &compute, sizeof(compute), 0);
    if (FAILED(hr) || compute != 1) return 1;

    if (pUid) *pUid = minEvent.uid;
    minEvent.state = eEventStateInactive;
    return 0;
}

EventQueueD3D11::Event *EventQueueD3D11::getNewEvent() {
    for (auto &ev : m_events) {
        if (ev.state == eEventStateInactive) return &ev;
    }
    Event newEvent;
    newEvent.uid = 0;
    newEvent.pushID = 0;
    newEvent.state = eEventStateInactive;
    m_events.push_back(newEvent);
    return &m_events.back();
}

void ContextD3D11::State::reset() {
    SafeRelease(indexBuffer);
    SafeRelease(layout);
    SafeRelease(vertexBuffers);

    SafeRelease(blendState);
    SafeRelease(depthStencilState);

    SafeRelease(renderTargetViews);

    SafeRelease(depthStencilView);

    SafeRelease(rasterizerState);

    SafeRelease(vsConstantBuffers);
    SafeRelease(vsSamplers);
    SafeRelease(vsSrvs);

    SafeRelease(psConstantBuffers);
    SafeRelease(psSamplers);
    SafeRelease(psSrvs);

    SafeRelease(csConstantBuffers);
    SafeRelease(csSamplers);
    SafeRelease(csSrvs);
    SafeRelease(csUavs);

    SafeRelease(computeShader);
    SafeRelease(pixelShader);
    SafeRelease(geometryShader);
    SafeRelease(hullShader);
    SafeRelease(domainShader);
}

ContextD3D11::State::~State() {
    reset();
}

}  // namespace NvFlow