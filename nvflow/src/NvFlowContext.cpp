#include "NvFlowContextImpl.h"
#include <nvflow/NvFlowContextExt.h>
#include "Types.h"
#include "Context.h"
#include "ClientHelper.h"

NV_FLOW_API NvFlowContextAPI NvFlowContextGetContextType(NvFlowContext* context) {
    return context->getContextType();
}

NV_FLOW_API void NvFlowContextFlushRequestPush(NvFlowContext* context) {
    return context->flushRequestPush();
}

NV_FLOW_API bool NvFlowContextFlushRequestPop(NvFlowContext* context) {
    return context->flushRequestPop();
}

NV_FLOW_API void NvFlowContextProcessFenceWait(NvFlowContext* context) {
    return context->processFenceWait(context);
}

NV_FLOW_API void NvFlowContextProcessFenceSignal(NvFlowContext* context) {
    return context->processFenceSignal(context);
}

NV_FLOW_API void NvFlowReleaseContext(NvFlowContext* context) {
    context->release();
}

NV_FLOW_API void NvFlowReleaseDepthStencilView(NvFlowDepthStencilView* view) {
    view->release();
}

NV_FLOW_API void NvFlowReleaseRenderTargetView(NvFlowRenderTargetView* view) {
    view->release();
}

NV_FLOW_API void NvFlowContextPush(NvFlowContext* context) {
    return context->contextPush();
}

NV_FLOW_API void NvFlowContextPop(NvFlowContext* context) {
    return context->contextPop();
}

NV_FLOW_API void NvFlowSetMallocFunc(void* (*malloc)(size_t size)) {
    return NvFlow::FlowSetMallocFunc(malloc);
}

NV_FLOW_API void NvFlowSetFreeFunc(void (*free)(void* ptr)) {
    return NvFlow::FlowSetFreeFunc(free);
}

NV_FLOW_API NvFlowUint NvFlowDeferredRelease(float timeoutMS) {
    return NvFlow::FlowDeferredRelease(timeoutMS);
}

NV_FLOW_API NvFlowUint NvFlowContextObjectAddRef(NvFlowContextObject* object) {
    return object->addRef();
}

NV_FLOW_API NvFlowUint NvFlowContextObjectRelease(NvFlowContextObject* object) {
    return object->release();
}

NV_FLOW_API NvFlowUint64 NvFlowContextObjectGetGPUBytesUsed(NvFlowContextObject* object) {
    return object->getGPUBytesUsed();
}

NV_FLOW_API void NvFlowConstantBufferGetDesc(NvFlowConstantBuffer* buffer,
                                             NvFlowConstantBufferDesc* desc) {
    if (buffer) *desc = NvFlow::implCast<NvFlow::ConstantBuffer>(buffer)->m_desc;
}

NV_FLOW_API NvFlowConstantBuffer* NvFlowCreateConstantBuffer(
    NvFlowContext* context, const NvFlowConstantBufferDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createConstantBuffer(desc);
}

NV_FLOW_API void NvFlowReleaseConstantBuffer(NvFlowConstantBuffer* buffer) {
    if (buffer) buffer->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowConstantBufferGetContextObject(
    NvFlowConstantBuffer* buffer) {
    return buffer;
}

NV_FLOW_API void* NvFlowConstantBufferMap(NvFlowContext* context,
                                          NvFlowConstantBuffer* constantBuffer) {
    if (constantBuffer && context)
        return NvFlow::implCast<NvFlow::Context>(context)->map(
            NvFlow::implCast<NvFlow::ConstantBuffer>(constantBuffer));
    return nullptr;
}

NV_FLOW_API void NvFlowConstantBufferUnmap(NvFlowContext* context,
                                           NvFlowConstantBuffer* constantBuffer) {
    if (context && constantBuffer)
        NvFlow::implCast<NvFlow::Context>(context)->unmap(
            NvFlow::implCast<NvFlow::ConstantBuffer>(constantBuffer));
}

NV_FLOW_API void NvFlowVertexBufferGetDesc(NvFlowVertexBuffer* buffer,
                                           NvFlowVertexBufferDesc* desc) {
    if (buffer) *desc = NvFlow::implCast<NvFlow::VertexBuffer>(buffer)->m_desc;
}

NV_FLOW_API NvFlowVertexBuffer* NvFlowCreateVertexBuffer(
    NvFlowContext* context, const NvFlowVertexBufferDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createVertexBuffer(desc);
}

NV_FLOW_API void NvFlowReleaseVertexBuffer(NvFlowVertexBuffer* vertexBuffer) {
    if (vertexBuffer) vertexBuffer->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowVertexBufferGetContextObject(
    NvFlowVertexBuffer* buffer) {
    return buffer;
}

NV_FLOW_API void* NvFlowVertexBufferMap(NvFlowContext* context,
                                        NvFlowVertexBuffer* vertexBuffer) {
    if (vertexBuffer)
        return NvFlow::implCast<NvFlow::Context>(context)->map(
            NvFlow::implCast<NvFlow::VertexBuffer>(vertexBuffer));
    return nullptr;
}

NV_FLOW_API void NvFlowVertexBufferUnmap(NvFlowContext* context,
                                         NvFlowVertexBuffer* vertexBuffer) {
    if (vertexBuffer)
        NvFlow::implCast<NvFlow::Context>(context)->unmap(
            NvFlow::implCast<NvFlow::VertexBuffer>(vertexBuffer));
}

NV_FLOW_API void NvFlowIndexBufferGetDesc(NvFlowIndexBuffer* index,
                                          NvFlowIndexBufferDesc* desc) {
    if (index) *desc = NvFlow::implCast<NvFlow::IndexBuffer>(index)->m_desc;
}

NV_FLOW_API NvFlowIndexBuffer* NvFlowCreateIndexBuffer(NvFlowContext* context,
                                                       const NvFlowIndexBufferDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createIndexBuffer(desc);
}

NV_FLOW_API void NvFlowReleaseIndexBuffer(NvFlowIndexBuffer* buffer) {
    if (buffer) buffer->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowIndexBufferGetContextObject(
    NvFlowIndexBuffer* buffer) {
    return buffer;
}

NV_FLOW_API void* NvFlowIndexBufferMap(NvFlowContext* context,
                                       NvFlowIndexBuffer* indexBuffer) {
    if (indexBuffer)
        return NvFlow::implCast<NvFlow::Context>(context)->map(
            NvFlow::implCast<NvFlow::IndexBuffer>(indexBuffer));
    return nullptr;
}

NV_FLOW_API void NvFlowIndexBufferUnmap(NvFlowContext* context,
                                        NvFlowIndexBuffer* indexBuffer) {
    if (indexBuffer)
        return NvFlow::implCast<NvFlow::Context>(context)->unmap(
            NvFlow::implCast<NvFlow::IndexBuffer>(indexBuffer));
}

NV_FLOW_API NvFlowContextObject* NvFlowResourceGetContextObject(NvFlowResource* resource) {
    return resource;
}

NV_FLOW_API NvFlowContextObject* NvFlowResourceRWGetContextObject(
    NvFlowResourceRW* resourceRW) {
    return resourceRW;
}

NV_FLOW_API NvFlowResource* NvFlowResourceRWGetResource(NvFlowResourceRW* resourceRW) {
    if (resourceRW) return NvFlow::implCast<NvFlow::ResourceRW>(resourceRW)->getResource();
    return nullptr;
}

NV_FLOW_API void NvFlowRenderTargetGetDesc(NvFlowRenderTarget* rt,
                                           NvFlowRenderTargetDesc* desc) {
    if (rt) {
        auto rt2 = NvFlow::implCast<NvFlow::RenderTarget>(rt);
        desc->rt_format = rt2->m_rt_format;
        desc->viewport = rt2->m_viewport;
    }
}

NV_FLOW_API void NvFlowRenderTargetSetViewport(NvFlowRenderTarget* rt,
                                               const NvFlowViewport* viewport) {
    if (rt) NvFlow::implCast<NvFlow::RenderTarget>(rt)->m_viewport = *viewport;
}

NV_FLOW_API void NvFlowDepthStencilGetDesc(NvFlowDepthStencil* ds,
                                           NvFlowDepthStencilDesc* desc) {
    if (ds) {
        auto ds2 = NvFlow::implCast<NvFlow::DepthStencil>(ds);
        desc->ds_format = ds2->m_ds_format;
        desc->viewport = ds2->m_viewport;
        desc->width = ds2->m_width;
        desc->height = ds2->m_height;
    }
}

NV_FLOW_API void NvFlowDepthStencilSetViewport(NvFlowDepthStencil* ds,
                                               const NvFlowViewport* viewport) {
    if (ds) {
        auto ds2 = NvFlow::implCast<NvFlow::DepthStencil>(ds);
        ds2->m_viewport = *viewport;
    }
}

NV_FLOW_API void NvFlowBufferGetDesc(NvFlowBuffer* buffer, NvFlowBufferDesc* desc) {
    if (buffer) *desc = NvFlow::implCast<NvFlow::Buffer>(buffer)->m_desc;
}

NV_FLOW_API NvFlowBuffer* NvFlowCreateBuffer(NvFlowContext* context,
                                             const NvFlowBufferDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createBuffer(desc);
}

NV_FLOW_API NvFlowBuffer* NvFlowCreateBufferView(NvFlowContext* context,
                                                 NvFlowBuffer* buffer,
                                                 const NvFlowBufferViewDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createBufferView(
        NvFlow::implCast<NvFlow::Buffer>(buffer), desc);
}

NV_FLOW_API void NvFlowReleaseBuffer(NvFlowBuffer* buffer) {
    if (buffer) buffer->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowBufferGetContextObject(NvFlowBuffer* buffer) {
    return buffer;
}

NV_FLOW_API NvFlowResource* NvFlowBufferGetResource(NvFlowBuffer* buffer) {
    return NvFlow::implCast<NvFlow::Buffer>(buffer)->getResource();
}

NV_FLOW_API NvFlowResourceRW* NvFlowBufferGetResourceRW(NvFlowBuffer* buffer) {
    return NvFlow::implCast<NvFlow::Buffer>(buffer)->getResourceRW();
}

NV_FLOW_API void* NvFlowBufferMap(NvFlowContext* context, NvFlowBuffer* buffer) {
    if (buffer)
        return NvFlow::implCast<NvFlow::Context>(context)->map(
            NvFlow::implCast<NvFlow::Buffer>(buffer));
    return nullptr;
}

NV_FLOW_API void NvFlowBufferUnmap(NvFlowContext* context, NvFlowBuffer* buffer) {
    if (buffer)
        NvFlow::implCast<NvFlow::Context>(context)->unmap(
            NvFlow::implCast<NvFlow::Buffer>(buffer));
}

NV_FLOW_API void NvFlowBufferUnmapRange(NvFlowContext* context, NvFlowBuffer* buffer,
                                        NvFlowUint offset, NvFlowUint numBytes) {
    if (buffer)
        NvFlow::implCast<NvFlow::Context>(context)->unmap(
            NvFlow::implCast<NvFlow::Buffer>(buffer), offset, numBytes);
}

NV_FLOW_API void NvFlowBufferDownload(NvFlowContext* context, NvFlowBuffer* buffer) {
    if (buffer)
        NvFlow::implCast<NvFlow::Context>(context)->download(
            NvFlow::implCast<NvFlow::Buffer>(buffer));
}

NV_FLOW_API void NvFlowBufferDownloadRange(NvFlowContext* context, NvFlowBuffer* buffer,
                                           NvFlowUint offset, NvFlowUint numBytes) {
    if (buffer)
        NvFlow::implCast<NvFlow::Context>(context)->download(
            NvFlow::implCast<NvFlow::Buffer>(buffer), offset, numBytes);
}

NV_FLOW_API void* NvFlowBufferMapDownload(NvFlowContext* context, NvFlowBuffer* buffer) {
    if (buffer)
        return NvFlow::implCast<NvFlow::Context>(context)->mapDownload(
            NvFlow::implCast<NvFlow::Buffer>(buffer));
    return nullptr;
}

NV_FLOW_API void NvFlowBufferUnmapDownload(NvFlowContext* context, NvFlowBuffer* buffer) {
    if (buffer)
        NvFlow::implCast<NvFlow::Context>(context)->unmapDownload(
            NvFlow::implCast<NvFlow::Buffer>(buffer));
}

NV_FLOW_API void NvFlowTexture1DGetDesc(NvFlowTexture1D* tex, NvFlowTexture1DDesc* desc) {
    if (tex) *desc = NvFlow::implCast<NvFlow::Texture1D>(tex)->m_desc;
}

NV_FLOW_API NvFlowTexture1D* NvFlowCreateTexture1D(NvFlowContext* context,
                                                   const NvFlowTexture1DDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTexture1D(desc);
}

NV_FLOW_API void NvFlowReleaseTexture1D(NvFlowTexture1D* tex) {
    if (tex) tex->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowTexture1DGetContextObject(NvFlowTexture1D* tex) {
    return tex;
}

NV_FLOW_API NvFlowResource* NvFlowTexture1DGetResource(NvFlowTexture1D* tex) {
    return NvFlow::implCast<NvFlow::Texture1D>(tex)->getResource();
}

NV_FLOW_API NvFlowResourceRW* NvFlowTexture1DGetResourceRW(NvFlowTexture1D* tex) {
    return NvFlow::implCast<NvFlow::Texture1D>(tex)->getResourceRW();
}

NV_FLOW_API void* NvFlowTexture1DMap(NvFlowContext* context, NvFlowTexture1D* tex) {
    if (tex)
        return NvFlow::implCast<NvFlow::Context>(context)->map(
            NvFlow::implCast<NvFlow::Texture1D>(tex));
    return nullptr;
}

NV_FLOW_API void NvFlowTexture1DUnmap(NvFlowContext* context, NvFlowTexture1D* tex) {
    if (tex)
        NvFlow::implCast<NvFlow::Context>(context)->unmap(
            NvFlow::implCast<NvFlow::Texture1D>(tex));
}

NV_FLOW_API void NvFlowTexture2DGetDesc(NvFlowTexture2D* tex, NvFlowTexture2DDesc* desc) {
    if (tex) {
        *desc = NvFlow::implCast<NvFlow::Texture2D>(tex)->m_desc;
    }
}

NV_FLOW_API NvFlowTexture2D* NvFlowCreateTexture2D(NvFlowContext* context,
                                                   const NvFlowTexture2DDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTexture2D(desc);
}

NV_FLOW_API NvFlowTexture2D* NvFlowShareTexture2D(NvFlowContext* context,
                                                  NvFlowTexture2D* sharedTexture) {
    return NvFlow::implCast<NvFlow::Context>(context)->shareTexture2D(
        NvFlow::implCast<NvFlow::Texture2D>(sharedTexture));
}

NV_FLOW_API NvFlowTexture2D* NvFlowCreateTexture2DCrossAPI(
    NvFlowContext* context, const NvFlowTexture2DDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTexture2DShared(desc);
}

NV_FLOW_API NvFlowTexture2D* NvFlowShareTexture2DCrossAPI(NvFlowContext* context,
                                                          NvFlowTexture2D* sharedTexture) {
    return NvFlow::implCast<NvFlow::Context>(context)->shareTexture2DShared(
        NvFlow::implCast<NvFlow::Texture2D>(sharedTexture));
}

NV_FLOW_API void NvFlowReleaseTexture2D(NvFlowTexture2D* tex) {
    if (tex) tex->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowTexture2DGetContextObject(NvFlowTexture2D* tex) {
    return tex;
}

NV_FLOW_API NvFlowResource* NvFlowTexture2DGetResource(NvFlowTexture2D* tex) {
    if (tex) return NvFlow::implCast<NvFlow::Texture2D>(tex)->getResource();
    return nullptr;
}

NV_FLOW_API NvFlowResourceRW* NvFlowTexture2DGetResourceRW(NvFlowTexture2D* tex) {
    if (tex) return NvFlow::implCast<NvFlow::Texture2D>(tex)->getResourceRW();
    return nullptr;
}

NV_FLOW_API void NvFlowTexture3DGetDesc(NvFlowTexture3D* tex, NvFlowTexture3DDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::Texture3D>(tex)->m_desc;
}

NV_FLOW_API NvFlowTexture3D* NvFlowCreateTexture3D(NvFlowContext* context,
                                                   const NvFlowTexture3DDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTexture3D(desc);
}

NV_FLOW_API void NvFlowReleaseTexture3D(NvFlowTexture3D* tex) {
    if (tex) tex->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowTexture3DGetContextObject(NvFlowTexture3D* tex) {
    return tex;
}

NV_FLOW_API NvFlowResource* NvFlowTexture3DGetResource(NvFlowTexture3D* tex) {
    return NvFlow::implCast<NvFlow::Texture3D>(tex)->getResource();
}

NV_FLOW_API NvFlowResourceRW* NvFlowTexture3DGetResourceRW(NvFlowTexture3D* tex) {
    return NvFlow::implCast<NvFlow::Texture3D>(tex)->getResourceRW();
}

NV_FLOW_API NvFlowMappedData NvFlowTexture3DMap(NvFlowContext* context,
                                                NvFlowTexture3D* tex) {
    NvFlowMappedData mapped;
    NvFlow::implCast<NvFlow::Context>(context)->map(
        &mapped, NvFlow::implCast<NvFlow::Texture3D>(tex));
    return mapped;
}

NV_FLOW_API void NvFlowTexture3DUnmap(NvFlowContext* context, NvFlowTexture3D* tex) {
    NvFlow::implCast<NvFlow::Context>(context)->unmap(
        NvFlow::implCast<NvFlow::Texture3D>(tex));
}

NV_FLOW_API void NvFlowTexture3DDownload(NvFlowContext* context, NvFlowTexture3D* tex) {
    NvFlow::implCast<NvFlow::Context>(context)->download(
        NvFlow::implCast<NvFlow::Texture3D>(tex));
}

NV_FLOW_API NvFlowMappedData NvFlowTexture3DMapDownload(NvFlowContext* context,
                                                        NvFlowTexture3D* tex) {
    NvFlowMappedData mapped;
    NvFlow::implCast<NvFlow::Context>(context)->mapDownload(
        &mapped, NvFlow::implCast<NvFlow::Texture3D>(tex));
    return mapped;
}

NV_FLOW_API void NvFlowTexture3DUnmapDownload(NvFlowContext* context,
                                              NvFlowTexture3D* tex) {
    NvFlow::implCast<NvFlow::Context>(context)->unmapDownload(
        NvFlow::implCast<NvFlow::Texture3D>(tex));
}

NV_FLOW_API void NvFlowHeapSparseGetDesc(NvFlowHeapSparse* heap,
                                         NvFlowHeapSparseDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::HeapVTR>(heap)->m_desc;
}

NV_FLOW_API NvFlowHeapSparse* NvFlowCreateHeapSparse(NvFlowContext* context,
                                                     const NvFlowHeapSparseDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createHeapVTR(desc);
}

NV_FLOW_API void NvFlowReleaseHeapSparse(NvFlowHeapSparse* heap) {
    if (heap) heap->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowHeapSparseGetContextObject(NvFlowHeapSparse* heap) {
    return heap;
}

NV_FLOW_API void NvFlowTexture3DSparseGetDesc(NvFlowTexture3DSparse* tex,
                                              NvFlowTexture3DSparseDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::Texture3DVTR>(tex)->m_desc;
}

NV_FLOW_API NvFlowTexture3DSparse* NvFlowCreateTexture3DSparse(
    NvFlowContext* context, const NvFlowTexture3DSparseDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTexture3DVTR(desc);
}

NV_FLOW_API void NvFlowReleaseTexture3DSparse(NvFlowTexture3DSparse* tex) {
    if (tex) tex->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowTexture3DSparseGetContextObject(
    NvFlowTexture3DSparse* tex) {
    return tex;
}

NV_FLOW_API NvFlowResource* NvFlowTexture3DSparseGetResource(NvFlowTexture3DSparse* tex) {
    return NvFlow::implCast<NvFlow::Texture3DVTR>(tex)->getResource();
}

NV_FLOW_API NvFlowResourceRW* NvFlowTexture3DSparseGetResourceRW(
    NvFlowTexture3DSparse* tex) {
    return NvFlow::implCast<NvFlow::Texture3DVTR>(tex)->getResourceRW();
}

NV_FLOW_API void NvFlowColorBufferGetDesc(NvFlowColorBuffer* tex,
                                          NvFlowColorBufferDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::ColorBuffer>(tex)->m_desc;
}

NV_FLOW_API NvFlowColorBuffer* NvFlowCreateColorBuffer(NvFlowContext* context,
                                                       const NvFlowColorBufferDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createColorBuffer(desc);
}

NV_FLOW_API void NvFlowReleaseColorBuffer(NvFlowColorBuffer* tex) {
    if (tex) tex->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowColorBufferGetContextObject(NvFlowColorBuffer* tex) {
    return tex;
}

NV_FLOW_API NvFlowResource* NvFlowColorBufferGetResource(NvFlowColorBuffer* tex) {
    return NvFlow::implCast<NvFlow::ColorBuffer>(tex)->getResource();
}

NV_FLOW_API NvFlowResourceRW* NvFlowColorBufferGetResourceRW(NvFlowColorBuffer* tex) {
    return NvFlow::implCast<NvFlow::ColorBuffer>(tex)->getResourceRW();
}

NV_FLOW_API NvFlowRenderTarget* NvFlowColorBufferGetRenderTarget(NvFlowColorBuffer* tex) {
    return NvFlow::implCast<NvFlow::ColorBuffer>(tex)->getRenderTarget();
}

NV_FLOW_API void NvFlowDepthBufferGetDesc(NvFlowDepthBuffer* depthBuffer,
                                          NvFlowDepthBufferDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::DepthBuffer>(depthBuffer)->m_desc;
}

NV_FLOW_API NvFlowDepthBuffer* NvFlowCreateDepthBuffer(NvFlowContext* context,
                                                       const NvFlowDepthBufferDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createDepthBuffer(desc);
}

NV_FLOW_API void NvFlowReleaseDepthBuffer(NvFlowDepthBuffer* depthBuffer) {
    if (depthBuffer) depthBuffer->release();
}

NV_FLOW_API NvFlowContextObject* NvFlowDepthBufferGetContextObject(
    NvFlowDepthBuffer* depthBuffer) {
    return depthBuffer;
}

NV_FLOW_API NvFlowResource* NvFlowDepthBufferGetResource(NvFlowDepthBuffer* depthBuffer) {
    return NvFlow::implCast<NvFlow::DepthBuffer>(depthBuffer)->getResource();
}

NV_FLOW_API NvFlowDepthStencil* NvFlowDepthBufferGetDepthStencil(
    NvFlowDepthBuffer* depthBuffer) {
    return NvFlow::implCast<NvFlow::DepthBuffer>(depthBuffer)->getDepthStencil();
}

NV_FLOW_API NvFlowResource* NvFlowDepthStencilViewGetResource(NvFlowDepthStencilView* dsv) {
    return NvFlow::implCast<NvFlow::DepthStencilView>(dsv)->getResource();
}

NV_FLOW_API NvFlowDepthStencil* NvFlowDepthStencilViewGetDepthStencil(
    NvFlowDepthStencilView* dsv) {
    return NvFlow::implCast<NvFlow::DepthStencilView>(dsv)->getDepthStencil();
}

NV_FLOW_API void NvFlowDepthStencilViewGetDepthBufferDesc(NvFlowDepthStencilView* dsv,
                                                          NvFlowDepthBufferDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::DepthStencilView>(dsv)->getDepthBufferDesc();
}

NV_FLOW_API NvFlowRenderTarget* NvFlowRenderTargetViewGetRenderTarget(
    NvFlowRenderTargetView* rtv) {
    return NvFlow::implCast<NvFlow::RenderTargetView>(rtv)->getRenderTarget();
}

NV_FLOW_API NvFlowComputeShader* NvFlowCreateComputeShader(
    NvFlowContext* context, const NvFlowComputeShaderDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createComputeShader(desc);
}

NV_FLOW_API void NvFlowReleaseComputeShader(NvFlowComputeShader* computeShader) {
    if (computeShader) computeShader->release();
}

NV_FLOW_API void NvFlowGraphicsShaderGetDesc(NvFlowGraphicsShader* shader,
                                             NvFlowGraphicsShaderDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::GraphicsShader>(shader)->m_desc;
}

NV_FLOW_API NvFlowGraphicsShader* NvFlowCreateGraphicsShader(
    NvFlowContext* context, const NvFlowGraphicsShaderDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createGraphicsShader(desc);
}

NV_FLOW_API void NvFlowReleaseGraphicsShader(NvFlowGraphicsShader* shader) {
    if (shader) shader->release();
}

NV_FLOW_API void NvFlowGraphicsShaderSetFormats(NvFlowContext* context,
                                                NvFlowGraphicsShader* shader,
                                                NvFlowFormat renderTargetFormat,
                                                NvFlowFormat depthStencilFormat) {
    NvFlow::implCast<NvFlow::Context>(context)->setFormats(
        NvFlow::implCast<NvFlow::GraphicsShader>(shader), renderTargetFormat,
        depthStencilFormat);
}

NV_FLOW_API NvFlowContextTimer* NvFlowCreateContextTimer(NvFlowContext* context) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTimer();
}

NV_FLOW_API void NvFlowReleaseContextTimer(NvFlowContextTimer* timer) {
    if (timer) timer->release();
}

NV_FLOW_API NvFlowContextEventQueue* NvFlowCreateContextEventQueue(NvFlowContext* context) {
    return NvFlow::implCast<NvFlow::Context>(context)->createEventQueue();
}

NV_FLOW_API void NvFlowReleaseContextEventQueue(NvFlowContextEventQueue* eventQueue) {
    if (eventQueue) eventQueue->release();
}

NV_FLOW_API void NvFlowContextCopyConstantBuffer(NvFlowContext* context,
                                                 NvFlowConstantBuffer* dst,
                                                 NvFlowBuffer* src) {
    NvFlow::implCast<NvFlow::Context>(context)->copy(
        NvFlow::implCast<NvFlow::ConstantBuffer>(dst),
        NvFlow::implCast<NvFlow::Buffer>(src));
}

NV_FLOW_API void NvFlowContextCopyBuffer(NvFlowContext* context, NvFlowBuffer* dst,
                                         NvFlowBuffer* src, NvFlowUint offset,
                                         NvFlowUint numBytes) {
    NvFlow::implCast<NvFlow::Context>(context)->copy(NvFlow::implCast<NvFlow::Buffer>(dst),
                                                     NvFlow::implCast<NvFlow::Buffer>(src),
                                                     offset, numBytes);
}

NV_FLOW_API void NvFlowContextCopyTexture3D(NvFlowContext* context, NvFlowTexture3D* dst,
                                            NvFlowTexture3D* src) {
    NvFlow::implCast<NvFlow::Context>(context)->copy(
        NvFlow::implCast<NvFlow::Texture3D>(dst), NvFlow::implCast<NvFlow::Texture3D>(src));
}

NV_FLOW_API void NvFlowContextCopyResource(NvFlowContext* context,
                                           NvFlowResourceRW* resourceRW,
                                           NvFlowResource* resource) {
    NvFlow::implCast<NvFlow::Context>(context)->copy(
        NvFlow::implCast<NvFlow::ResourceRW>(resourceRW),
        NvFlow::implCast<NvFlow::Resource>(resource));
}

NV_FLOW_API void NvFlowContextDispatch(NvFlowContext* context,
                                       const NvFlowDispatchParams* params) {
    NvFlow::implCast<NvFlow::Context>(context)->dispatch(params);
}

NV_FLOW_API void NvFlowContextSetVertexBuffer(NvFlowContext* context,
                                              NvFlowVertexBuffer* vertexBuffer,
                                              NvFlowUint stride, NvFlowUint offset) {
    NvFlow::implCast<NvFlow::Context>(context)->setVertexBuffer(
        NvFlow::implCast<NvFlow::VertexBuffer>(vertexBuffer), stride, offset);
}

NV_FLOW_API void NvFlowContextSetIndexBuffer(NvFlowContext* context,
                                             NvFlowIndexBuffer* indexBuffer,
                                             NvFlowUint offset) {
    NvFlow::implCast<NvFlow::Context>(context)->setIndexBuffer(
        NvFlow::implCast<NvFlow::IndexBuffer>(indexBuffer), offset);
}

NV_FLOW_API void NvFlowContextDrawIndexedInstanced(NvFlowContext* context,
                                                   NvFlowUint indicesPerInstance,
                                                   NvFlowUint numInstances,
                                                   const NvFlowDrawParams* params) {
    NvFlow::implCast<NvFlow::Context>(context)->drawIndexedInstanced(indicesPerInstance,
                                                                     numInstances, params);
}

NV_FLOW_API void NvFlowContextSetRenderTarget(NvFlowContext* context,
                                              NvFlowRenderTarget* rt,
                                              NvFlowDepthStencil* ds) {
    NvFlow::implCast<NvFlow::Context>(context)->setRenderTarget(
        NvFlow::implCast<NvFlow::RenderTarget>(rt),
        NvFlow::implCast<NvFlow::DepthStencil>(ds));
}

NV_FLOW_API void NvFlowContextSetViewport(NvFlowContext* context,
                                          const NvFlowViewport* viewport) {
    NvFlow::implCast<NvFlow::Context>(context)->setViewport(viewport);
}

NV_FLOW_API void NvFlowContextClearRenderTarget(NvFlowContext* context,
                                                NvFlowRenderTarget* rt,
                                                const NvFlowFloat4 color) {
    NvFlow::implCast<NvFlow::Context>(context)->clearRenderTarget(
        NvFlow::implCast<NvFlow::RenderTarget>(rt), (float*)&color);
}

NV_FLOW_API void NvFlowContextClearDepthStencil(NvFlowContext* context,
                                                NvFlowDepthStencil* ds, const float depth) {
    NvFlow::implCast<NvFlow::Context>(context)->clearDepthStencil(
        NvFlow::implCast<NvFlow::DepthStencil>(ds), depth);
}

NV_FLOW_API void NvFlowContextRestoreResourceState(NvFlowContext* context,
                                                   NvFlowResource* resource) {
    NvFlow::implCast<NvFlow::Context>(context)->restoreResourceState(
        NvFlow::implCast<NvFlow::Resource>(resource));
}

NV_FLOW_API bool NvFlowContextIsSparseTextureSupported(NvFlowContext* context) {
    return NvFlow::implCast<NvFlow::Context>(context)->is_VTR_supported();
}

NV_FLOW_API void NvFlowContextUpdateSparseMapping(
    NvFlowContext* context, NvFlowTexture3DSparse* tex, NvFlowHeapSparse* heap,
    NvFlowUint* blockTableImage, NvFlowUint rowPitch, NvFlowUint depthPitch) {
    NvFlow::implCast<NvFlow::Context>(context)->updateVTRMapping(
        NvFlow::implCast<NvFlow::Texture3DVTR>(tex),
        NvFlow::implCast<NvFlow::HeapVTR>(heap), blockTableImage, rowPitch, depthPitch);
}

NV_FLOW_API void NvFlowContextTimerBegin(NvFlowContext* context,
                                         NvFlowContextTimer* timer) {
    NvFlow::implCast<NvFlow::Context>(context)->timerBegin(
        NvFlow::implCast<NvFlow::Timer>(timer));
}

NV_FLOW_API void NvFlowContextTimerEnd(NvFlowContext* context, NvFlowContextTimer* timer) {
    NvFlow::implCast<NvFlow::Context>(context)->timerEnd(
        NvFlow::implCast<NvFlow::Timer>(timer));
}

NV_FLOW_API void NvFlowContextTimerGetResult(NvFlowContext* context,
                                             NvFlowContextTimer* timer, float* timeGPU,
                                             float* timeCPU) {
    NvFlow::implCast<NvFlow::Context>(context)->timerGetResult(
        NvFlow::implCast<NvFlow::Timer>(timer), timeGPU, timeCPU);
}

NV_FLOW_API void NvFlowContextEventQueuePush(NvFlowContext* context,
                                             NvFlowContextEventQueue* eventQueue,
                                             NvFlowUint64 uid) {
    NvFlow::implCast<NvFlow::Context>(context)->eventQueuePush(
        NvFlow::implCast<NvFlow::EventQueue>(eventQueue), uid);
}

NV_FLOW_API NvFlowResult NvFlowContextEventQueuePop(NvFlowContext* context,
                                                    NvFlowContextEventQueue* eventQueue,
                                                    NvFlowUint64* pUid) {
    return (NvFlowResult)NvFlow::implCast<NvFlow::Context>(context)->eventQueuePop(
        NvFlow::implCast<NvFlow::EventQueue>(eventQueue), pUid);
}

NV_FLOW_API void NvFlowContextProfileGroupBegin(NvFlowContext* context,
                                                const wchar_t* label) {
    NvFlow::implCast<NvFlow::Context>(context)->profileGroupBegin(label);
}

NV_FLOW_API void NvFlowContextProfileGroupEnd(NvFlowContext* context) {
    NvFlow::implCast<NvFlow::Context>(context)->profileGroupEnd();
}

NV_FLOW_API void NvFlowContextProfileItemBegin(NvFlowContext* context,
                                               const wchar_t* label) {
    NvFlow::implCast<NvFlow::Context>(context)->profileItemBegin(label);
}

NV_FLOW_API void NvFlowContextProfileItemEnd(NvFlowContext* context) {
    NvFlow::implCast<NvFlow::Context>(context)->profileItemEnd();
}

NV_FLOW_API void NvFlowFenceGetDesc(NvFlowFence* fence, NvFlowFenceDesc* desc) {
    *desc = NvFlow::implCast<NvFlow::Fence>(fence)->m_desc;
}

NV_FLOW_API NvFlowFence* NvFlowCreateFence(NvFlowContext* context,
                                           const NvFlowFenceDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createFence(desc);
}

NV_FLOW_API NvFlowFence* NvFlowShareFence(NvFlowContext* context, NvFlowFence* fence) {
    return NvFlow::implCast<NvFlow::Context>(context)->shareFence(
        NvFlow::implCast<NvFlow::Fence>(fence));
}

NV_FLOW_API void NvFlowReleaseFence(NvFlowFence* fence) {
    if (fence) fence->release();
}

NV_FLOW_API void NvFlowContextWaitOnFence(NvFlowContext* context, NvFlowFence* fence,
                                          NvFlowUint64 fenceValue) {
    NvFlow::implCast<NvFlow::Context>(context)->waitOnFence(
        NvFlow::implCast<NvFlow::Fence>(fence), fenceValue);
}

NV_FLOW_API void NvFlowContextSignalFence(NvFlowContext* context, NvFlowFence* fence,
                                          NvFlowUint64 fenceValue) {
    NvFlow::implCast<NvFlow::Context>(context)->signalFence(
        NvFlow::implCast<NvFlow::Fence>(fence), fenceValue);
}

NV_FLOW_API NvFlowTexture2DCrossAdapter* NvFlowCreateTexture2DCrossAdapter(
    NvFlowContext* context, const NvFlowTexture2DDesc* desc) {
    return NvFlow::implCast<NvFlow::Context>(context)->createTexture2DCrossAdapter(desc);
}

NV_FLOW_API NvFlowTexture2DCrossAdapter* NvFlowShareTexture2DCrossAdapter(
    NvFlowContext* context, NvFlowTexture2DCrossAdapter* sharedTexture) {
    return NvFlow::implCast<NvFlow::Context>(context)->shareTexture2DCrossAdapter(
        NvFlow::implCast<NvFlow::Texture2DCrossAdapter>(sharedTexture));
}

NV_FLOW_API void NvFlowReleaseTexture2DCrossAdapter(NvFlowTexture2DCrossAdapter* tex) {
    if (tex) tex->release();
}

NV_FLOW_API void NvFlowContextTransitionToCommonState(NvFlowContext* context,
                                                      NvFlowResource* resource) {
    NvFlow::implCast<NvFlow::Context>(context)->transitionToCommonState(
        NvFlow::implCast<NvFlow::Resource>(resource));
}

NV_FLOW_API void NvFlowContextCopyToTexture2DCrossAdapter(NvFlowContext* context,
                                                          NvFlowTexture2DCrossAdapter* dst,
                                                          NvFlowTexture2D* src,
                                                          NvFlowUint height) {
    NvFlow::implCast<NvFlow::Context>(context)->copyToShared(
        NvFlow::implCast<NvFlow::Texture2DCrossAdapter>(dst),
        NvFlow::implCast<NvFlow::Texture2D>(src), height);
}

NV_FLOW_API void NvFlowContextCopyFromTexture2DCrossAdapter(
    NvFlowContext* context, NvFlowTexture2D* dst, NvFlowTexture2DCrossAdapter* src,
    NvFlowUint height) {
    NvFlow::implCast<NvFlow::Context>(context)->copyFromShared(
        NvFlow::implCast<NvFlow::Texture2D>(dst),
        NvFlow::implCast<NvFlow::Texture2DCrossAdapter>(src), height);
}

NV_FLOW_API NvFlowResourceReference* NvFlowShareResourceReference(
    NvFlowContext* context, NvFlowResource* resource) {
    return NvFlow::implCast<NvFlow::Context>(context)->shareResourceReference(
        NvFlow::implCast<NvFlow::Resource>(resource));
}

NV_FLOW_API void NvFlowReleaseResourceReference(NvFlowResourceReference* resource) {
    if (resource) resource->release();
}
