#ifndef CONTEXT_H
#define CONTEXT_H
#include "NvFlowContextImpl.h"
#include <nvflow/NvFlowContextExt.h>
#include "Object.h"
#include <dxgi.h>

namespace NvFlow {

uint64_t FlowDeferredRelease(float timeoutMS);

struct ConstantBuffer : NvFlowConstantBuffer {
    NvFlowConstantBufferDesc m_desc;
};

struct VertexBuffer : NvFlowVertexBuffer {
    NvFlowVertexBufferDesc m_desc;
};

struct IndexBuffer : NvFlowIndexBuffer {
    NvFlowIndexBufferDesc m_desc;
};

struct Resource : NvFlowResource {};

struct ResourceRW : NvFlowResourceRW {
    virtual Resource *getResource() = 0;
};

struct DepthStencil : NvFlowDepthStencil {
    NvFlowFormat m_ds_format;
    NvFlowViewport m_viewport;
    float m_width;
    float m_height;
};

struct RenderTarget : NvFlowRenderTarget {
    NvFlowFormat m_rt_format;
    NvFlowViewport m_viewport;
};

struct Buffer : NvFlowBuffer {
    NvFlowBufferDesc m_desc;

    virtual Resource *getResource() = 0;
    virtual ResourceRW *getResourceRW() = 0;
};

struct Texture1D : NvFlowTexture1D {
    NvFlowTexture1DDesc m_desc;

    virtual Resource *getResource() = 0;
    virtual ResourceRW *getResourceRW() = 0;
};

struct Texture2D : NvFlowTexture2D {
    virtual Resource *getResource() = 0;
    virtual ResourceRW *getResourceRW() = 0;

    virtual void openSharedHandle(HANDLE *) = 0;
    virtual void closeSharedHandle(HANDLE) = 0;

    NvFlowTexture2DDesc m_desc;
};

struct Texture3D : NvFlowTexture3D {
    virtual Resource *getResource() = 0;
    virtual ResourceRW *getResourceRW() = 0;

    NvFlowTexture3DDesc m_desc;
};

struct Texture2DCrossAdapter : NvFlowTexture2DCrossAdapter {};

struct ResourceReference : NvFlowResourceReference {};

struct HeapVTR : NvFlowHeapSparse {
    NvFlowHeapSparseDesc m_desc;
    uint32_t m_numTiles;
};

struct Texture3DVTR : NvFlowTexture3DSparse {
    virtual Resource *getResource() = 0;
    virtual ResourceRW *getResourceRW() = 0;

    NvFlowTexture3DSparseDesc m_desc;
    NvFlowDim m_blockDim;
    NvFlowDim m_gridDim;
};

struct ColorBuffer : NvFlowColorBuffer {
    virtual Resource *getResource() = 0;
    virtual ResourceRW *getResourceRW() = 0;
    virtual RenderTarget *getRenderTarget() = 0;

    NvFlowColorBufferDesc m_desc;
};

struct DepthBuffer : NvFlowDepthBuffer {
    virtual Resource *getResource() = 0;
    virtual DepthStencil *getDepthStencil() = 0;

    NvFlowDepthBufferDesc m_desc;
};

struct DepthStencilView : NvFlowDepthStencilView {
    virtual Resource *getResource() = 0;
    virtual DepthStencil *getDepthStencil() = 0;
    virtual NvFlowDepthBufferDesc getDepthBufferDesc() = 0;
};

struct RenderTargetView : NvFlowRenderTargetView {
    virtual RenderTarget *getRenderTarget() = 0;
};

struct ComputeShader : NvFlowComputeShader {
    NvFlowComputeShaderDesc m_desc;
};

struct GraphicsShader : NvFlowGraphicsShader {
    NvFlowGraphicsShaderDesc m_desc;
};

struct Timer : NvFlowContextTimer {};

struct EventQueue : NvFlowContextEventQueue {};

struct Fence : NvFlowFence {
    NvFlowFenceDesc m_desc;
};

struct Context : NvFlowContext {
 public:
    uint64_t getGPUBytesUsed() override;  // Default to zero

    void flushRequestPush() override;

    bool flushRequestPop() override;

    virtual ConstantBuffer *createConstantBuffer(const NvFlowConstantBufferDesc *desc) = 0;

    virtual NvFlowMappedData *map(NvFlowMappedData *result, Texture3D *buffer) = 0;

    virtual void *map(Buffer *buffer) = 0;

    virtual void *map(ConstantBuffer *buffer) = 0;

    virtual void *map(IndexBuffer *buffer) = 0;

    virtual void *map(Texture1D *buffer) = 0;

    virtual void *map(VertexBuffer *buffer) = 0;

    virtual void unmap(Buffer *buffer) = 0;

    virtual void unmap(Buffer *buffer, uint32_t offset, uint32_t numBytes) = 0;

    virtual void unmap(ConstantBuffer *buffer) = 0;

    virtual void unmap(IndexBuffer *buffer) = 0;

    virtual void unmap(VertexBuffer *buffer) = 0;

    virtual void unmap(Texture1D *texture) = 0;

    virtual void unmap(Texture3D *texture) = 0;

    virtual void copy(Buffer *dst, Buffer *src, uint32_t offset, uint32_t numBytes) = 0;

    virtual void copy(Buffer *dst, Resource *src, uint32_t offset, uint32_t numBytes) = 0;

    virtual void copy(ConstantBuffer *dst, Buffer *src) = 0;

    virtual void copy(DepthStencil *dst, Resource *src) = 0;

    virtual void copy(ResourceRW *dst, Resource *src) = 0;

    virtual void copy(Texture3D *dst, Texture3D *src) = 0;

    virtual void copy(Texture3D *dst, Resource *src) = 0;

    virtual VertexBuffer *createVertexBuffer(const NvFlowVertexBufferDesc *desc) = 0;

    virtual IndexBuffer *createIndexBuffer(const NvFlowIndexBufferDesc *desc) = 0;

    virtual Buffer *createBuffer(const NvFlowBufferDesc *desc) = 0;

    virtual Buffer *createBufferView(Buffer *buffer, const NvFlowBufferViewDesc *desc) = 0;

    virtual void download(Buffer *buffer) = 0;

    virtual void download(Buffer *buffer, uint32_t offset, uint32_t numBytes) = 0;

    virtual void download(Texture3D *buffer) = 0;

    virtual NvFlowMappedData *mapDownload(NvFlowMappedData *result, Texture3D *buffer) = 0;

    virtual void *mapDownload(Buffer *buffer) = 0;

    virtual void unmapDownload(Buffer *buffer) = 0;

    virtual void unmapDownload(Texture3D *texture) = 0;

    virtual Texture1D *createTexture1D(const NvFlowTexture1DDesc *desc) = 0;

    virtual Texture2D *createTexture2D(const NvFlowTexture2DDesc *desc) = 0;

    virtual Texture2D *shareTexture2D(Texture2D *sharedTexture) = 0;

    virtual Texture2D *shareTexture2DShared(Texture2D *sharedTexture) = 0;

    virtual Texture2D *createTexture2DShared(const NvFlowTexture2DDesc *desc) = 0;

    virtual Texture3D *createTexture3D(const NvFlowTexture3DDesc *desc) = 0;

    virtual ResourceReference *shareResourceReference(Resource *resource) = 0;

    virtual HeapVTR *createHeapVTR(const NvFlowHeapSparseDesc *desc) = 0;

    virtual Texture3DVTR *createTexture3DVTR(const NvFlowTexture3DSparseDesc *desc) = 0;

    virtual ColorBuffer *createColorBuffer(const NvFlowColorBufferDesc *desc) = 0;

    virtual DepthBuffer *createDepthBuffer(const NvFlowDepthBufferDesc *desc) = 0;

    virtual ComputeShader *createComputeShader(const NvFlowComputeShaderDesc *desc) = 0;

    virtual GraphicsShader *createGraphicsShader(const NvFlowGraphicsShaderDesc *desc) = 0;

    virtual void setFormats(GraphicsShader *graphicsShader, NvFlowFormat renderTargetFormat,
                            NvFlowFormat depthStencilFormat) = 0;

    virtual Texture2DCrossAdapter *createTexture2DCrossAdapter(const NvFlowTexture2DDesc *desc) = 0;

    virtual Texture2DCrossAdapter *shareTexture2DCrossAdapter(
        Texture2DCrossAdapter *sharedTexture) = 0;

    virtual void transitionToCommonState(Resource *resource) = 0;

    virtual void copyFromShared(Texture2D *dstTexture, Texture2DCrossAdapter *sharedTexture,
                                uint32_t height) = 0;

    virtual void copyToShared(Texture2DCrossAdapter *dstSharedTexture,
                              Texture2D *srcTexture, uint32_t height) = 0;

    virtual Fence *createFence(const NvFlowFenceDesc *desc) = 0;

    virtual Fence *shareFence(Fence *fence) = 0;

    virtual void waitOnFence(Fence *fence, uint64_t fenceValue) = 0;

    virtual void signalFence(Fence *fence, uint64_t fenceValue) = 0;

    virtual void dispatch(const NvFlowDispatchParams *params) = 0;

    virtual void setVertexBuffer(VertexBuffer *buffer, uint32_t stride,
                                 uint32_t offset) = 0;

    virtual void setIndexBuffer(IndexBuffer *buffer, uint32_t offset) = 0;

    virtual void drawIndexedInstanced(uint32_t indicesPerInstance, uint32_t numInstances,
                                      const NvFlowDrawParams *params) = 0;

    virtual void setRenderTarget(RenderTarget *rtv, DepthStencil *dsv) = 0;

    virtual void setViewport(const NvFlowViewport *vp) = 0;

    virtual void clearRenderTarget(RenderTarget *rtv, const float color[4]) = 0;

    virtual void clearDepthStencil(DepthStencil *dsv, float depth) = 0;

    virtual void restoreResourceState(Resource *resource) = 0;

    virtual int is_VTR_supported() = 0;

    virtual void updateVTRMapping(Texture3DVTR *textureIn, HeapVTR *heapIn,
                                  uint32_t *blockTableImage, uint32_t rowPitch,
                                  uint32_t depthPitch) = 0;

    virtual Timer *createTimer() = 0;

    virtual void timerBegin(Timer *timer) = 0;

    virtual void timerEnd(Timer *timer) = 0;

    virtual int timerGetResult(Timer *timer, float *timeGPU, float *timeCPU) = 0;

    virtual EventQueue *createEventQueue() = 0;

    virtual void eventQueuePush(EventQueue *eventQueueIn, uint64_t uid) = 0;

    virtual int eventQueuePop(EventQueue *eventQueueIn, uint64_t *pUid) = 0;

    virtual void profileGroupBegin(const wchar_t *);
    virtual void profileGroupEnd();

    virtual void profileItemBegin(const wchar_t *);
    virtual void profileItemEnd();

    struct Profiler : NvFlowObject {
        virtual void begin(const wchar_t *label) = 0;
        virtual void end() = 0;
    };

 protected:
    ~Context() = default;

    bool m_flushRequestPending = 0;
    Profiler *m_profiler = nullptr;
};

}  // namespace NvFlow

#endif /* CONTEXT_H */
