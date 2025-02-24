#ifndef NVFLOWCONTEXTIMPL_H
#define NVFLOWCONTEXTIMPL_H
#include "NvFlowObjectImpl.h"
#include <nvflow/NvFlowContext.h>

struct NvFlowContextObject : NvFlowObject {};

struct NvFlowConstantBuffer : NvFlowContextObject {};

struct NvFlowVertexBuffer : NvFlowContextObject {};

struct NvFlowIndexBuffer : NvFlowContextObject {};

struct NvFlowDepthStencilView : NvFlowContextObject {};

struct NvFlowRenderTargetView : NvFlowContextObject {};

struct NvFlowResource : NvFlowContextObject {};

struct NvFlowResourceRW : NvFlowContextObject {};

struct NvFlowDepthStencil : NvFlowContextObject {};

struct NvFlowRenderTarget : NvFlowContextObject {};

struct NvFlowBuffer : NvFlowContextObject {};

struct NvFlowTexture1D : NvFlowContextObject {};

struct NvFlowTexture2D : NvFlowContextObject {};

struct NvFlowTexture3D : NvFlowContextObject {};

struct NvFlowTexture2DCrossAdapter : NvFlowContextObject {};

struct NvFlowHeapSparse : NvFlowContextObject {};

struct NvFlowTexture3DSparse : NvFlowContextObject {};

struct NvFlowColorBuffer : NvFlowContextObject {};

struct NvFlowDepthBuffer : NvFlowContextObject {};

struct NvFlowComputeShader : NvFlowContextObject {};

struct NvFlowGraphicsShader : NvFlowContextObject {};

struct NvFlowContextTimer : NvFlowContextObject {};

struct NvFlowContextEventQueue : NvFlowContextObject {};

struct NvFlowResourceReference : NvFlowContextObject {};

struct NvFlowFence : NvFlowContextObject {};

struct NvFlowContext : NvFlowContextObject {
    virtual NvFlowContextAPI getContextType() = 0;

    virtual void flushRequestPush() = 0;

    virtual bool flushRequestPop() = 0;

    virtual void processFenceSignal(NvFlowContext *context) = 0;

    virtual void processFenceWait(NvFlowContext *context) = 0;

    virtual void contextPush() = 0;

    virtual void contextPop() = 0;
};

#endif /* NVFLOWCONTEXTIMPL_H */
