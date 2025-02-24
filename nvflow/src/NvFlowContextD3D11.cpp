#include "ContextD3D11.h"

NV_FLOW_API NvFlowContext* NvFlowCreateContextD3D11(NvFlowUint version,
                                                    const NvFlowContextDescD3D11* desc) {
    return new NvFlow::ContextD3D11(desc);
}

NV_FLOW_API NvFlowDepthStencilView* NvFlowCreateDepthStencilViewD3D11(
    NvFlowContext* context, const NvFlowDepthStencilViewDescD3D11* desc) {
    return NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->createDepthStencilView(desc);
}

NV_FLOW_API NvFlowRenderTargetView* NvFlowCreateRenderTargetViewD3D11(
    NvFlowContext* context, const NvFlowRenderTargetViewDescD3D11* desc) {
    return NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->createRenderTargetView(desc);
}

NV_FLOW_API void NvFlowUpdateContextD3D11(NvFlowContext* context,
                                          const NvFlowContextDescD3D11* desc) {
    NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->updateContext(desc);
}

NV_FLOW_API void NvFlowUpdateContextDescD3D11(NvFlowContext* context,
                                              NvFlowContextDescD3D11* desc) {
    NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->updateContextDesc(desc);
}

NV_FLOW_API void NvFlowUpdateDepthStencilViewD3D11(
    NvFlowContext* context, NvFlowDepthStencilView* view,
    const NvFlowDepthStencilViewDescD3D11* desc) {
    NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->updateDepthStencilView(view, desc);
}

NV_FLOW_API void NvFlowUpdateRenderTargetViewD3D11(
    NvFlowContext* context, NvFlowRenderTargetView* view,
    const NvFlowRenderTargetViewDescD3D11* desc) {
    NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->updateRenderTargetView(view, desc);
}

NV_FLOW_API void NvFlowUpdateResourceViewDescD3D11(NvFlowContext* context,
                                                   NvFlowResource* resource,
                                                   NvFlowResourceViewDescD3D11* desc) {
    NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->updateResourceViewDesc(
        NvFlow::implSafeCast<NvFlow::ResourceD3D11>(resource), desc);
}

NV_FLOW_API void NvFlowUpdateResourceRWViewDescD3D11(NvFlowContext* context,
                                                     NvFlowResourceRW* resourceRW,
                                                     NvFlowResourceRWViewDescD3D11* desc) {
    NvFlow::implSafeCast<NvFlow::ContextD3D11>(context)->updateResourceRWViewDesc(
        NvFlow::implSafeCast<NvFlow::ResourceRWD3D11>(resourceRW), desc);
}
