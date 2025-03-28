#include "Types.h"
#include "ContextD3D12.h"
#include <nvflow/NvFlowContextD3D12.h>

using namespace NvFlow;

NV_FLOW_API NvFlowContext* NvFlowCreateContextD3D12(NvFlowUint version,
                                                    const NvFlowContextDescD3D12* desc) {
    return new ContextD3D12(desc);
}

NV_FLOW_API NvFlowDepthStencilView* NvFlowCreateDepthStencilViewD3D12(
    NvFlowContext* context, const NvFlowDepthStencilViewDescD3D12* desc) {
    return implSafeCast<ContextD3D12>(context)->createDepthStencilView(desc);
}

NV_FLOW_API NvFlowRenderTargetView* NvFlowCreateRenderTargetViewD3D12(
    NvFlowContext* context, const NvFlowRenderTargetViewDescD3D12* desc) {
    return implSafeCast<ContextD3D12>(context)->createRenderTargetView(desc);
}

NV_FLOW_API void NvFlowUpdateContextD3D12(NvFlowContext* context,
                                          const NvFlowContextDescD3D12* desc) {
    implSafeCast<ContextD3D12>(context)->updateContext(desc);
}

NV_FLOW_API void NvFlowUpdateContextDescD3D12(NvFlowContext* context,
                                              NvFlowContextDescD3D12* desc) {
    implSafeCast<ContextD3D12>(context)->updateContextDesc(desc);
}

NV_FLOW_API void NvFlowUpdateDepthStencilViewD3D12(
    NvFlowContext* context, NvFlowDepthStencilView* view,
    const NvFlowDepthStencilViewDescD3D12* desc) {
    implSafeCast<ContextD3D12>(context)->updateDepthStencilView(view, desc);
}

NV_FLOW_API void NvFlowUpdateRenderTargetViewD3D12(
    NvFlowContext* context, NvFlowRenderTargetView* view,
    const NvFlowRenderTargetViewDescD3D12* desc) {
    implSafeCast<ContextD3D12>(context)->updateRenderTargetView(view, desc);
}

NV_FLOW_API void NvFlowUpdateResourceViewDescD3D12(NvFlowContext* context,
                                                   NvFlowResource* resource,
                                                   NvFlowResourceViewDescD3D12* desc) {
    implSafeCast<ContextD3D12>(context)->updateResourceViewDesc(
        implSafeCast<ResourceD3D12>(resource), desc);
}

NV_FLOW_API void NvFlowUpdateResourceRWViewDescD3D12(NvFlowContext* context,
                                                     NvFlowResourceRW* resourceRW,
                                                     NvFlowResourceRWViewDescD3D12* desc) {
    implSafeCast<ContextD3D12>(context)->updateResoruceRWViewDesc(
        implSafeCast<ResourceRWD3D12>(resourceRW), desc);
}
