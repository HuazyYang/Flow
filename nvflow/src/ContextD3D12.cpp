#include "ContextD3D12.h"
#include <d3d12.h>

namespace NvFlow {

int64_t FlowDeferredReleaseD3D12(float timeoutMS) {
    return 0;
}

extern bool FlowDedicatedDeviceAvailableD3D12(NvFlowContext *context) {
    return 0;
}

extern bool FlowDedicatedDeviceQueueAvailableD3D12(NvFlowContext *renderContext) {
    return 1;
}

}  // namespace NvFlow
