#include "Device.h"

namespace NvFlow {

extern bool FlowDedicatedDeviceAvailableD3D11(NvFlowContext *context);
extern bool FlowDedicatedDeviceAvailableD3D12(NvFlowContext *context);

bool FlowDedicatedDeviceAvailable(NvFlowContext *renderContext) {
    NvFlowContextAPI api = NvFlowContextGetContextType(renderContext);
    if (api == eNvFlowContextD3D11)
        return FlowDedicatedDeviceAvailableD3D11(renderContext);
    else if (api == eNvFlowContextD3D12)
        return FlowDedicatedDeviceAvailableD3D12(renderContext);
    return 0;
}

extern bool FlowDedicatedDeviceQueueAvailableD3D11(NvFlowContext *renderContext);
extern bool FlowDedicatedDeviceQueueAvailableD3D12(NvFlowContext *renderContext);

bool FlowDedicatedDeviceQueueAvailable(NvFlowContext *renderContext) {
    NvFlowContextAPI api = NvFlowContextGetContextType(renderContext);
    if (api == eNvFlowContextD3D11)
        return FlowDedicatedDeviceQueueAvailableD3D11(renderContext);
    else if (api = eNvFlowContextD3D12)
        return FlowDedicatedDeviceQueueAvailableD3D12(renderContext);
    return 0;
}

}  // namespace NvFlow