#include "DeviceD3D12.h"

uint64_t NvFlow::DeviceD3D12::getGPUBytesUsed() {
    return 0;
}

NvFlowDeviceQueue* NvFlow::DeviceD3D12::createDeviceQueue(
    const NvFlowDeviceQueueDesc* desc) {
    return nullptr;
}

NvFlow::DeviceD3D12::DeviceD3D12(NvFlowContext* renderContext,
                                 const NvFlowDeviceDesc* desc) {}

NvFlow::DeviceD3D12::~DeviceD3D12() {}

uint64_t NvFlow::DeviceQueueD3D12::getGPUBytesUsed() {
    return 0;
}

NvFlowContext* NvFlow::DeviceQueueD3D12::createContext() {
    return nullptr;
}

void NvFlow::DeviceQueueD3D12::updateContext(NvFlowContext* context,
                                             const NvFlowDeviceQueueStatus* status) {}

void NvFlow::DeviceQueueD3D12::flush(NvFlowContext* context) {}

void NvFlow::DeviceQueueD3D12::conditionalFlush(NvFlowContext* context) {}

void NvFlow::DeviceQueueD3D12::waitOnFence(NvFlowContext* context, uint64_t fenceId) {}

NvFlow::DeviceQueueD3D12::DeviceQueueD3D12(DeviceD3D12* parent,
                                           const NvFlowDeviceQueueDesc* desc) {}

NvFlow::DeviceQueueD3D12::~DeviceQueueD3D12() {}
