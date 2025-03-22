#ifndef NVFLOW_DEVICED3D12_H
#define NVFLOW_DEVICED3D12_H
#include "Device.h"
#include "Object.h"
#include "ClientHelper.h"

namespace NvFlow {

struct DeviceD3D12 : Object, NvFlowDevice {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    NvFlowDeviceQueue *createDeviceQueue(const NvFlowDeviceQueueDesc *desc) override;

    // Details
    DeviceD3D12(NvFlowContext *renderContext, const NvFlowDeviceDesc *desc);
    ~DeviceD3D12();
};

struct DeviceQueueD3D12 : Object, NvFlowDeviceQueue {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    NvFlowContext *createContext() override;
    void updateContext(NvFlowContext *context,
                       const NvFlowDeviceQueueStatus *status) override;
    void flush(NvFlowContext *context) override;
    void conditionalFlush(NvFlowContext *context) override;
    void waitOnFence(NvFlowContext *context, uint64_t fenceId) override;

    DeviceQueueD3D12(DeviceD3D12 *parent, const NvFlowDeviceQueueDesc *desc);
    ~DeviceQueueD3D12();
};
}  // namespace NvFlow

#endif /* NVFLOW_DEVICED3D12_H */
