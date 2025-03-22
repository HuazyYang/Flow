#ifndef NVFLOW_DEVICE_H
#define NVFLOW_DEVICE_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"

struct NvFlowDeviceQueue;

struct NvFlowDevice : NvFlowObject {
    virtual NvFlowDeviceQueue *createDeviceQueue(const NvFlowDeviceQueueDesc *desc) = 0;
};

struct NvFlowDeviceQueue : NvFlowObject {
    virtual NvFlowContext *createContext() = 0;
    virtual void updateContext(NvFlowContext *context,
                               const NvFlowDeviceQueueStatus *status) = 0;
    virtual void flush(NvFlowContext *context) = 0;
    virtual void conditionalFlush(NvFlowContext *context) = 0;
    virtual void waitOnFence(NvFlowContext *context, uint64_t fenceId) = 0;
};

namespace NvFlow {

bool FlowDedicatedDeviceAvailable(NvFlowContext *renderContext);

bool FlowDedicatedDeviceQueueAvailable(NvFlowContext *renderContext);

}  // namespace NvFlow

#endif /* NVFLOW_DEVICE_H */
