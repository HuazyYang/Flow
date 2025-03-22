#ifndef NVFLOW_GRIDPROXY_H
#define NVFLOW_GRIDPROXY_H
#include "NvFlowImpl.h"
#include "NvFlowObjectImpl.h"

struct NvFlowGridProxy : NvFlowObject {
    virtual void push(NvFlowGridExport *gridExport,
                      const NvFlowGridProxyFlushParams *params) = 0;
    virtual void flush(const NvFlowGridProxyFlushParams *params) = 0;
    virtual NvFlowGridExport *getGridExport(NvFlowContext *renderContext) = 0;
};

namespace NvFlow {

NvFlowGridProxy *FlowCreateGridProxy(const NvFlowGridProxyDesc *desc);
}  // namespace NvFlow

#endif /* NVFLOW_GRIDPROXY_H */
