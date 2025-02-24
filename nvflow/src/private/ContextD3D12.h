#ifndef CONTEXTD3D12_H
#define CONTEXTD3D12_H
#include "Context.h"
#include <d3d12.h>

namespace NvFlow {

int64_t FlowDeferredReleaseD3D12(float timeoutMS);
}

#endif /* CONTEXTD3D12_H */
