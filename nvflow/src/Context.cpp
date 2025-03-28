#include "Context.h"
#include "ClientHelper.h"

namespace NvFlow {

extern int64_t FlowDeferredReleaseD3D11(float timeoutMS);

extern int64_t FlowDeferredReleaseD3D12(float timeoutMS);

uint64_t Context::getGPUBytesUsed() {
    return 0;
}

void Context::flushRequestPush() {
    m_flushRequestPending = 1;
}
bool Context::flushRequestPop() {
    bool prev = m_flushRequestPending;
    m_flushRequestPending = false;
    return prev;
}

void Context::profileGroupBegin(const wchar_t *) {}
void Context::profileGroupEnd() {
}
void Context::profileItemBegin(const wchar_t *) {
}
void Context::profileItemEnd() {
}

uint64_t FlowDeferredRelease(float timeoutMS) {
    uint64_t sum = FlowDeferredReleaseD3D11(timeoutMS);
    sum += FlowDeferredReleaseD3D12(timeoutMS);
    return sum;
}
IDXGIFactory1 *getDXGIFactoryD3D(IDXGIAdapter1 *pAdapter1) {
    IDXGIFactory1 *pFactory1 = 0;
    if (pAdapter1) {
        IDXGIFactory *pFactory = 0;
        if (FAILED(pAdapter1->GetParent(IID_PPV_ARGS(&pFactory)))) return 0;

        if (FAILED(pFactory->QueryInterface(IID_PPV_ARGS(&pFactory1)))) {
            SafeRelease(pFactory);
            return 0;
        }
        SafeRelease(pFactory);
    }
    return pFactory1;
}
}  // namespace NvFlow