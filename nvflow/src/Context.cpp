#include "Context.h"

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
void Context::profileGroupBegin(const wchar_t *) {
}
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
}  // namespace NvFlow