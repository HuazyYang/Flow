#include "traceUtils.h"
#include "appGraphCtx.h"


AppGraphCtx* g_pAppGraphContext = nullptr;

void test() {

}


void _TraceCallBegin(const char* call) {

    AppGraphCtxBeginMarker(g_pAppGraphContext, call);
}

void _TraceCallEnd() {
    AppGraphCtxEndMarker(g_pAppGraphContext);
}
