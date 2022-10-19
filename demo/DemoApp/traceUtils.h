#pragma once

#include <type_traits>

// NvFlow(((?!Defaults)\w)+)\((.*)\)
// TRACE(NvFlow$1($3))

extern struct AppGraphCtx* g_pAppGraphContext;

extern void _TraceCallBegin(const char *call);
extern void _TraceCallEnd();

struct TraceCallGuard {
    TraceCallGuard(const char* label) {
        _TraceCallBegin(label);
    }

    ~TraceCallGuard() {
        _TraceCallEnd();
    }
};

inline void
TraceCallPutLabel(const char *label) {
  TraceCallGuard _guard(label);
}

#define TRACE(call)                                                                                                    \
  ([&] {                                                                                                               \
    TraceCallGuard _gurad(#call);                                                                                      \
    return (call);                                                                                                     \
  }())