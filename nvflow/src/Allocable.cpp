#include "Allocable.h"
#include <malloc.h>
#include <exception>
#if NVFLOW__USE_MICROSOFT_VLD
#include <vld.h>
#endif

namespace NvFlow {

static void *(*g_FlowMalloc)(size_t) = malloc;

static void (*g_FlowFree)(void *) = free;

void *Allocable::operator new(size_t count) {
    void *p = malloc(count);
    if (!p) throw std::bad_alloc();
    return p;
}

void *Allocable::operator new[](size_t count) {
    void *p = malloc(count);
    if (!p) throw std::bad_alloc();
    return p;
}

void Allocable::operator delete(void *ptr) {
    free(ptr);
}

void Allocable::operator delete[](void *ptr) {
    free(ptr);
}

void *Allocable::allocate(size_t sz) {
    return malloc(sz);
}

void Allocable::deallocate(void *ptr) {
    free(ptr);
}

void FlowSetMallocFunc(void *(*malloc)(size_t)) {
    g_FlowMalloc = malloc;
}

void FlowSetFreeFunc(void (*free)(void *)) {
    g_FlowFree = free;
}

}  // namespace NvFlow