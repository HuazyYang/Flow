#ifndef ALLOCABLE_H
#define ALLOCABLE_H
#include <stddef.h>
#if NVFLOW__USE_MICROSOFT_VLD
#include <vld.h>
#endif

namespace NvFlow {

void FlowSetMallocFunc(void* (*malloc)(size_t));

void FlowSetFreeFunc(void (*free)(void*));

class Allocable {
 public:
    void* operator new(size_t count);
    void* operator new[](size_t count);

    void operator delete(void* ptr);
    void operator delete[](void* ptr);

    static void* allocate(size_t sz);
    static void deallocate(void* ptr);

 protected:
    Allocable() = default;
    ~Allocable() = default;
};

}  // namespace NvFlow

#endif /* ALLOCABLE_H */
