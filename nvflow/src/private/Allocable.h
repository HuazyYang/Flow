#ifndef ALLOCABLE_H
#define ALLOCABLE_H
#include <cstdlib>

namespace NvFlow {

void FlowSetMallocFunc(void* (*malloc)(size_t));

void FlowSetFreeFunc(void (*free)(void*));

class Allocable {
 public:
    void* operator new(std::size_t count);
    void* operator new[](std::size_t count);

    void operator delete(void* ptr);
    void operator delete[](void* ptr);

    static void* allocate(std::size_t sz);
    static void deallocate(void* ptr);

 protected:
    Allocable() = default;
    ~Allocable() = default;
};

}  // namespace NvFlow

#endif /* ALLOCABLE_H */
