#ifndef OBJECT_H
#define OBJECT_H
#include "NvFlowObjectImpl.h"
#include "Allocable.h"
#include "DeferredRelease.h"
#include <atomic>

namespace NvFlow {

class Object : public NvFlowObject, public Allocable {
 public:
    uint32_t addRef() override;
    uint32_t release() override;

    DeferredRelease* getDeferredRelease() const;

    uint64_t getGPUBytesUsed() override;  // Default to zero

 protected:
    Object();
    Object(DeferredRelease* deferredRelease);
    virtual ~Object() = default;

 private:
    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    std::atomic<uint32_t> m_refCount = 1;
    DeferredRelease* m_deferredRelease;
};

}  // namespace NvFlow

#endif /* OBJECT_H */
