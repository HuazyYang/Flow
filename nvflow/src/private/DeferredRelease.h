#ifndef DEFERREDRELEASE_H
#define DEFERREDRELEASE_H
#include "NvFlowObjectImpl.h"
#include "Allocable.h"

namespace NvFlow {

struct DeferredRelease : NvFlowObject, Allocable {
    virtual void registerObject(NvFlowObject *object) = 0;

    virtual void pushForRelease(NvFlowObject *object) = 0;
};

}  // namespace NvFlow

#endif /* DEFERREDRELEASE_H */
