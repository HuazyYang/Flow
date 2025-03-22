#ifndef NVFLOW_SPARSEMAPPABLE_H
#define NVFLOW_SPARSEMAPPABLE_H
#include "Types.h"
#include "Object.h"
#include "NvFlowContextImpl.h"

namespace NvFlow {

struct SparseMapping;

struct SparseMappable {
    virtual void pushMapping(NvFlowContext *context, uint64_t fenceIndex,
                             SparseMapping *mapping) = 0;

    virtual void updateMapping(NvFlowContext *context, uint32_t maxUpdateTextures) = 0;

    virtual bool canCommitMapping(NvFlowContext *context, uint64_t fenceIndex) = 0;

    virtual NvFlowResult commitMapping(NvFlowContext *context, uint64_t fenceIndex) = 0;
};

}  // namespace NvFlow

#endif /* NVFLOW_SPARSEMAPPABLE_H */
