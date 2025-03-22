#ifndef NVFLOW_PRESSURE_H
#define NVFLOW_PRESSURE_H
#include "NvFlowObjectImpl.h"

struct NvFlowContext;

namespace NvFlow {

struct SparseTextureFront;
struct SparseFadeField;

struct PressureParams {
    unsigned int iterations;
    bool legacyMode;
};

struct Pressure : NvFlowObject {
    virtual void execute(NvFlowContext *context, SparseTextureFront *velocity,
                         SparseTextureFront *pressureIn, SparseFadeField *fadeField,
                         const PressureParams *params) = 0;
};

struct PressureDesc {
    bool enableVTR;
};

Pressure *createPressure(NvFlowContext *context, const PressureDesc *desc);

}  // namespace NvFlow

#endif /* NVFLOW_PRESSURE_H */
