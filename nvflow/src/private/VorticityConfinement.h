#ifndef NVFLOW_VORTICITYCONFINEMENT_H
#define NVFLOW_VORTICITYCONFINEMENT_H
#include "NvFlowObjectImpl.h"

struct NvFlowContext;

namespace NvFlow {

struct SparseTextureFront;

struct VorticityConfinementPerLayerParams {
    float forceScale;
    float velocityMask;
    float temperatureMask;
    float smokeMask;
    float fuelMask;
    float constantMask;
};

struct VorticityConfinementParams {
    float deltaTime;
    void (*getPerLayer)(VorticityConfinementPerLayerParams *, void *, unsigned int);
    void *userdata;
};

struct VorticityConfinement : NvFlowObject {
    virtual void execute(NvFlowContext *context, SparseTextureFront *velocity,
                         SparseTextureFront *coarseDensity,
                         const VorticityConfinementParams *params) = 0;
};

struct VorticityConfinementDesc {
    bool enableVTR;
    SparseTextureFront *velocityField;
};

VorticityConfinement *createVorticityConfinement(NvFlowContext *context,
                                                 const VorticityConfinementDesc *desc);

}  // namespace NvFlow

#endif /* NVFLOW_VORTICITYCONFINEMENT_H */
