#ifndef NVFLOW_VOLUMERENDERUTILS_H
#define NVFLOW_VOLUMERENDERUTILS_H
#include "Types.h"
#include "NvFlowContextImpl.h"

namespace NvFlow {

struct VolumeRenderUtils {
    static void compute_depthInvTransform(NvFlowFloat4 *depthInvTransoform,
                                          const NvFlowFloat4 &rayForwardDirVirtual,
                                          const NvFlowViewport &offscreenViewport,
                                          const NvFlowFloat4x4 &projectionMatrixInv,
                                          const NvFlowFloat4 &vdim,
                                          const NvFlowFloat4x4 &modelViewMatrix);

    static void compute_rayVirtual(NvFlowFloat4 *rayOriginVirtual,
                                   NvFlowFloat4 *rayForwardDirVirtual,
                                   const NvFlowFloat4x4 &modelViewMatrixInv,
                                   const NvFlowFloat4 &vdim,
                                   const NvFlowFloat4x4 &projectionMatrix);

    static void compute_tlimit(float *tmin, float *tmax,
                               const NvFlowFloat4x4 &modelViewMatrix,
                               const NvFlowFloat4 &depthInvTransform);

    static float compute_tmax(float viewport_width,
                              const NvFlowFloat4x4 &projectionMatrixInv,
                              float multiResSamplingScale);

    static void compute_linearDepthTransform(NvFlowFloat4 *linearDepthTransform,
                                        const NvFlowFloat4x4 &projectionMatrix);
};

};  // namespace NvFlow

#endif /* NVFLOW_VOLUMERENDERUTILS_H */
