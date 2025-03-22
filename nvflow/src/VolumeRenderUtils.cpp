#include "VolumeRenderUtils.h"

namespace NvFlow {

void VolumeRenderUtils::compute_depthInvTransform(NvFlowFloat4* depthInvTransoform,
                                                  const NvFlowFloat4& rayForwardDirVirtual,
                                                  const NvFlowViewport& offscreenViewport,
                                                  const NvFlowFloat4x4& projectionMatrixInv,
                                                  const NvFlowFloat4& vdim,
                                                  const NvFlowFloat4x4& modelViewMatrix) {
    auto& viewport = offscreenViewport;
    float vpInvScale = 1.f / (viewport.maxDepth - viewport.minDepth);
    float vpInvOffset = -viewport.minDepth / (viewport.maxDepth - viewport.minDepth);
    float m22 = projectionMatrixInv.z.z;
    float m23 = projectionMatrixInv.z.w;
    float m32 = projectionMatrixInv.w.z;
    float m33 = projectionMatrixInv.w.w;

    float a22 = vpInvScale * m22;
    float a23 = vpInvScale * m23;
    float a32 = vpInvOffset * m22 + m32;
    float a33 = vpInvOffset * m23 + m33;

    NvFlowFloat4x4 virtualToGridNDC =
        matrixScaling(2.f / vdim.x, 2.f / vdim.y, 2.f / vdim.z);
    auto rayDirGridNDC = transform4(rayForwardDirVirtual, virtualToGridNDC);
    auto rayDirView = transform4(rayDirGridNDC, modelViewMatrix);
    float zscale = 1.f / rayDirView.z;

    a22 = a22 * zscale;
    a32 = a32 * zscale;

    *depthInvTransoform = make_float4(a22, a32, a23, a33);
}

void VolumeRenderUtils::compute_rayVirtual(NvFlowFloat4* rayOriginVirtual,
                                           NvFlowFloat4* rayForwardDirVirtual,
                                           const NvFlowFloat4x4& modelViewMatrixInv,
                                           const NvFlowFloat4& vdim,
                                           const NvFlowFloat4x4& projectionMatrix) {
    NvFlowFloat4 eyeScreen = make_float4(0.f, 0.f, 0.f, 1.f);
    NvFlowFloat4 eyeVirtualNDC = transform4(eyeScreen, modelViewMatrixInv);
    eyeVirtualNDC = eyeVirtualNDC / eyeVirtualNDC.w;

    NvFlowFloat4 eyeVirtualUVW = 0.5f * eyeVirtualNDC + 0.5f;
    NvFlowFloat4 eyeVirtual = eyeVirtualUVW * vdim;

    *rayOriginVirtual = eyeVirtual;

    NvFlowFloat4 eyeDirScreen = make_float4(0.f, 0.f, 1.f, 0.f);
    if (projectionMatrix.z.w < 0.f) {
        eyeDirScreen = make_float4(0.f, 0.f, -1.f, 0.f);
    }

    NvFlowFloat4 rayForwardDirVirtualNDC = transform4(eyeDirScreen, modelViewMatrixInv);
    NvFlowFloat4 rayForwardDirVirtualUVW = 0.5f * rayForwardDirVirtualNDC;
    *rayForwardDirVirtual = vdim * rayForwardDirVirtualUVW;

    const float stepSizeCell = 0.75f;
    *(NvFlowFloat3*)rayForwardDirVirtual =
        normalize(*(const NvFlowFloat3*)rayForwardDirVirtual);
    *rayForwardDirVirtual = stepSizeCell * *rayForwardDirVirtual;
}

void VolumeRenderUtils::compute_tlimit(float* tmin, float* tmax,
                                       const NvFlowFloat4x4& modelViewMatrix,
                                       const NvFlowFloat4& depthInvTransform) {
    NvFlowFloat4 pts[] = {
        {-1.f, -1.f, -1.f, 1.f}, {1.f, -1.f, -1.f, 1.f}, {-1.f, 1.f, -1.f, 1.f},
        {1.f, 1.f, -1.f, 1.f},   {-1.f, -1.f, 1.f, 1.f}, {1.f, -1.f, 1.f, 1.f},
        {-1.f, 1.f, 1.f, 1.f},   {1.f, 1.f, 1.f, 1.f},
    };

    NvFlowFloat4 postProj = transform4(pts[0], modelViewMatrix);
    float z = postProj.z / postProj.w;
    float t = (z * depthInvTransform.x + depthInvTransform.y) /
              (z * depthInvTransform.z + depthInvTransform.w);
    *tmin = *tmax = t;

    for (int i = 1; i < countof(pts); ++i) {
        postProj = transform4(pts[0], modelViewMatrix);
        z = postProj.z / postProj.w;
        float t = (z * depthInvTransform.x + depthInvTransform.y) /
                  (z * depthInvTransform.z + depthInvTransform.w);
        *tmin = min(*tmin, t);
        *tmax = max(*tmax, t);
    }
}

float VolumeRenderUtils::compute_tmax(float viewport_width,
                                      const NvFlowFloat4x4& projectionMatrixInv,
                                      float multiResSamplingScale) {
    float pixelSize = 2.f / viewport_width;
    NvFlowFloat4 ptA = make_float4(0.f, 0.f, 0.f, 1.f);
    NvFlowFloat4 ptB = make_float4(pixelSize, 0.f, 0.f, 1.f);
    NvFlowFloat4 ptA1 = transform4(ptA, projectionMatrixInv);
    NvFlowFloat4 ptB1 = transform4(ptB, projectionMatrixInv);
    ptA1 = ptA1 / ptA1.w;
    ptB1 = ptB1 / ptB1.w;
    float lenA = vector3Length(ptA1);
    float lenB = vector3Length(ptB1 - ptA1);
    float k = 1.f / multiResSamplingScale;
    float tmax = k * lenA / lenB;
    return tmax;
}

void VolumeRenderUtils::compute_linearDepthTransform(
    NvFlowFloat4* linearDepthTransform, const NvFlowFloat4x4& projectionMatrix) {
    float a = (projectionMatrix.z.z * projectionMatrix.w.w -
               projectionMatrix.z.w * projectionMatrix.w.z) /
              (projectionMatrix.z.z - projectionMatrix.z.w);
    float b = projectionMatrix.z.w / projectionMatrix.z.z;
    float c = (projectionMatrix.z.z - projectionMatrix.z.w) / projectionMatrix.z.z;

    linearDepthTransform->x = 1.f / a;
    linearDepthTransform->y = a;
    linearDepthTransform->z = a * b;
    linearDepthTransform->w = a * c;
}

}  // namespace NvFlow