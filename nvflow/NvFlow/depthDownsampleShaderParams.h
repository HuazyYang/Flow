#ifndef HEREAFTER_DEPTHDOWNSAMPLESHADERPARAMS_H
#define HEREAFTER_DEPTHDOWNSAMPLESHADERPARAMS_H

NvFlowFloat4 uvScale;

NvFlowFloat4 viewportInvScale;
NvFlowFloat4 viewportInvOffset;
NvFlowFloat4 depth_uvScale;
NvFlowUint4 depth_reverseZ;

float warpLeft;
float warpRight;
float warpUp;
float warpDown;

float2 leftX;
float2 rightX;
float2 upY;
float2 downY;

#endif /* HEREAFTER_DEPTHDOWNSAMPLESHADERPARAMS_H */
