#ifndef HEREAFTER_COMPOSITESHADERPARAMS_H
#define HEREAFTER_COMPOSITESHADERPARAMS_H

NvFlowFloat4 uvScale;

NvFlowFloat4 scene_uvScale;
NvFlowFloat4 uvStepSize;
NvFlowFloat4 depthInvTransform;
NvFlowFloat4 tlimitRange;

float warpLeft;
float warpRight;
float warpUp;
float warpDown;

float2 leftX;
float2 rightX;
float2 upY;
float2 downY;

NvFlowFloat4 screenPercent;

NvFlowUint compositeMode;
float depthAlphaThreshold;
float depthIntensityThreshold;
float pad1;

#endif /* HEREAFTER_COMPOSITESHADERPARAMS_H */