#ifndef HEREAFTER_VOLUMERENDERSHADERPARAMS_H
#define HEREAFTER_VOLUMERENDERSHADERPARAMS_H

/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

NvFlowFloat4x4 modelViewProj;
NvFlowFloat4 viewportInvScale;
NvFlowFloat4 viewportInvOffset;

NvFlowShaderLinearParams valueParams;
NvFlowUint4 vGridDim;
NvFlowFloat4 vGridDimInv;

NvFlowFloat4 rayOriginVirtual;
NvFlowFloat4 rayForwardDirVirtual;

NvFlowUint4 blockListStartCount;
NvFlowUint4 renderMode;

float eyePad;
float eyeFade;
float eyeFadeInv;
float eyeFadeOffset;

NvFlowFloat4 depthUVTransform;
NvFlowFloat4 depthMaxInvTransform;
NvFlowFloat4 depthMinInvTransform;

float alphaScale_layer0;
float alphaScale_layer1;
float alphaScale_layer2;
float alphaScale_layer3;

float additiveFactor_layer0;
float additiveFactor_layer1;
float additiveFactor_layer2;
float additiveFactor_layer3;

float alphaBias_layer0;
float alphaBias_layer1;
float alphaBias_layer2;
float alphaBias_layer3;

float intensityBias_layer0;
float intensityBias_layer1;
float intensityBias_layer2;
float intensityBias_layer3;

NvFlowFloat4 colorMapCompMask_layer0;
NvFlowFloat4 colorMapRange_layer0;
NvFlowFloat4 alphaCompMask_layer0;
NvFlowFloat4 intensityCompMask_layer0;

NvFlowFloat4 colorMapCompMask_layer1;
NvFlowFloat4 colorMapRange_layer1;
NvFlowFloat4 alphaCompMask_layer1;
NvFlowFloat4 intensityCompMask_layer1;

NvFlowFloat4 colorMapCompMask_layer2;
NvFlowFloat4 colorMapRange_layer2;
NvFlowFloat4 alphaCompMask_layer2;
NvFlowFloat4 intensityCompMask_layer2;

NvFlowFloat4 colorMapCompMask_layer3;
NvFlowFloat4 colorMapRange_layer3;
NvFlowFloat4 alphaCompMask_layer3;
NvFlowFloat4 intensityCompMask_layer3;

#endif /* HEREAFTER_VOLUMERENDERSHADERPARAMS_H */
