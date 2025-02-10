#ifndef HEREAFTER_VOLUMESHADOWSHADERPARAMS_H
#define HEREAFTER_VOLUMESHADOWSHADERPARAMS_H

/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

NvFlowFloat4x4 vidxNormToShadow;
NvFlowFloat4x4 shadowToVidxNorm;
NvFlowFloat4 linearDepthTransform;

NvFlowUint4 shadowVdim;
NvFlowFloat4 shadowVdimInv;
NvFlowUint4 shadowRdim;
NvFlowFloat4 shadowRdimInv;

NvFlowUint4 shadowBlockDim;
NvFlowUint4 shadowBlockDimBits;
NvFlowFloat4 shadowBlockDimInv;
NvFlowFloat4 shadowCellIdxInflate;
NvFlowFloat4 shadowCellIdxInflateInv;
NvFlowFloat4 shadowCellIdxOffset;

float alphaScale;
float minIntensity;
float shadowBlendBias;
float pad3;
NvFlowUint4 renderMode;
float alphaBias_layer0;
float intensityBias_layer0;
float pad4;
float pad5;
NvFlowFloat4 colorMapCompMask_layer0;
NvFlowFloat4 colorMapRange_layer0;
NvFlowFloat4 alphaCompMask_layer0;
NvFlowFloat4 intensityCompMask_layer0;

NvFlowFloat4 shadowBlendCompMask;

NvFlowShaderLinearParams exportParams;
NvFlowShaderLinearParams importParams;

#endif /* HEREAFTER_VOLUMESHADOWSHADERPARAMS_H */
