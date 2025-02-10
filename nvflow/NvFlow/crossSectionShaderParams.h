#ifndef CROSSSECTIONPARAMS_HLSLI
#define CROSSSECTIONPARAMS_HLSLI
/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NvFlowShaderLinearParams valueParams;
NvFlowFloat4 posScale;
NvFlowFloat4 uvScale;
NvFlowFloat4 crossSectionPosition;
NvFlowFloat4 crossSectionScale;
NvFlowUint crossSectionAxis;
NvFlowUint renderMode;
float alphaBias_layer0;
float intensityBias_layer0;
NvFlowFloat4 colorMapCompMask_layer0;
NvFlowFloat4 colorMapRange_layer0;
NvFlowFloat4 alphaCompMask_layer0;
NvFlowFloat4 intensityCompMask_layer0;
float intensityScale;
NvFlowUint pointFilter;
float velocityScale;
float vectorLengthScale;
NvFlowFloat4 lineColor;
NvFlowFloat4 backgroundColor;
NvFlowFloat4 pixelSize;
NvFlowFloat4 cellColor;

#endif /* CROSSSECTIONPARAMS_HLSLI */
