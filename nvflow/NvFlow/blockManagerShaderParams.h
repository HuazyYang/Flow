/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

NvFlowUint4 velFactor;
NvFlowUint4 velFactorBits;
NvFlowUint4 denFactor;
NvFlowUint4 denFactorBits;

float velocityWeight;
float smokeWeight;
float tempWeight;
float fuelWeight;

float velocityThreshold;
float smokeThreshold;
float tempThreshold;
float fuelThreshold;

float importanceThreshold;
float velocityScale;
float pad1;
float pad2;

NvFlowInt4 mapIdxReadOffset;
