/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

#include "framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowInt4 maskOffset;
};

Texture3D<uint> maskSRV : register(t0);

RWTexture3D<uint> maskUAV : register(u0);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void sparseShiftCS(uint3 tidx : SV_DispatchThreadID)
{
	maskUAV[tidx] = maskSRV[int3(tidx) + maskOffset.xyz];
}