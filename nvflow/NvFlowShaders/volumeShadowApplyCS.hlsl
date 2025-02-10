/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef THREAD_DIM_X

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

#define SIMULATE_ONCE 1

#endif

#include "frameworkHybrid.hlsli"

/// End NvFlow address translation utilities

cbuffer params : register(b0)
{
#include "../NvFlow/volumeShadowShaderParams.h"
};

Buffer<uint> exportBlockList : register(t0);
Texture3D<uint> exportBlockTable : register(t1);
Texture3D<float4> exportData : register(t2);

Buffer<uint> importBlockList : register(t3);
Texture3D<uint> importBlockTable : register(t4);
RWTexture3D<float4> importDataRW : register(u0);

Texture3D<float> depthSRV : register(t5);

Texture3D<uint> shadowBlockTable : register(t6);

//DISPATCH_ID_TO_VIRTUAL(importBlockList, importParams);

DISPATCH_ID_TO_VIRTUAL_AND_REAL(importBlockList, importBlockTable, importParams);

VIRTUAL_TO_REAL_LINEAR(VirtualToRealExport, exportBlockTable, exportParams);
//VIRTUAL_TO_REAL_IMPORT(VirtualToRealImport, importBlockTable, importParams);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void volumeShadowApplyCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, vidx, ridx);

	float3 vidxf = float3(vidx)+0.5f.xxx;

	float4 vidxNorm = float4(2.f * vidxf * exportParams.vdimInv.xyz - 1.f, 1.f);

	float4 shadowNDC = mul(vidxNorm, vidxNormToShadow);
	float linearZ = linearDepthTransform.x * shadowNDC.z;

	shadowNDC.xyz /= shadowNDC.w;
	shadowNDC.w = 1.f;

	float3 shadowUVW = float3(0.5f.xx * shadowNDC.xy + 0.5f.xx, linearZ);

	float shadowVal;
	{
		float3 shadowVidxf = float3(shadowVdim.xyz) * shadowUVW;

		float3 vBlockIdxf = shadowBlockDimInv.xyz * shadowVidxf;
		int3 vBlockIdx = int3(floor(vBlockIdxf));
		int3 rBlockIdx = tableVal_to_coord(shadowBlockTable[vBlockIdx]);
		float3 rBlockIdxf = float3(rBlockIdx);
		float3 ridx = float3(rBlockIdx << shadowBlockDimBits.xyz) + float3(shadowBlockDim.xyz - uint3(1,1,1)) * (vBlockIdxf - float3(vBlockIdx)) + shadowCellIdxOffset.xyz;

		float3 ridxNorm = shadowRdimInv.xyz * ridx;

		shadowVal = depthSRV.SampleLevel(borderSampler, ridxNorm, 0);
	}

	//float shadowVal = depthSRV.SampleLevel(borderSampler, shadowUVW, 0);
	float intensity = shadowVal;

	intensity = max(intensity, minIntensity);

	float3 ridxExport = VirtualToRealExport(vidxf);
	float4 value = exportData.SampleLevel(borderSampler, exportParams.dimInv.xyz * ridxExport, 0);

	float blendWeight = saturate(dot(shadowBlendCompMask, value) + shadowBlendBias);

	float4 color = float4(value.rg, intensity, value.a);

	color.b = intensity * (blendWeight) + 1.f * (1.f - blendWeight);

	//int3 ridxImport = VirtualToRealImport(vidx);
	importDataRW[ridx] = color;
}