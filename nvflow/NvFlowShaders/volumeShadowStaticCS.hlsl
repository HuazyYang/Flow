/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 1

#include "frameworkHybrid.hlsli"

/// End NvFlow address translation utilities

cbuffer params : register(b0)
{
#include "../NvFlow/volumeShadowShaderParams.h"
};

cbuffer params : register(b1)
{
	uint shadowBlockListLen;
	uint padb0;
	uint padb1;
	uint padb2;
};

Buffer<uint> exportBlockList : register(t0);
Texture3D<uint> exportBlockTable : register(t1);
Texture3D<float4> exportData : register(t2);

Texture1D<float4> colorMapSRV_layer0 : register(t3);

Buffer<uint2> shadowBlockListSRV : register(t4);

Texture3D<uint> shadowBlockTable : register(t5);

RWTexture3D<float> depthUAV : register(u0);

VIRTUAL_TO_REAL_LINEAR(VirtualToRealExport, exportBlockTable, exportParams);

#define LAYER_NAME layer0
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME

float4 depthLinear4(float3 ndc)
{
	float a = linearDepthTransform.z * ndc.z + linearDepthTransform.w;
	float b = linearDepthTransform.y * ndc.z;

	return float4(a * ndc.xy, b, a);
}

float3 shadowCoord_to_vidxf(float3 shadowCoordf)
{
	float3 shadowNDC = float3(2.f, 2.f, 1.f) * (shadowCoordf * shadowVdimInv.xyz) + float3(-1.f, -1.f, 0.f);

	float4 shadowCoord4 = depthLinear4(shadowNDC);

	float4 vidxNorm = mul(shadowCoord4, shadowToVidxNorm);
	float3 vidxUVW = 0.5f * vidxNorm.xyz + 0.5f;
	float3 vidxf = exportParams.vdim.xyz * vidxUVW;

	return vidxf;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void volumeShadowStaticCS(uint3 tidx : SV_DispatchThreadID)
{
	uint shadowBlockListIdx = (tidx.x >> shadowBlockDimBits.x);
	int2 cellIdx2 = int2(
		tidx.x & (shadowBlockDim.x - 1),
		tidx.y
		);

	float2 cellIdx2f = shadowCellIdxInflate.xy * float2(cellIdx2);

	if (shadowBlockListIdx < shadowBlockListLen)
	{
		uint2 shadowBlockListVal = shadowBlockListSRV[shadowBlockListIdx];

		int3 beginCoord = tableVal_to_coord(shadowBlockListVal.x);
		int3 endCoord   = tableVal_to_coord(shadowBlockListVal.y);

		float2 coord2f = float2(beginCoord.xy << shadowBlockDimBits.xy) + cellIdx2f;
		float minkf = float(beginCoord.z << shadowBlockDimBits.z);
		float maxkf = (endCoord.z << shadowBlockDimBits.z) + shadowBlockDim.z;

		float3 vidxf      = shadowCoord_to_vidxf(float3(coord2f, minkf));
		float3 endVidxf   = shadowCoord_to_vidxf(float3(coord2f, maxkf));

		float3 vidxInc = (endVidxf - vidxf) / (float(maxkf - minkf) * shadowCellIdxInflateInv.z);

		float trans = 1.f;

		for (int k2 = beginCoord.z; k2 <= endCoord.z; k2++)
		{
			uint blockTableVal = shadowBlockTable[int3(beginCoord.xy, k2)];

			int3 rBlockIdx = tableVal_to_coord(blockTableVal);

			for (int k1 = 0u; k1 < int(shadowBlockDim.z); k1++)
			{
				//int k = (k2 << shadowBlockDimBits.z) + k1;
				//int3 shadowCoord = int3(coord2, k);
				int3 shadowCoord = (rBlockIdx << shadowBlockDimBits.xyz) | int3(cellIdx2, k1);

				float3 ridxExport = VirtualToRealExport(vidxf);
				float4 color = exportData.SampleLevel(borderSampler, exportParams.dimInv.xyz * ridxExport, 0);

				// compute color consistent with volumeRenderPS
				color = colorMap_layer0(color);

				color.rgb = max(0.f.xxx, color.rgb);
				color.a = saturate(color.a);

				color.a *= alphaScale.x;

				depthUAV[shadowCoord] = trans;

				if (k1 == int(shadowBlockDim.z - 1)) break;

				trans *= (1.f - color.a);

				vidxf += vidxInc;
			}
		}
	}
}