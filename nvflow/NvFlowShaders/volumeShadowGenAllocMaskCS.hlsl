/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define THREAD_DIM_X 128
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

#include "frameworkHybrid.hlsli"

/// End NvFlow address translation utilities

cbuffer params : register(b0)
{
	NvFlowFloat4x4 vidxNormToShadow;
	NvFlowFloat4x4 shadowToVidxNorm;
	NvFlowFloat4 linearDepthTransform;

	NvFlowUint4 shadowVolumeDim;
	NvFlowFloat4 shadowVolumeDimInv;
	NvFlowUint4 shadowBlockDimBits;

	NvFlowUint4 numBlocks;
	NvFlowFloat4 gridDimInv;

	NvFlowShaderLinearParams exportParams;
};

Buffer<uint> exportBlockList : register(t0);

RWTexture3D<float> allocMaskUAV : register(u0);

int3 vBlockIdxf_to_shadowBlockidx(float3 vBlockIdxf)
{
	float3 vBlockIdxNorm = 2.f.xxx * (gridDimInv.xyz * vBlockIdxf) - 1.f.xxx;

	float4 shadowNDC = mul(float4(vBlockIdxNorm.xyz, 1.f), vidxNormToShadow);
	float linearZ = linearDepthTransform.x * shadowNDC.z;

	shadowNDC.xyz /= shadowNDC.w;
	shadowNDC.w = 1.f;

	float3 shadowUVW = float3(0.5f.xx * shadowNDC.xy + 0.5f.xx, linearZ);

	int3 shadowIdx = int3(floor(shadowVolumeDim.xyz * shadowUVW));

	int3 shadowBlockIdx = shadowIdx >> shadowBlockDimBits.xyz;
	
	return shadowBlockIdx;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void volumeShadowGenAllocMaskCS(uint3 tidx : SV_DispatchThreadID)
{
	if (tidx.x < numBlocks.x)
	{
		uint blockListVal = exportBlockList[tidx.x];
		int3 vBlockIdx = tableVal_to_coord(blockListVal);

		//float3 vBlockIdxf = float3(vBlockIdx)+0.5f.xxx;
		//int3 shadowBlockIdx = vBlockIdxf_to_shadowBlockidx(vBlockIdxf);

		//allocMaskUAV[shadowBlockIdx] = 1.f;

		int3 minIdx = int3(shadowVolumeDim.xyz >> shadowBlockDimBits.xyz);
		int3 maxIdx = int3(0, 0, 0);
		for (int k = 0; k < 2; k++)
		{
			for (int j = 0; j < 2; j++)
			{
				for (int i = 0; i < 2; i++)
				{
					float3 vBlockIdxf = float3(vBlockIdx) + float3(i, j, k);

					int3 shadowBlockIdx = vBlockIdxf_to_shadowBlockidx(vBlockIdxf);

					minIdx = min(minIdx, shadowBlockIdx);
					maxIdx = max(maxIdx, shadowBlockIdx);
				}
			}
		}

		if (all(minIdx == maxIdx))
		{
			allocMaskUAV[minIdx] = 1.f;
		}
		else
		{
			for (int k = minIdx.z; k <= maxIdx.z; k++)
			{
				for (int j = minIdx.y; j <= maxIdx.y; j++)
				{
					for (int i = minIdx.x; i <= maxIdx.x; i++)
					{
						allocMaskUAV[int3(i,j,k)] = 1.f;
					}
				}
			}
		}
	}
}