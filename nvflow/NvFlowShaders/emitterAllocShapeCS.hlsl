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

#ifndef EMIT_ALLOC_SHAPE_SDF
#define EMIT_ALLOC_SHAPE_SDF 1
#endif

#include "framework.hlsli"

#define EMIT_ALLOC_SHAPE_DEFINES 1

#include "../NvFlow/emitterAllocShapeParams.h"

cbuffer params : register(b0)
{
	EmitAllocShapeShaderParams params;
};

Buffer<float4> paramsSRV : register(t0);
Texture3D<float> sdf_SRV : register(t1);

RWTexture3D<uint> maskUAV : register(u0);

groupshared NvFlowFloat4 sparams[EMIT_ALLOC_SHAPE_BATCH_SIZE * EMIT_ALLOC_SHAPE_SIZE];
groupshared NvFlowFloat4 sshape[EMIT_ALLOC_SHAPE_CACHE_SIZE];

float sdfDist(float3 emitterLocal, uint shapeIdx)
{
	float3 uvw = 0.5f * emitterLocal + 0.5f;
	return sdf_SRV.SampleLevel(clampSampler, uvw, 0);
}

float sphereDist(float3 emitterLocal, uint shapeIdx)
{
	float d = length(emitterLocal);
	float v = d - sshape[shapeIdx].x;
	return v;
}

float boxDist(float3 emitterLocal, uint shapeIdx)
{
	float3 dr = abs(emitterLocal) - sshape[shapeIdx].xyz;
	float v = max(dr.x, max(dr.y, dr.z));
	return v;
}

float capsuleDist(float3 emitterLocal, uint shapeIdx)
{
	emitterLocal.x = max(0.f, abs(emitterLocal.x) - 0.5f * sshape[shapeIdx].y);
	float d = length(emitterLocal);
	float v = d - sshape[shapeIdx].x;
	return v;
}

float convexDist(float3 emitterLocal, uint shapeOffset, uint shapeCountIn)
{
	NvFlowUint shapeMax = min(EMIT_ALLOC_SHAPE_CACHE_SIZE, shapeOffset + shapeCountIn);
	NvFlowUint shapeCount = shapeMax - shapeOffset;
	float dist = -1.f;
	if (0u < shapeCount)
	{
		float4 plane = sshape[shapeOffset];
		dist = dot(emitterLocal, plane.xyz) - plane.w;
	}
	for (NvFlowUint i = 1u; i < shapeCount; i++)
	{
		float4 plane = sshape[shapeOffset + i];
		float v = dot(emitterLocal, plane.xyz) - plane.w;
		dist = max(dist, v);
	}
	return dist;
}

float sampleDist(float3 emitterLocal, uint paramIdx, uint poff)
{
	float dist = 1.f;
	uint type = emit_alloc_shape_shapeType(sparams, poff);

	uint shapeIdx = emit_alloc_shape_shapeRangeOffset(sparams, poff);

#if EMIT_ALLOC_SHAPE_SDF
	if (type == 0) dist = sdfDist(emitterLocal, shapeIdx);
#endif
	if (type == 1) dist = sphereDist(emitterLocal, shapeIdx);
	if (type == 2) dist = boxDist(emitterLocal, shapeIdx);
	if (type == 3) dist = capsuleDist(emitterLocal, shapeIdx);
	if (type == 4) dist = convexDist(emitterLocal, shapeIdx, emit_alloc_shape_shapeRangeSize(sparams, poff));

	dist *= emit_alloc_shape_shapeDistScale(sparams, poff);

	return dist;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void emitterAllocShapeCS(uint3 tidx : SV_DispatchThreadID, uint3 threadIdx : SV_GroupThreadID)
{
	uint3 blockIdx = tidx + params.blockIdxOffset.xyz;
	uint threadIdx1D =
		(threadIdx.x) |
		(threadIdx.y << 3) |
		(threadIdx.z << 6);

	float4 grid_ndc = float4(2.f * (params.gridDimInv.xyz * (float3(blockIdx)+0.5f.xxx)) - 1.f.xxx, 1.f);

	bool shouldAlloc = false;

	if (threadIdx1D < params.numShapes.x * EMIT_ALLOC_SHAPE_SIZE)
	{
		sparams[threadIdx1D] = paramsSRV[threadIdx1D];
	}

	if (threadIdx1D < params.numShapes.y)
	{
		sshape[threadIdx1D] = paramsSRV[threadIdx1D + EMIT_ALLOC_SHAPE_BATCH_SIZE * EMIT_ALLOC_SHAPE_SIZE];
	}

	GroupMemoryBarrierWithGroupSync();

	for (NvFlowUint paramIdx = 0u; paramIdx < params.numShapes.x; paramIdx++)
	{
		NvFlowUint poff = EMIT_ALLOC_SHAPE_SIZE * paramIdx;

		// apply
		{
			uint3 minIdx = emit_alloc_shape_minAllocMaskIdx(sparams, poff);
			uint3 maxIdx = emit_alloc_shape_maxAllocMaskIdx(sparams, poff);
			if (all(blockIdx >= minIdx) && all(blockIdx < maxIdx))
			{

				float4 emitterLocal = mul(grid_ndc, emit_alloc_shape_gridToEmitter(sparams, poff));

				float dist = sampleDist(emitterLocal.xyz, paramIdx, poff);

				bool materialMatch = (emit_alloc_shape_materialIdx(sparams, poff) == params.materialIdx) ||
					(emit_alloc_shape_materialIdx(sparams, poff).x == 0u);

				if (dist >= emit_alloc_shape_minActiveDist(sparams, poff) &&
					dist <= emit_alloc_shape_maxActiveDist(sparams, poff) &&
					materialMatch )
				{
					shouldAlloc = true;
				}
			}
		}
	}

	if (shouldAlloc)
	{
		maskUAV[blockIdx] = 1u;
	}
}