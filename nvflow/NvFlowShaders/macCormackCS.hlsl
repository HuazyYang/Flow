/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// set default defines
#ifndef ENABLE_VELOCITY
#define ENABLE_VELOCITY 1
#endif
#ifndef ENABLE_DENSITY
#define ENABLE_DENSITY 1
#endif
#ifndef ENABLE_EMIT
#define ENABLE_EMIT 1
#endif

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

#define SIMULATE_ONCE 1

#include "frameworkHybrid.hlsli"

#include "combustion.hlsli"

#define EMIT_DEFINES 1

#include "../NvFlow/emitterShaderParams.h"

cbuffer params : register(b0)
{
	NvFlowFloat4x4 gridToWorld;

	NvFlowShaderPointParams outputParams;

	NvFlowShaderLinearParams valueParams;
	NvFlowShaderLinearParams predictParams;
	NvFlowShaderLinearParams velocityParams;
	NvFlowShaderLinearParams densityParams;

	NvFlowFloat4 vidxOldOffsetf;
	NvFlowFloat4 vidxNormOldOffsetf;
	NvFlowFloat4 deltaTime;
	NvFlowFloat4 blendFactor;
	NvFlowFloat4 blendThreshold;

	NvFlowFloat4 damping;
	NvFlowFloat4 linearFade;

	NvFlowUint4 combustMode;
	CombustionParams combust;

	NvFlowUint4 emitterCount;
	NvFlowUint materialIdx;
	NvFlowUint matPad0;
	NvFlowUint matPad1;
	NvFlowUint matPad2;
};

#include "combustion.hlsli"

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The target to write out results to
RWTexture3D<float4> valueOutUAV : register(u0);
Texture3D<uint> valueOutBlockTable : register(t1);

//! The value to advect
Texture3D<float4> valueSRV : register(t2);
Texture3D<uint> valueBlockTable : register(t3);

//! The semi-Lagrangian prediction
Texture3D<float4> predictSRV : register(t4);
Texture3D<uint> predictBlockTable : register(t5);

//! The velocity field for advection
Texture3D<float4> velocitySRV : register(t6);
Texture3D<uint> velocityBlockTable : register(t7);

//! The density field for combustion use
Texture3D<float4> densitySRV : register(t8);
Texture3D<uint> densityBlockTable : register(t9);

//! A fade field for softer boundary conditions
Texture3D<float> fadeFieldSRV : register(t10);

//! A coarse density representation
Texture3D<float4> coarseDensitySRV : register(t11);
RWTexture3D<float4> coarseDensityUAV : register(u1);

//! Emitter parameters
Buffer<float4> emitterParametersSRV : register(t12);
//! SDF field, used in emitterFunctions.hlsli
Texture3D<float> sdf_SRV : register(t13);

#if ENABLE_EMIT
groupshared NvFlowFloat4 sdata[EMIT_DATA_SIZE];
groupshared NvFlowFloat4 sshape[EMIT_SHAPE_CACHE_SIZE*EMIT_SHAPE_SIZE];
groupshared NvFlowUint4 soffsets[512u];
groupshared NvFlowUint satomic;

#include "emitterFunctions.hlsli"
#endif

SAMPLE_LINEAR_3D(sampleValue, float4, valueSRV, valueBlockTable, valueParams);
SAMPLE_LINEAR_3D(samplePredict, float4, predictSRV, predictBlockTable, predictParams);
SAMPLE_LINEAR_3D_NORM(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);
SAMPLE_LINEAR_3D_NORM(sampleDensity, float4, densitySRV, densityBlockTable, densityParams)
SAMPLE_LINEAR_3D_NORM(sampleCoarseDensity, float4, coarseDensitySRV, velocityBlockTable, velocityParams);

VIRTUAL_TO_REAL_LINEAR(VirtualToReal, valueBlockTable, valueParams);

OUTPUT_3D(outputCoarseDensity, float4, coarseDensityUAV, valueOutBlockTable, outputParams);

//OUTPUT_3D(outputValue, float4, valueOutUAV, valueOutBlockTable, outputParams);
//DISPATCH_ID_TO_VIRTUAL(blockListSRV, outputParams);
DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, valueOutBlockTable, outputParams);

void minmax4(inout float4 minVal, inout float4 maxVal, float4 val)
{
	minVal = min(minVal, val);
	maxVal = max(maxVal, val);
}

float4 fade(float4 value, float4 linearRate, float4 expRate)
{
	float4 expCorrection = value * (1.f - expRate);
	float4 linCorrection = deltaTime.w * linearRate;
	float4 correction = sign(value) * min(abs(value), max(abs(expCorrection), abs(linCorrection)));
	value -= correction;
	return value;
}

#if ENABLE_EMIT

float4 emitVelocity(float4 value, uint threadIdx1D, float4 grid_ndc, int3 vidx)
{
	for (uint cullID = 0u; cullID < emitterCount.z; cullID++)
	{
		if (threadIdx1D == 0u)
		{
			satomic = 0u;
		}

		GroupMemoryBarrierWithGroupSync();

		uint emitterIdx = 512 * cullID + threadIdx1D;
		if (emitterIdx < emitterCount.x)
		{
			uint4 headerData = asuint(emitterParametersSRV[emitterIdx]);
			uint3 boxmin = headerData.xyz & 0xFFFF;
			uint3 boxmax = headerData.xyz >> 16u;
			uint3 tmin = vidx & 0xFFFFFFF8;
			uint3 tmax = tmin + 8u;

			bool isOverlap = !(
				tmin.x > boxmax.x || boxmin.x > tmax.x ||
				tmin.y > boxmax.y || boxmin.y > tmax.y ||
				tmin.z > boxmax.z || boxmin.z > tmax.z
				);

			if (isOverlap)
			{
				uint allocIdx = 0u;
				InterlockedAdd(satomic, 1u, allocIdx);

				// fetch offset data
				uint4 headerData2 = asuint(emitterParametersSRV[emitterIdx + emitterCount.y]);

				soffsets[allocIdx] = headerData2;
			}
		}

		GroupMemoryBarrierWithGroupSync();

		uint numOverlaps = satomic;
		for (uint idx = 0; idx < numOverlaps; idx++)
		{
			uint4 emitOffset = soffsets[idx];

			if (threadIdx1D < EMIT_DATA_SIZE)
			{
				sdata[threadIdx1D] = emitterParametersSRV[threadIdx1D + emitOffset.x];
			}
			if (threadIdx1D < (emitOffset.w - emitOffset.z))
			{
				sshape[threadIdx1D] = emitterParametersSRV[threadIdx1D + emitOffset.z];
			}

			GroupMemoryBarrierWithGroupSync();

			value = emitVelocityFunction(value, grid_ndc);

			GroupMemoryBarrierWithGroupSync();
		}

		GroupMemoryBarrierWithGroupSync();
	}

	return value;
}

float4 emitDensity(float4 value, uint threadIdx1D, float4 grid_ndc, float4 coarseDensityVal, int3 vidx)
{
	for (uint cullID = 0u; cullID < emitterCount.z; cullID++)
	{
		if (threadIdx1D == 0u)
		{
			satomic = 0u;
		}

		GroupMemoryBarrierWithGroupSync();

		uint emitterIdx = 512 * cullID + threadIdx1D;
		if (emitterIdx < emitterCount.x)
		{
			uint4 headerData = asuint(emitterParametersSRV[emitterIdx]);
			uint3 boxmin = headerData.xyz & 0xFFFF;
			uint3 boxmax = headerData.xyz >> 16u;
			uint3 tmin = vidx & 0xFFFFFFF8;
			uint3 tmax = tmin + 8u;

			bool isOverlap = !(
				tmin.x > boxmax.x || boxmin.x > tmax.x ||
				tmin.y > boxmax.y || boxmin.y > tmax.y ||
				tmin.z > boxmax.z || boxmin.z > tmax.z
				);

			if (isOverlap)
			{
				uint allocIdx = 0u;
				InterlockedAdd(satomic, 1u, allocIdx);

				// fetch offset data
				uint4 headerData2 = asuint(emitterParametersSRV[emitterIdx + emitterCount.y]);

				soffsets[allocIdx] = headerData2;
			}
		}

		GroupMemoryBarrierWithGroupSync();

		uint numOverlaps = satomic;
		for (uint idx = 0; idx < numOverlaps; idx++)
		{
			uint4 emitOffset = soffsets[idx];

			if (threadIdx1D < EMIT_DATA_SIZE)
			{
				sdata[threadIdx1D] = emitterParametersSRV[threadIdx1D + emitOffset.x];
			}
			if (threadIdx1D < (emitOffset.w - emitOffset.z))
			{
				sshape[threadIdx1D] = emitterParametersSRV[threadIdx1D + emitOffset.z];
			}

			GroupMemoryBarrierWithGroupSync();

			float coarseTemp = coarseDensityVal.x;

			value = emitDensityFunction(value, coarseTemp, grid_ndc);

			GroupMemoryBarrierWithGroupSync();
		}

		GroupMemoryBarrierWithGroupSync();
	}

	return value;
}

#endif

float3 compute_vidxOldf(float3 vidxNewf)
{
	return vidxNewf + vidxOldOffsetf.xyz;
}

float3 compute_vidxNormOld(float3 vidxNormNew)
{
	return vidxNormNew + vidxNormOldOffsetf.xyz;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void macCormackCS(uint3 tidx : SV_DispatchThreadID, uint3 threadIdx : SV_GroupThreadID)
{
	int3 vidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, vidx, ridx);

	float3 vidxf = float3(vidx)+0.5f.xxx;
	float3 vidxNorm = valueParams.vdimInv.xyz * vidxf;

	// lookup semi-Lagrangian result for this cell
	float4 predict = samplePredict(vidxf);

	bool shouldCorrect = any(abs(predict) >= blendThreshold);
	float4 value = predict;
	if (shouldCorrect)
	{
		// get this cell's velocity
		float4 vel = sampleVelocity(compute_vidxNormOld(vidxNorm));
		// trace backwards in time
		float3 predictFetchIdxf = vidxf + deltaTime.xyz * vel.xyz;
		// advect semi-Lagrangian result
		float4 predictRev = samplePredict(predictFetchIdxf);

		// sample last frame's value for this cell
		float4 valueOld = sampleValue(compute_vidxOldf(vidxf));

		value = predict + 0.5f * blendFactor * (valueOld - predictRev);

		// add clamping for stability
		{
			float3 clampFetchIdxf = compute_vidxOldf(vidxf) - deltaTime.xyz * vel.xyz;
			float3 ridxf = VirtualToReal(clampFetchIdxf);

			int3 c = int3(floor(ridxf - float3(0.5f, 0.5f, 0.5f)));

			float4 minVal = valueSRV[c + int3(0, 0, 0)];
			float4 maxVal = minVal;
			minmax4(minVal, maxVal, valueSRV[c + int3(1, 0, 0)]);
			minmax4(minVal, maxVal, valueSRV[c + int3(0, 1, 0)]);
			minmax4(minVal, maxVal, valueSRV[c + int3(1, 1, 0)]);
			minmax4(minVal, maxVal, valueSRV[c + int3(0, 0, 1)]);
			minmax4(minVal, maxVal, valueSRV[c + int3(1, 0, 1)]);
			minmax4(minVal, maxVal, valueSRV[c + int3(0, 1, 1)]);
			minmax4(minVal, maxVal, valueSRV[c + int3(1, 1, 1)]);

			value = clamp(value, minVal, maxVal);
		}
	}

	uint threadIdx1D =
		(threadIdx.x) |
		(threadIdx.y << 3) |
		(threadIdx.z << 6);

	float4 grid_ndc = 2.f * float4(vidxNorm, 1.f) - 1.f;

#if ENABLE_DENSITY
	if (combustMode.x == 1)
	{
		float4 coarseDensity = sampleCoarseDensity(compute_vidxNormOld(vidxNorm));
		// density mode
		value = combustSimulate(value, coarseDensity);

#if ENABLE_EMIT
		// density emitter
		value = emitDensity(value, threadIdx1D, grid_ndc, coarseDensity, vidx);
#endif
	}
#endif
#if ENABLE_VELOCITY
	if (combustMode.x == 2)
	{
		// velocity mode
		float4 density = sampleDensity(valueParams.vdimInv.xyz * vidxf);

		outputCoarseDensity(vidx, density);

		value = combustVelocity(value, density);

#if ENABLE_EMIT
		// velocity emitter
		value = emitVelocity(value, threadIdx1D, grid_ndc, vidx);
#endif
	}
#endif

	// fade field
	float fadeRate = fadeFieldSRV.SampleLevel(borderSampler, vidxNorm, 0);
	
	// Reduces fade region size
	fadeRate = saturate(1.3333f * fadeRate);

	value = fade(value, linearFade, damping);

	value *= fadeRate;

	valueOutUAV[ridx] = value;
	//outputValue(vidx, value);
}