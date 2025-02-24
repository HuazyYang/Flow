/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "frameworkHybrid.hlsli"

#ifndef NO_DENSITY
#define NO_DENSITY 0
#endif

#define THREAD_DIM_X 10
#define THREAD_DIM_Y 10
#define THREAD_DIM_Z 10

cbuffer params : register(b0)
{
	NvFlowShaderPointParams velocityParams;

	float scale;
	float velocityMask;
	float temperatureMask;
	float smokeMask;

	float fuelMask;
	float constantMask;
	float pad1;
	float pad2;

	NvFlowShaderLinearParams densityParams;
};

Buffer<uint> blockListSRV : register(t0);

RWTexture3D<float4> velocityUAV : register(u0);
Texture3D<uint> velocityBlockTable : register(t1);

Texture3D<float4> velocitySRV : register(t2);

Texture3D<uint> densityBlockTable : register(t3);
Texture3D<float4> densitySRV : register(t4);

SAMPLE_POINT_3D(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);

SAMPLE_LINEAR_3D(sampleDensity, float4, densitySRV, densityBlockTable, densityParams);

VIRTUAL_TO_REAL(VirtualToReal, velocityBlockTable, velocityParams);

DISPATCH_ID_TO_VIRTUAL(blockListSRV, velocityParams);

float3 crossProduct(float3 a, float3 b)
{
	return float3(a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x);
}

groupshared float4 sLowpass[THREAD_DIM_Z][THREAD_DIM_Y][THREAD_DIM_X];
groupshared float4 sCurl[THREAD_DIM_Z][THREAD_DIM_Y][THREAD_DIM_X];

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void vorticityConfinementCS(uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
	int3 tidx = int3(blockIdx << 3u) + int3(clamp(threadIdx, 1, 8)) - 1;
	int3 vidx = DispatchIDToVirtual(tidx);

	vidx += threadIdx - clamp(threadIdx, 1, 8);

	int3 ridx = VirtualToReal(vidx);

	// curl and lowpass 0
	{
		float4 temp;
		float4 curl = float4(0.f, 0.f, 0.f, 0.f);
		float4 val = 0.f.xxxx;

		temp = sampleVelocity(vidx + int3(-1, 0, 0));
		curl.y += temp.z;
		curl.z -= temp.y;
		val += temp;

		temp = sampleVelocity(vidx + int3(+1, 0, 0));
		curl.y -= temp.z;
		curl.z += temp.y;
		val += temp;

		temp = sampleVelocity(vidx + int3(0, -1, 0));
		curl.x -= temp.z;
		curl.z += temp.x;
		val += temp;

		temp = sampleVelocity(vidx + int3(0, +1, 0));
		curl.x += temp.z;
		curl.z -= temp.x;
		val += temp;

		temp = sampleVelocity(vidx + int3(0, 0, -1));
		curl.y -= temp.x;
		curl.x += temp.y;
		val += temp;

		temp = sampleVelocity(vidx + int3(0, 0, +1));
		curl.y += temp.x;
		curl.x -= temp.y;
		val += temp;

		val += velocitySRV[ridx];

		val *= (1.f / 7.f);

		curl.w = sqrt(curl.x*curl.x + curl.y*curl.y + curl.z*curl.z);

		sCurl[threadIdx.z][threadIdx.y][threadIdx.x] = curl;
		sLowpass[threadIdx.z][threadIdx.y][threadIdx.x] = val;
	}

	GroupMemoryBarrierWithGroupSync();

	if (all(threadIdx > 0) && all(threadIdx < 9))
	{
		float4 n = float4(0.f, 0.f, 0.f, 0.f);
		n.x -= sCurl[threadIdx.z][threadIdx.y][threadIdx.x - 1].w;
		n.x += sCurl[threadIdx.z][threadIdx.y][threadIdx.x + 1].w;
		n.y -= sCurl[threadIdx.z][threadIdx.y - 1][threadIdx.x].w;
		n.y += sCurl[threadIdx.z][threadIdx.y + 1][threadIdx.x].w;
		n.z -= sCurl[threadIdx.z - 1][threadIdx.y][threadIdx.x].w;
		n.z += sCurl[threadIdx.z + 1][threadIdx.y][threadIdx.x].w;

		n.w = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		if (n.w > 0.f)
		{
			n.x /= n.w;
			n.y /= n.w;
			n.z /= n.w;
		}

		float4 curl = sCurl[threadIdx.z][threadIdx.y][threadIdx.x];

		float3 impulse = crossProduct(n.xyz, curl.xyz);
		float impulseWeight = scale;

		float speed = 0.f;
		if (velocityMask != 0.f)
		{
			float4 velocityLowpass = sLowpass[threadIdx.z][threadIdx.y][threadIdx.x];
			velocityLowpass += sLowpass[threadIdx.z][threadIdx.y][threadIdx.x - 1];
			velocityLowpass += sLowpass[threadIdx.z][threadIdx.y][threadIdx.x + 1];
			velocityLowpass += sLowpass[threadIdx.z][threadIdx.y - 1][threadIdx.x];
			velocityLowpass += sLowpass[threadIdx.z][threadIdx.y + 1][threadIdx.x];
			velocityLowpass += sLowpass[threadIdx.z - 1][threadIdx.y][threadIdx.x];
			velocityLowpass += sLowpass[threadIdx.z + 1][threadIdx.y][threadIdx.x];

			velocityLowpass *= (1.f / 7.f);

			velocityLowpass.w = length(velocityLowpass.xyz);
			speed = velocityLowpass.w;
			if (speed > 0.f)
			{
				speed = log2(speed + 1.f);
			}
		}
		float4 density = 0.f.xxxx;
		#if (NO_DENSITY == 0)
		//if (temperatureMask != 0.f || smokeMask != 0.f || fuelMask != 0.f)
		{
			density = sampleDensity(float3(vidx)+0.5f);
		}
		#endif

		impulseWeight *= max(0.f, (
			velocityMask * speed +
			temperatureMask * abs(density.x) +
			smokeMask * abs(density.w) +
			fuelMask * abs(density.y) +
			constantMask
			));

		float4 velocity = velocitySRV[ridx];

		velocity.xyz += impulseWeight * impulse;

		velocityUAV[ridx] = velocity;
	}
}