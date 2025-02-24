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

#include "../NvFlowShaders/frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowFloat4x4 positionToField;
	NvFlowShaderLinearParams linearParams;
	NvFlowUint4 fieldDim;
	NvFlowUint4 particleCount;
};

RWTexture3D<float2> fieldUAV : register(u0);

Buffer<float4> positionSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);

VIRTUAL_TO_REAL(VirtualToReal, blockTableSRV, linearParams);

void writeField(int3 vidx, float2 val)
{
	int3 ridx = VirtualToReal(vidx);
	fieldUAV[ridx] = val;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceSplatCS(uint3 tidx : SV_DispatchThreadID)
{
	if (tidx.x < particleCount.x)
	{
		float4 position = positionSRV[tidx.x];

		float4 gridPos = mul(position, positionToField);

		float3 fieldIdxf = fieldDim.xyz * (0.5f * gridPos.xyz + 0.5f) - 0.5f;

		int3 fieldIdx = floor(fieldIdxf);

		float3 w1 = fieldIdxf - float3(fieldIdx);
		float3 w0 = 1.f.xxx - w1;

		writeField(fieldIdx + int3(0, 0, 0), float(w0.x * w0.y * w0.z).xx);
		writeField(fieldIdx + int3(1, 0, 0), float(w1.x * w0.y * w0.z).xx);
		writeField(fieldIdx + int3(0, 1, 0), float(w0.x * w1.y * w0.z).xx);
		writeField(fieldIdx + int3(1, 1, 0), float(w1.x * w1.y * w0.z).xx);
		writeField(fieldIdx + int3(0, 0, 1), float(w0.x * w0.y * w1.z).xx);
		writeField(fieldIdx + int3(1, 0, 1), float(w1.x * w0.y * w1.z).xx);
		writeField(fieldIdx + int3(0, 1, 1), float(w0.x * w1.y * w1.z).xx);
		writeField(fieldIdx + int3(1, 1, 1), float(w1.x * w1.y * w1.z).xx);
	}
}