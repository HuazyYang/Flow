/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define BLOCK_DIM 8
#define SCAN_DIM 256

Texture3D<float4> blockNormalFieldSRV : register(t0);

RWBuffer<uint> blockListUAV : register(u0);

RWBuffer<uint> atomicUAV : register(u1);

groupshared uint sdata0[SCAN_DIM];
groupshared uint sdata1[SCAN_DIM];

groupshared uint sglobalOffset;

// Note: this is designed for 256 threads in a 1D block
uint2 blockScan(uint3 threadIdx, uint2 val, out uint totalCount)
{
	uint localVal = val.x + val.y;
	sdata0[threadIdx.x] = localVal;

	GroupMemoryBarrierWithGroupSync();

	if (threadIdx.x >= 1) localVal += sdata0[threadIdx.x - 1];
	if (threadIdx.x >= 2) localVal += sdata0[threadIdx.x - 2];
	if (threadIdx.x >= 3) localVal += sdata0[threadIdx.x - 3];
	sdata1[threadIdx.x] = localVal;

	GroupMemoryBarrierWithGroupSync();

	if (threadIdx.x >= 4) localVal += sdata1[threadIdx.x - 4];
	if (threadIdx.x >= 8) localVal += sdata1[threadIdx.x - 8];
	if (threadIdx.x >= 12) localVal += sdata1[threadIdx.x - 12];
	sdata0[threadIdx.x] = localVal;

	GroupMemoryBarrierWithGroupSync();

	if (threadIdx.x >= 16) localVal += sdata0[threadIdx.x - 16];
	if (threadIdx.x >= 32) localVal += sdata0[threadIdx.x - 32];
	if (threadIdx.x >= 48) localVal += sdata0[threadIdx.x - 48];
	sdata1[threadIdx.x] = localVal;

	GroupMemoryBarrierWithGroupSync();

	if (threadIdx.x >= 64) localVal += sdata1[threadIdx.x - 64];
	if (threadIdx.x >= 128) localVal += sdata1[threadIdx.x - 128];
	if (threadIdx.x >= 192) localVal += sdata1[threadIdx.x - 192];

	uint2 retVal;
	retVal.y = localVal;
	retVal.x = retVal.y - val.y;

	// compute totalCount
	totalCount = sdata1[63] + sdata1[127] + sdata1[191] + sdata1[255];

	return retVal;
}

[numthreads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM / 2)]
void blockCompactCS(uint3 tidx : SV_DispatchThreadID, uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
	uint3 threadIdx1D = uint3(threadIdx.z * BLOCK_DIM * BLOCK_DIM + threadIdx.y * BLOCK_DIM + threadIdx.x, 1, 1);

	uint2 pred = uint2(0u, 0u);

	// check for boundary condition
	float4 normal0 = blockNormalFieldSRV[uint3(1, 1, 2) * tidx + uint3(0, 0, 0)];
	float4 normal1 = blockNormalFieldSRV[uint3(1, 1, 2) * tidx + uint3(0, 0, 1)];

	pred.x = (normal0.w != 0.f) ? 1u : 0u;
	pred.y = (normal1.w != 0.f) ? 1u : 0u;

	uint totalCount;
	uint2 allocIdx = blockScan(threadIdx1D, pred, totalCount) - pred;

	// make global allocation
	if (all(threadIdx == uint3(0, 0, 0)))
	{
		uint oldVal;
		InterlockedAdd(atomicUAV[0], totalCount, oldVal);
		sglobalOffset = oldVal;
	}

	GroupMemoryBarrierWithGroupSync();

	if (pred.x)
	{
		uint3 blockID = uint3(tidx.x, tidx.y, 2 * tidx.z + 0);
		blockListUAV[sglobalOffset + allocIdx.x] = ~((blockID.x << 0) | (blockID.y << 10) | (blockID.z << 20));
	}
	if (pred.y)
	{
		uint3 blockID = uint3(tidx.x, tidx.y, 2 * tidx.z + 1);
		blockListUAV[sglobalOffset + allocIdx.y] = ~((blockID.x << 0) | (blockID.y << 10) | (blockID.z << 20));
	}
}