#define BLOCK_DIM 256

#include "framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowUint4 blockDim;
	NvFlowFloat4 rayOriginVirtual;
	NvFlowFloat4 rayForwardDirVirtual;
	NvFlowUint4 numBlocks;
}

Buffer<uint> blockListSRV : register(t0);

RWBuffer<uint> keyUAV : register(u0);
RWBuffer<uint> valUAV : register(u1);

uint flip(float zfloat)
{
	uint zuint = asuint(zfloat);
	uint mask = -int(zuint >> 31) | 0x80000000;
	return zuint ^ mask;
}

[numthreads(BLOCK_DIM, 1, 1)]
void volumeRenderSortCS( uint3 tidx : SV_DispatchThreadID )
{
	uint blockID = tidx.x;

	uint blockListVal = blockListSRV[blockID];
	int3 vBlockIdx = tableVal_to_coord(blockListVal);

	float3 vidx = float3(vBlockIdx * blockDim.xyz) + 0.5f * float3(blockDim.xyz);
	float3 displacement = vidx - rayOriginVirtual.xyz;

	//float keyf = dot(displacement,rayForwardDirVirtual.xyz);
	float keyf = dot(displacement, displacement);

	uint key;
	if (blockID < numBlocks.x && blockListVal != 0)
	{
		key = flip(keyf);
	}
	else
	{
		key = 0xFFFFFFFF;
	}

	keyUAV[tidx.x] = key;
	valUAV[tidx.x] = blockListVal;
}