
struct Output
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	#include "../NvFlow/crossSectionShaderParams.h"
}

Texture3D<float4> valueSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);

VIRTUAL_TO_REAL_LINEAR(VirtualToReal, blockTableSRV, valueParams)

Output crossSectionVectorVS(float4 pos : POSITION, uint instanceID : SV_InstanceID, uint vertexID : SV_VertexID)
{
	int3 vidx = int3(-1, -1, -1);
	vidx.x = instanceID % valueParams.vdim.x;
	vidx.y = instanceID / valueParams.vdim.x;
	vidx.z = floor((0.5f * (0.f - crossSectionPosition.z) * crossSectionScale.z + 0.5f) * valueParams.vdim.z);

	float3 vidxf = float3(vidx)+0.5f.xxx;
	if (crossSectionAxis == 1u)
	{
		vidxf.xyz = vidxf.yzx;
	}
	else if (crossSectionAxis == 2u)
	{
		vidxf.xyz = vidxf.zyx;
	}
	int3 ridx = floor(VirtualToReal(vidxf));
	
	float4 value = valueSRV[ridx];
	if (crossSectionAxis == 1u)
	{
		value.xyz = value.yzx;
	}
	else if (crossSectionAxis == 2u)
	{
		value.xyz = value.zyx;
	}

	float3 uvw = float3(ridx)* valueParams.dimInv.xyz;

	// TODO: maybe render points
	float lengthBias = pixelSize.x;

	Output output;
	if (value.w > 0.f)
	{
		output.position.xyz = 2.f * (valueParams.vdimInv.xyz * (float3(vidx) + 0.5f)) - 1.f;

		output.position.z = 0.f;

		if (vertexID == 3)
		{
			float2 offset = value.xy;
			float len = length(value.xy);
			if (len > 0.f)
			{
				output.position.xy += value.w * (vectorLengthScale * saturate(len * velocityScale) + lengthBias) * (value.xy / len);
			}
		}

		if (vertexID == 4)
		{
			output.position.xy += valueParams.vdimInv.xy * float2(-1.f, -1.f);
		}
		if (vertexID == 5)
		{
			output.position.xy += valueParams.vdimInv.xy * float2(+1.f, -1.f);
		}
		if (vertexID == 6)
		{
			output.position.xy += valueParams.vdimInv.xy * float2(+1.f, +1.f);
		}
		if (vertexID == 7)
		{
			output.position.xy += valueParams.vdimInv.xy * float2(-1.f, +1.f);
		}

		output.position.w = 1.f;
	}
	else
	{
		output.position = float4(1.f, 0.f, 0.f, 0.f);
	}

	output.position.xy = (1.f.xx / crossSectionScale.xy) * output.position.xy + crossSectionPosition.xy;
	output.position.xy = posScale.xy * output.position.xy + posScale.zw;
	output.position.z = 0.f;
	output.position.w = 1.f;

	output.color = lineColor;
	if (vertexID >= 4)
	{
		output.color = cellColor;
	}

	return output;
}