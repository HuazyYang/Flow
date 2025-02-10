
struct Output
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	#include "../NvFlow/crossSectionShaderParams.h"
}

Output crossSectionVS(float4 pos : POSITION)
{
	Output output;
	output.position.xy = posScale.xy * pos.xy + posScale.zw;
	output.position.z = 0.f;
	output.position.w = 1.f;
	output.uv = uvScale.xy * pos.zw + uvScale.zw;
	return output;
}