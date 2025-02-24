
struct Output
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#define NvFlowFloat4 float4
#define NvFlowUint4 uint4
#define NvFlowUint uint

cbuffer params : register(b0)
{
#include "../NvFlow/compositeShaderParams.h"
}

Output compositeVS(float4 pos : POSITION)
{
	Output output;
	output.position.xy = pos.xy;
	output.position.z = 0.f;
	output.position.w = 1.f;
	//output.uv = float2(0.5f, -0.5f) * pos.xy + 0.5f.xx;
	output.uv = uvScale.xy * pos.zw + uvScale.zw;
	return output;
}