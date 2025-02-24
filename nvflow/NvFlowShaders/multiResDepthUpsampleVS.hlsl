
struct Output
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

cbuffer params : register(b0)
{
	float4 uvScale;
}

Output multiResDepthUpsampleVS(float4 pos : POSITION)
{
	Output output;
	output.position.xy = pos.xy;
	output.position.z = 0.f;
	output.position.w = 1.f;
	output.uv = uvScale.xy * pos.zw + uvScale.zw;
	return output;
}