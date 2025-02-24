
struct Output
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

cbuffer params : register(b0)
{
	float4 uvScale;
}

Output depthDownsampleVS(float4 pos : POSITION)
{
	Output output;
	output.position.xy = pos.zw;
	output.position.z = 0.f;
	output.position.w = 1.f;
	//output.uv = float2(0.5f, -0.5f) * pos.xy + 0.5f.xx;
	output.uv = uvScale.xy * pos.xy + uvScale.zw;
	return output;
}