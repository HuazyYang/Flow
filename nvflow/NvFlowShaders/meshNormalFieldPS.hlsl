
struct Input
{
	float4 position : SV_POSITION;
	float3 normal : NORMAL;
	float4 coord : COORD;
};

RWTexture3D<float4> normalFieldUAV : register(u0);

void meshNormalFieldPS(Input input)
{
	normalFieldUAV[floor(input.coord.xyz)] = float4(input.normal.xyz, 1.0f);
}