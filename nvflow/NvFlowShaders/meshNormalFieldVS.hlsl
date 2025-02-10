
cbuffer params : register(b0)
{
	float4x4 projection;
	float4 normalFieldDim;
	uint4 swizzleMode;
};

struct Input
{
	float4 position : POSITION;
	float4 normal : NORMAL;
};

struct Output
{
	float4 position : SV_POSITION;
	float3 normal : NORMAL;
	float4 coord : COORD;
};

Output meshNormalFieldVS(Input input, uint instance : SV_InstanceID)
{
	Output output;
	output.position = mul(input.position, projection);
	output.normal = input.normal.xyz;

	output.coord = normalFieldDim * (0.5f * output.position + 0.5f) + 0.5f;

	// swizzle position
	if (swizzleMode.x == 0)
	{
		output.position.xyz = output.position.xyz;
	}
	if (swizzleMode.x == 1)
	{
		output.position.yzx = output.position.xyz;
	}
	if (swizzleMode.x == 2)
	{
		output.position.zxy = output.position.xyz;
	}

	//output.position.z *= -1.f;
	//output.position.z += 0.5f;

	output.position.z = 0.5f * output.position.z + 0.5f;

	//output.position.y -= 1.f * float(instance) + 0.1f;

	return output;
}