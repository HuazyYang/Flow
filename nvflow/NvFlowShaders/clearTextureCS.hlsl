
RWTexture3D<float4> textureUAV : register(u0);

[numthreads(8, 8, 8)]
void clearTextureCS(uint3 tidx : SV_DispatchThreadID)
{
	//uint3 dim;
	//textureUAV.GetDimensions(dim.x, dim.y, dim.z);

	//float3 center = (float3(dim)+0.5f.xxx) * 0.5f;
	//float3 coord = (float3(tidx)+0.5f.xxx);

	//float3 offset = coord - center;
	////float dist2 = dot(offset, offset);
	////float dist = sqrt(dist2);
	//float dist = max(abs(offset.x), max(abs(offset.y), abs(offset.z)));

	//float thresh = float(dim.x) * 0.25f;

	//textureUAV[tidx] = (-thresh + dist) / thresh; // dist < thresh ? 0.5f.xxxx : -0.1f.xxxx;

	textureUAV[tidx] = 0.f.xxxx;
}