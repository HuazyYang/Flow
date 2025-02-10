
struct Input
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

float4 crossSectionVectorPS(Input input) : SV_TARGET
{
	return input.color;
}