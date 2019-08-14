#version 450

layout (set = 0, binding = 0) uniform sampler2D diffuse_texture;

layout(location = 0) out vec4 outColor;


layout(location = 0) in vec2 inUV;



void main() 
{
	vec4 tint = vec4(1.0f,0.5f,0.5f,1.0f);

	outColor = texture(diffuse_texture, inUV, 0.0f) * tint;
}