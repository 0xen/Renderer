#version 450

layout (binding = 1) uniform sampler2D object_texture;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec2 inUV;

void main() 
{
	vec4 color = texture(object_texture, inUV, 0.0f);
    outColor = color;
}