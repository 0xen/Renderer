#version 450

layout (input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput inputDepth;

layout(location = 0) out vec4 outColor;

void main() 
{
	vec4 tint = vec4(0.0f,1.0f,1.0f,1.0f);
	//vec4 tint = vec4(1.0f);

	vec3 color = subpassLoad(inputColor).rgb;
	outColor = vec4(color,1.0f) * tint;
}