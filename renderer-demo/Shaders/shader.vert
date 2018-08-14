#version 450

layout(binding = 0) uniform UniformBufferObjectStatic {
    mat4 view;
    mat4 proj;
}ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

layout(location = 3) in mat4 model;


layout(location = 0) out vec2 outUV;

layout(location = 2) out Data{
	mat4 MVP;
	vec3 normal;
}outData;



void main() {
	outUV = inUV;
	outData.MVP = ubo.proj * ubo.view * model;
	gl_Position = outData.MVP * vec4(inPosition, 1.0);
    outData.normal = inNormal;
}