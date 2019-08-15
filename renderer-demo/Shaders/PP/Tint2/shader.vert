#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inDiffuseColor;
layout(location = 3) in vec2 inUV;
layout(location = 4) in uint matID;



void main()
{
	gl_Position = vec4(inPosition,1.0f);

}