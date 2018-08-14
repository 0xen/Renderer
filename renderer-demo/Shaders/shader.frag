#version 450

layout(location = 0) out vec4 outColor;

layout(location = 2) in Data{
    mat4 MVP;
    vec3 normal;
}inData;

layout(location = 0) in vec2 inUV;

vec4 toGrayscale(in vec4 color)
{
  float average = (color.r + color.g + color.b) / 3.0;
  return vec4(average, average, average, 1.0);
}


void main() 
{
	vec4 color = vec4(inUV,1.0f,1.0f);
    outColor = color;
}