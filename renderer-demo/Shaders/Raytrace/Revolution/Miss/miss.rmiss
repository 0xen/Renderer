#version 460
#extension GL_NV_ray_tracing : require

struct RayPayload {
    uint instance;
    uint primitive;
    vec2 barycentric;
    bool hit;
};

layout(location = 0) rayPayloadInNV RayPayload rayPayload;

void main()
{
	// We have missed so we have no data
   	rayPayload.hit = false;
}