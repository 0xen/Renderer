#version 460
#extension GL_NV_ray_tracing : require

struct RayPayload {
    uint instance;
    uint primitive;
    vec2 barycentric;
    bool hit;
};

layout(location = 0) rayPayloadInNV RayPayload rayPayload;


hitAttributeNV vec3 attribs;



void main()
{
	rayPayload.instance = gl_InstanceID;
	// AND the 'gl_PrimitiveID' data to the hit bool
	rayPayload.primitive = gl_PrimitiveID;
	rayPayload.barycentric = attribs.xy;
	rayPayload.hit = true;
}


