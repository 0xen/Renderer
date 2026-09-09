// Point-light shadow capture (fragment stage only — pairs with the
// engine's scene.vert.spv like the probe capture does): the scene drawn
// into one cube face from the light's position, writing the WORLD
// DISTANCE from the light to the surface into an R32F target. The
// capture camera slot's position IS the light position. Alpha-masked
// materials still discard so foliage shadows match their silhouettes.
// Sampled later by shading.hlsli's shadePointLights (distance compare).

struct PushConstants {
    uint cameraSlot; // capture face slot; cameras[slot].position = the light
    uint cascade;    // unused; layout matches the scene pass
};
[[vk::push_constant]] PushConstants pc;

#include "shading.hlsli"

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : POSITION1;
    float viewDepth : DEPTH0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint objectIndex : OBJECT0;
};

float PSMain(VSOutput input) : SV_Target0 {
    const ObjectData object = objects[input.objectIndex];
    if ((object.flags & kFlagAlphaMasked) != 0) {
        const float alpha = textures[NonUniformResourceIndex(object.textureIndex)]
                                .Sample(linearSampler, input.uv)
                                .a;
        if (alpha < object.alphaCutoff) {
            discard;
        }
    }
    // Touch viewDepth/normal at negligible weight so dxc keeps them in
    // the input signature — scene.vert.spv writes every location, and a
    // stripped input triggers an interface-mismatch validation warning.
    const float keep =
        (input.viewDepth + input.normal.x + input.normal.y + input.normal.z) * 1.0e-20f;
    return length(input.worldPos - cameras[pc.cameraSlot].position.xyz) + keep;
}
