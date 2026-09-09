// Deferred G-buffer pass fragment shader: pairs the engine's scene.vert.spv
// (VSOutput mirrored EXACTLY, every field consumed — dxc strips untouched
// input locations, which breaks the pairing) and evaluates the surface's
// MATERIAL only: albedo fetch + alpha-mask discard, cotangent-frame normal
// map, metallic/roughness. All lighting happens later in deferred.hlsl,
// which Loads these four targets. Target order/formats must match
// FrameRenderer::kGBufferFormats.

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera buffer
    uint cascade;    // unused here; layout shared with the scene pass
};
[[vk::push_constant]] PushConstants pc;

#include "shading.hlsli"
#include "lighting.hlsli"

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : POSITION1;
    float viewDepth : DEPTH0;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint objectIndex : OBJECT0;
};

struct PSOutput {
    float4 albedo : SV_Target0;   // rgb = albedo (sRGB target)
    float4 normal : SV_Target1;   // xyz = world normal (post normal-map)
    float4 material : SV_Target2; // r = roughness, g = metallic, b = reflective flag
    float viewDepth : SV_Target3; // clip w; the cleared 0 marks background
};

PSOutput PSMain(VSOutput input) {
    const ObjectData object = objects[input.objectIndex];
    const float4 albedo =
        textures[NonUniformResourceIndex(object.textureIndex)].Sample(linearSampler, input.uv);
    if ((object.flags & kFlagAlphaMasked) != 0 && albedo.a < object.alphaCutoff) {
        discard;
    }

    float3 n = normalize(input.normal);
    n = applyNormalMap(object.normalIndex, n, input.worldPos, input.uv);
    float metallic = object.metallicFactor;
    float roughness = object.roughnessFactor;
    if (object.mrIndex != 0) {
        const float2 mr =
            textures[NonUniformResourceIndex(object.mrIndex)].Sample(linearSampler, input.uv).gb;
        roughness *= mr.x;
        metallic *= mr.y;
    }

    PSOutput output;
    output.albedo = float4(albedo.rgb, 1.0f);
    output.normal = float4(n, 0.0f);
    output.material = float4(roughness, metallic,
                             (object.flags & kFlagReflective) != 0 ? 1.0f : 0.0f, 0.0f);
    // Touch the remaining interpolant so dxc keeps every input location.
    output.viewDepth = input.viewDepth + input.worldPos.x * 1.0e-20f;
    return output;
}
