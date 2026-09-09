// Deferred lighting pass: one fullscreen triangle that Loads the G-buffer
// (bindings 28-31, written by gbuffer.hlsl) and runs the SAME lighting
// chain as the forward scene pass — cascade or traced sun shadow, GGX,
// ambient, point lights (+shadow cubes / traced), reflections, debug tint,
// volumetric fog. World position is reconstructed from the camera's
// premultiplied ray axes and the stored view depth (the rt_primary
// mechanism), so no inverse matrices are needed. Background pixels
// (viewDepth 0) discard, leaving the baked clear color for the sky pass
// to overdraw. Compiled twice like scene.hlsl: RT_SHADOWS=0 (ps_6_0) and
// RT_SHADOWS=1 (ps_6_5, RayQuery devices).

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera buffer
    uint cascade;    // unused; layout shared with the scene pass
};
[[vk::push_constant]] PushConstants pc;

#include "shading.hlsli"
#include "lighting.hlsli"
#if RT_SHADOWS
#include "rt_common.hlsli"
#endif

[[vk::binding(28, 0)]] Texture2D gbufferAlbedo;
[[vk::binding(29, 0)]] Texture2D gbufferNormal;
[[vk::binding(30, 0)]] Texture2D gbufferMaterial;
[[vk::binding(31, 0)]] Texture2D<float> gbufferViewDepth;

struct VSOutput {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

// Fullscreen triangle from SV_VertexID (no vertex buffer); depth state is
// fully disabled on this pipeline, so z is irrelevant.
VSOutput VSMain(uint vertexId : SV_VertexID) {
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VSOutput output;
    output.position = float4(corners[vertexId], 1.0f, 1.0f);
    output.ndc = corners[vertexId];
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const int3 pixel = int3(int2(input.position.xy), 0);
    const float viewDepth = gbufferViewDepth.Load(pixel);
    if (viewDepth <= 0.0f) {
        // No geometry here: keep the baked clear color (the sky pass
        // paints over depth-1.0 pixels right after this triangle).
        discard;
    }
    const float3 albedo = gbufferAlbedo.Load(pixel).rgb;
    const float3 n = normalize(gbufferNormal.Load(pixel).xyz);
    const float4 material = gbufferMaterial.Load(pixel);
    const float roughness = material.r;
    const float metallic = material.g;
    const bool reflective = material.b > 0.5f;

    const CameraData cam = cameras[pc.cameraSlot];
    // The stored view depth is the distance along the forward axis; the
    // premultiplied ray axes rebuild the exact world position rasterized
    // into this pixel (see rt_primary.hlsl's ray generation).
    const float3 worldPos =
        cam.position.xyz + (cam.forwardAxis.xyz + cam.rightAxis.xyz * input.ndc.x -
                            cam.upAxis.xyz * input.ndc.y) *
                               viewDepth;

    const LightData light = lights[pc.cameraSlot];
    const float3 l = -normalize(light.direction);
    // A zero-intensity sun (scripted night) contributes nothing — skip
    // its shadow work entirely instead of tracing/sampling for it.
    const float direct = light.intensity > 0.0f ? saturate(dot(n, l)) : 0.0f;
    uint cascade = 0;
    float shadow = 1.0f;
#if RT_SHADOWS
    if (light.rtShadows != 0) {
        shadow = direct > 0.0f ? shadowRay(worldPos, n, light) : 0.0f;
    } else
#endif
    if (light.cascadeCount > 0) {
        shadow = direct > 0.0f ? shadowFactor(worldPos, n, viewDepth, light, cascade) : 0.0f;
    }

    const float3 v = normalize(cam.position.xyz - worldPos);
    const float3 sun =
        shadeSurface(albedo, metallic, roughness, n, v, l, light.color, light.intensity, shadow);
    float3 color = albedo * ambientLight(n, light.ambientColor.rgb) + sun;
#if RT_SHADOWS
    if (light.rtShadows != 0) {
        color += shadePointLightsTraced(albedo, metallic, roughness, n, v, worldPos, light);
    } else
#endif
    {
        color += shadePointLights(albedo, metallic, roughness, n, v, worldPos, light);
    }
    if (reflective && light.reflections != kReflectionNone) {
        float3 reflected = 0.0f;
        bool haveReflection = false;
#if RT_SHADOWS
        if (light.reflections == kReflectionTraced) {
            reflected = traceReflection(worldPos + n * 1.0e-3f, reflect(-v, n),
                                        length(worldPos - cam.position.xyz), cam.position.w,
                                        light);
            haveReflection = true;
        }
#endif
        if (!haveReflection) {
            reflected = sampleReflectionProbe(reflect(-v, n), roughness);
        }
        const float3 f0 = lerp(0.04f, albedo, metallic);
        color = mixReflection(color, reflected, f0, roughness, dot(n, v));
    }
    if (light.debugTint != 0) {
        const float3 tints[4] = {float3(1.0f, 0.6f, 0.6f), float3(0.6f, 1.0f, 0.6f),
                                 float3(0.6f, 0.6f, 1.0f), float3(1.0f, 1.0f, 0.6f)};
        color *= tints[cascade];
    }
    if (light.fogColor.w > 0.0f) {
        color = applyFog(color, cam.position.xyz, worldPos, viewDepth, input.position.xy, light);
    }
    return float4(color, 1.0f);
}
