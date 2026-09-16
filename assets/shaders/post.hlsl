// Post-process pass: one fullscreen triangle that Loads the RGBA16F
// scene-color target (binding 35 — everything the composite pass drew:
// lighting, sky, transparents, or the traced-primary output) and writes
// the exposed + tonemapped result to the sRGB swapchain. Exposure and the
// curve selection ride the per-slot light buffer, so both are live under
// static recordings. The ImGui overlay draws after this pass, untouched.

#include "backend.hlsli"

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the light buffer
    uint cascade;    // unused; layout shared with the scene passes
};
REND_PUSH(PushConstants, pc);

#include "shading.hlsli"

[[vk::binding(35, 0)]] Texture2D<float4> sceneColor REND_T(35);

struct VSOutput {
    float4 position : SV_Position;
};

// Fullscreen triangle from SV_VertexID (no vertex buffer); the pipeline
// has no depth attachment, so z is irrelevant.
VSOutput VSMain(uint vertexId : SV_VertexID) {
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VSOutput output;
    output.position = rendClip(float4(corners[vertexId], 1.0f, 1.0f));
    return output;
}

// ACES filmic fit (Narkowicz): cheap rational approximation of the RRT+ODT
// reference, applied per channel in linear light.
float3 acesFitted(float3 x) {
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const LightData light = lights[pc.cameraSlot];
    const float4 scene = sceneColor.Load(int3(int2(input.position.xy), 0));
    float3 color = scene.rgb;
    color *= max(light.exposure, 0.0f);
    if (light.tonemap == 1u) {
        color = acesFitted(color);
    }
    // The swapchain is sRGB: hardware encodes on store; clamp keeps the
    // "off" curve from wrapping on HDR values. Alpha passes through: the
    // composite pass leaves 1 under geometry and the clear alpha (1, or 0
    // for a transparent window) on uncovered pixels; blended transparents
    // over a 0 background already leave premultiplied color, which is
    // what a PRE_MULTIPLIED swapchain composites.
    return float4(saturate(color), scene.a);
}
