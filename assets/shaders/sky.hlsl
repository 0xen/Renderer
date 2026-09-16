// Sky pass: one fullscreen triangle at the far plane, depth test
// LESS_OR_EQUAL against the cleared depth (1.0) with writes off, drawn
// after the opaques — only pixels no geometry covered pass, so the
// background is per-frame dynamic without touching the static command
// recordings (which bake the real clear value at prerecord).
//
// Two modes, selected by the light buffer per frame:
//   ambientColor.w <  0  — flat skyColor.rgb (the original clear-color
//                          replacement; every scene's default).
//   ambientColor.w >= 0  — procedural skybox: one UNIFORM color blended
//                          from the day phase in [0,1) (0 sunrise, 0.25
//                          noon, 0.5 sunset). Deliberately directionless
//                          — earlier per-face tints made the box corners
//                          visible. A real TextureCube later would
//                          reintroduce direction here (rebuild it from
//                          the camera ray axes as rt_primary does).

#include "backend.hlsli"

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera/light buffers
    uint cascade;    // unused; layout matches the scene pass
};
REND_PUSH(PushConstants, pc);

#include "shading.hlsli"

struct VSOutput {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0; // Vulkan NDC: x right, y down
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
    // Fullscreen triangle from the vertex id, pinned to depth 1.0.
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VSOutput output;
    output.position = rendClip(float4(corners[vertexId], 1.0f, 1.0f));
    output.ndc = corners[vertexId];
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const LightData light = lights[pc.cameraSlot];
    const float timeOfDay = light.ambientColor.w;
    if (timeOfDay < 0.0f) {
        return float4(light.skyColor.rgb, 1.0f);
    }

    // Day phase -> palette weights. altitude = sin(2 pi t): positive is
    // daytime; the dusk band peaks while the sun crosses the horizon.
    const float t = frac(timeOfDay);
    const float altitude = sin(t * 2.0f * kPi);
    const float day = smoothstep(0.0f, 0.25f, altitude);
    const float dusk = saturate(1.0f - abs(altitude) * 3.0f);

    // One uniform color: night -> day, warmed through dusk.
    float3 sky = lerp(float3(0.006f, 0.009f, 0.024f), float3(0.42f, 0.58f, 0.83f), day);
    sky = lerp(sky, float3(0.70f, 0.28f, 0.10f), dusk * 0.6f);
    return float4(sky, 1.0f);
}
