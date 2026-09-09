// Sky pass: one fullscreen triangle at the far plane, depth test
// LESS_OR_EQUAL against the cleared depth (1.0) with writes off, drawn
// after the opaques — only pixels no geometry covered pass, so the
// background is per-frame dynamic without touching the static command
// recordings (which bake the real clear value at prerecord).
//
// Two modes, selected by the light buffer per frame:
//   ambientColor.w <  0  — flat skyColor.rgb (the original clear-color
//                          replacement; every scene's default).
//   ambientColor.w >= 0  — procedural skybox CUBE: the fragment's view
//                          direction (rebuilt from the camera's ray axes,
//                          same math as rt_primary) picks a cube face by
//                          dominant axis, and the day phase in [0,1)
//                          (0 sunrise, 0.25 noon, 0.5 sunset) drives the
//                          palette. A stand-in for a real cubemap: face
//                          selection is exactly what a cubemap sample
//                          would do, so swapping in a TextureCube later
//                          changes only this shader.

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera/light buffers
    uint cascade;    // unused; layout matches the scene pass
};
[[vk::push_constant]] PushConstants pc;

#include "shading.hlsli"

struct VSOutput {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0; // Vulkan NDC: x right, y down
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
    // Fullscreen triangle from the vertex id, pinned to depth 1.0.
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VSOutput output;
    output.position = float4(corners[vertexId], 1.0f, 1.0f);
    output.ndc = corners[vertexId];
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const LightData light = lights[pc.cameraSlot];
    const float timeOfDay = light.ambientColor.w;
    if (timeOfDay < 0.0f) {
        return float4(light.skyColor.rgb, 1.0f);
    }

    // View direction through this pixel (rt_primary's reconstruction: the
    // camera axes are premultiplied by tan(fov/2), NDC y points down).
    const CameraData cam = cameras[pc.cameraSlot];
    const float3 dir = normalize(cam.forwardAxis.xyz + cam.rightAxis.xyz * input.ndc.x -
                                 cam.upAxis.xyz * input.ndc.y);

    // Day phase -> palette weights. altitude = sin(2 pi t): positive is
    // daytime; the dusk band peaks while the sun crosses the horizon.
    const float t = frac(timeOfDay);
    const float altitude = sin(t * 2.0f * kPi);
    const float day = smoothstep(0.0f, 0.25f, altitude);
    const float dusk = saturate(1.0f - abs(altitude) * 3.0f);

    // Vertical gradient in each palette, blended night -> day -> dusk.
    const float up = saturate(dir.y);
    const float3 dayColor = lerp(float3(0.55f, 0.68f, 0.85f), float3(0.28f, 0.48f, 0.80f), up);
    const float3 nightColor =
        lerp(float3(0.012f, 0.016f, 0.038f), float3(0.003f, 0.005f, 0.016f), up);
    float3 sky = lerp(nightColor, dayColor, day);
    sky = lerp(sky, float3(0.85f, 0.32f, 0.10f), dusk * saturate(1.0f - up * 2.0f) * 0.8f);

    // Colored-cube face tint: the dominant axis is the face a cubemap
    // lookup would hit — tinted so the placeholder cube reads as a cube.
    const float3 a = abs(dir);
    float3 faceTint;
    if (a.x >= a.y && a.x >= a.z) {
        faceTint = dir.x > 0.0f ? float3(1.10f, 0.97f, 0.93f) : float3(0.90f, 1.00f, 1.08f);
    } else if (a.y >= a.z) {
        faceTint = dir.y > 0.0f ? float3(0.98f, 1.02f, 1.10f) : float3(0.88f, 0.90f, 0.88f);
    } else {
        faceTint = dir.z > 0.0f ? float3(1.06f, 1.00f, 0.92f) : float3(0.94f, 1.00f, 1.05f);
    }
    sky *= faceTint;

    // A soft glow around the sun direction ties the box to the light.
    const float glow = pow(saturate(dot(dir, -normalize(light.direction))), 32.0f);
    sky += light.color * light.intensity * glow * 0.35f;
    return float4(sky, 1.0f);
}
