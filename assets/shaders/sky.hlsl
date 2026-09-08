// Sky pass: one fullscreen triangle at the far plane painting the
// per-slot light buffer's skyColor over background pixels. Depth test
// LESS_OR_EQUAL against the cleared depth (1.0) with writes off, drawn
// after the opaques — only pixels no geometry covered pass, so the
// "clear color" becomes per-frame dynamic without touching the static
// command recordings (which bake the real clear value at prerecord).

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the light buffer
    uint cascade;    // unused; layout matches the scene pass
};
[[vk::push_constant]] PushConstants pc;

#include "shading.hlsli"

struct VSOutput {
    float4 position : SV_Position;
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
    // Fullscreen triangle from the vertex id, pinned to depth 1.0.
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VSOutput output;
    output.position = float4(corners[vertexId], 1.0f, 1.0f);
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    return float4(lights[pc.cameraSlot].skyColor.rgb, 1.0f);
}
