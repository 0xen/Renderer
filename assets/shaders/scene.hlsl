// Scene pass, milestone 7 step 2: geometry from the memory pool via
// indirect draws, camera via push constant. Untextured — normal-based
// shading until bindless textures land.

struct PushConstants {
    float4x4 viewProj; // column_major (HLSL default), matches rend::math
};
[[vk::push_constant]] PushConstants pc;

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = mul(pc.viewProj, float4(input.position, 1.0f));
    output.normal = input.normal;
    output.uv = input.uv;
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    // Simple hemispherical + directional shading off the world normal, so
    // geometry reads as 3D without any material system.
    const float3 n = normalize(input.normal);
    const float3 lightDir = normalize(float3(0.3f, 1.0f, 0.2f));
    const float direct = saturate(dot(n, lightDir));
    const float sky = n.y * 0.25f + 0.45f;
    const float3 base = float3(0.82f, 0.79f, 0.74f);
    return float4(base * (sky + 0.55f * direct), 1.0f);
}
