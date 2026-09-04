// Single-source HLSL, compiled offline to SPIR-V by dxc (see assets/CMakeLists.txt).
// Milestone 6: no vertex buffers — the three vertices are generated from
// SV_VertexID so the first triangle needs no geometry upload path yet.

struct VSOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
    // Vulkan clip space has +Y pointing down, so the first vertex is the apex.
    const float2 positions[3] = {
        float2(0.0f, -0.6f),
        float2(0.6f, 0.6f),
        float2(-0.6f, 0.6f),
    };
    const float3 colors[3] = {
        float3(1.0f, 0.15f, 0.15f),
        float3(0.15f, 1.0f, 0.25f),
        float3(0.2f, 0.35f, 1.0f),
    };

    VSOutput output;
    output.position = float4(positions[vertexId], 0.0f, 1.0f);
    output.color = colors[vertexId];
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    return float4(input.color, 1.0f);
}
