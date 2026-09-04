// Scene pass: geometry from the memory pool via indirect draws, camera via
// a per-frame-slot buffer (so a moving camera never touches the static
// command buffers — only the slot index is baked as a push constant),
// materials through the bindless table. Each indirect entry's
// firstInstance is the object index (dxc maps SV_InstanceID to SPIR-V
// InstanceIndex, which includes firstInstance).

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera buffer
};
[[vk::push_constant]] PushConstants pc;

// Explicitly column_major: dxc does NOT apply the cbuffer default to
// matrices inside structured buffers, so an unqualified float4x4 here
// reads transposed (wrong camera position/orientation).
struct CameraData {
    column_major float4x4 viewProj; // matches rend::math memcpy
};
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras;

struct ObjectData {
    uint textureIndex; // into the bindless texture array
    uint alphaMasked;  // non-zero: discard below alphaCutoff
    float alphaCutoff;
    float pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects;
[[vk::binding(1, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState linearSampler;

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint instanceId : SV_InstanceID;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint objectIndex : OBJECT0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = mul(cameras[pc.cameraSlot].viewProj, float4(input.position, 1.0f));
    output.normal = input.normal;
    output.uv = input.uv;
    output.objectIndex = input.instanceId;
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const ObjectData object = objects[input.objectIndex];
    const float4 albedo =
        textures[NonUniformResourceIndex(object.textureIndex)].Sample(linearSampler, input.uv);
    if (object.alphaMasked != 0 && albedo.a < object.alphaCutoff) {
        discard;
    }

    // Hemispherical + directional shading off the world normal; a proper
    // lighting model arrives with the render-technique work.
    const float3 n = normalize(input.normal);
    const float3 lightDir = normalize(float3(0.3f, 1.0f, 0.2f));
    const float direct = saturate(dot(n, lightDir));
    const float sky = n.y * 0.25f + 0.45f;
    return float4(albedo.rgb * (sky + 0.55f * direct), 1.0f);
}
