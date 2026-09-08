// Shadow pass: the scene's draw stream rendered depth-only from the sun's
// point of view. Alpha-masked materials (foliage, banners) still discard,
// so their shadows match their silhouettes instead of their quads.

struct PushConstants {
    uint slot;    // frame-in-flight index into the light buffer
    uint cascade; // which cascade this pass renders
};
[[vk::push_constant]] PushConstants pc;

// Must match LightData in the viewer / scene.hlsl.
struct LightData {
    column_major float4x4 cascadeViewProj[4];
    float4 splitDepths;
    float3 direction;
    float intensity;
    float3 color;
    float pcfRadius;
    float biasBase;
    float mapSize;
    uint cascadeCount;
    uint debugTint;
    uint rtShadows;
    uint reflections;
    uint pad3;
    uint pad4;
    float4 fogBoxMin; // volumetric fog box; only the stride matters here
    float4 fogBoxMax;
    float4 fogColor;
    // Sky/ambient + 16 point lights (2 float4 each); stride only here.
    float4 skyColor;
    float4 ambientColor;
    float4 pointLights[32];
};

struct ObjectData {
    uint textureIndex;
    uint normalIndex;
    uint mrIndex;
    uint flags; // bit 0: alpha-masked, bit 1: transparent (blend)
    float alphaCutoff;
    float baseAlpha;
    float metallicFactor;
    float roughnessFactor;
};

static const uint kFlagAlphaMasked = 1u;

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects;
[[vk::binding(1, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState linearSampler;
[[vk::binding(7, 0)]] StructuredBuffer<LightData> lights;
// Must match shading.hlsli: per-slot per-object world transforms.
static const uint kTransformCapacity = 4096;
[[vk::binding(19, 0)]] StructuredBuffer<column_major float4x4> objectTransforms;
// Must match shading.hlsli: SV_InstanceID resolves through these rows.
struct InstanceRow {
    uint objectIndex;
    uint transformIndex;
};
[[vk::binding(20, 0)]] StructuredBuffer<InstanceRow> instanceRows;

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint instanceId : SV_InstanceID;
};

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation uint objectIndex : OBJECT0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    const InstanceRow row = instanceRows[input.instanceId];
    const float3 worldPos =
        mul(objectTransforms[pc.slot * kTransformCapacity + row.transformIndex],
            float4(input.position, 1.0f))
            .xyz;
    output.position = mul(lights[pc.slot].cascadeViewProj[pc.cascade], float4(worldPos, 1.0f));
    output.uv = input.uv;
    output.objectIndex = row.objectIndex;
    return output;
}

void PSMain(VSOutput input) {
    const ObjectData object = objects[input.objectIndex];
    if ((object.flags & kFlagAlphaMasked) != 0) {
        const float alpha = textures[NonUniformResourceIndex(object.textureIndex)]
                                .Sample(linearSampler, input.uv)
                                .a;
        if (alpha < object.alphaCutoff) {
            discard;
        }
    }
}
