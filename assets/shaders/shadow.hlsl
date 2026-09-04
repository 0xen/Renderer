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
};

struct ObjectData {
    uint textureIndex;
    uint alphaMasked;
    float alphaCutoff;
    float pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects;
[[vk::binding(1, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState linearSampler;
[[vk::binding(7, 0)]] StructuredBuffer<LightData> lights;

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
    output.position = mul(lights[pc.slot].cascadeViewProj[pc.cascade], float4(input.position, 1.0f));
    output.uv = input.uv;
    output.objectIndex = input.instanceId;
    return output;
}

void PSMain(VSOutput input) {
    const ObjectData object = objects[input.objectIndex];
    if (object.alphaMasked != 0) {
        const float alpha = textures[NonUniformResourceIndex(object.textureIndex)]
                                .Sample(linearSampler, input.uv)
                                .a;
        if (alpha < object.alphaCutoff) {
            discard;
        }
    }
}
