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

// Must match LightData in the viewer / shadow.hlsl. One region per frame
// slot, like the camera.
struct LightData {
    column_major float4x4 viewProj; // light-space transform for shadow lookup
    float3 direction;               // world space, from the light toward the scene
    float intensity;
    float3 color;
    float pcfRadius; // filter radius in shadow-map texels (0 = hard 2x2)
    float biasBase;  // depth-compare bias floor; slope-scaled up to 8x
    float mapSize;   // shadow map resolution (texel size = 1/mapSize)
    float pad0;
    float pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects;
[[vk::binding(1, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState linearSampler;
[[vk::binding(7, 0)]] StructuredBuffer<LightData> lights;
[[vk::binding(8, 0)]] Texture2D<float> shadowMap;
[[vk::binding(9, 0)]] SamplerComparisonState shadowSampler;

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint instanceId : SV_InstanceID;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : POSITION1; // vertices are pre-baked world space
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint objectIndex : OBJECT0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = mul(cameras[pc.cameraSlot].viewProj, float4(input.position, 1.0f));
    output.worldPos = input.position;
    output.normal = input.normal;
    output.uv = input.uv;
    output.objectIndex = input.instanceId;
    return output;
}

// Shadow visibility: transform to light space, 3x3 PCF over hardware 2x2
// compares (radius in texels from the light data). The projection already
// bakes Vulkan's Y flip, so NDC y maps straight to V.
float shadowFactor(float3 worldPos, float3 n, LightData light) {
    const float4 lightClip = mul(light.viewProj, float4(worldPos, 1.0f));
    const float2 uv = lightClip.xy * 0.5f + 0.5f;
    // Slope-scaled bias against acne on faces the light grazes.
    const float ndotl = saturate(dot(n, -light.direction));
    const float bias = clamp(light.biasBase / max(ndotl, 0.05f), light.biasBase,
                             light.biasBase * 8.0f);
    const float depth = lightClip.z - bias;
    if (light.pcfRadius <= 0.0f) {
        return shadowMap.SampleCmpLevelZero(shadowSampler, uv, depth);
    }
    const float texel = light.pcfRadius / light.mapSize;
    float sum = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            sum += shadowMap.SampleCmpLevelZero(shadowSampler,
                                                uv + float2(x, y) * texel, depth);
        }
    }
    return sum / 9.0f;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const ObjectData object = objects[input.objectIndex];
    const float4 albedo =
        textures[NonUniformResourceIndex(object.textureIndex)].Sample(linearSampler, input.uv);
    if (object.alphaMasked != 0 && albedo.a < object.alphaCutoff) {
        discard;
    }

    const LightData light = lights[pc.cameraSlot];
    const float3 n = normalize(input.normal);
    const float direct = saturate(dot(n, -normalize(light.direction)));
    const float shadow = direct > 0.0f ? shadowFactor(input.worldPos, n, light) : 0.0f;

    // Sun + simple hemispherical ambient; a real lighting model arrives
    // with the render-technique work.
    const float3 sun = light.color * (light.intensity * direct * shadow);
    const float3 ambient = float3(0.30f, 0.32f, 0.36f) * (n.y * 0.2f + 0.5f);
    return float4(albedo.rgb * (ambient + sun), 1.0f);
}
