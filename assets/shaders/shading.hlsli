// Shared surface-shading vocabulary for the scene passes: the bindless
// data layouts (must match the viewer structs byte for byte), the common
// bindings every shading shader uses, and the one BRDF. Included by
// scene.hlsl and rt_primary.hlsl so the raster and traced paths can never
// drift apart. The including shader declares its own push constants.

static const float kPi = 3.14159265f;

// ObjectData.flags bits.
static const uint kFlagAlphaMasked = 1u;
static const uint kFlagTransparent = 2u;
static const uint kFlagReflective = 4u; // per-object RT: fragments trace a reflection ray

// Explicitly column_major: dxc does NOT apply the cbuffer default to
// matrices inside structured buffers, so an unqualified float4x4 reads
// transposed. One region per frame slot, CPU-written after waitFrameSlot.
// The ray-generation axes are premultiplied: right *= tan(fov/2)*aspect,
// up *= tan(fov/2); position.w = per-pixel ray-cone spread angle.
struct CameraData {
    column_major float4x4 viewProj;
    float4 position;
    float4 rightAxis;
    float4 upAxis;
    float4 forwardAxis;
};

struct ObjectData {
    uint textureIndex; // into the bindless texture array
    uint normalIndex;  // 0 = no normal map (use the vertex normal)
    uint mrIndex;      // 0 = factors only (glTF: B = metallic, G = roughness)
    uint flags;        // kFlag* bits above
    float alphaCutoff;
    float baseAlpha; // baseColorFactor.a: blend opacity multiplier
    float metallicFactor;
    float roughnessFactor;
};

struct LightData {
    column_major float4x4 cascadeViewProj[4]; // light-space transforms per cascade
    float4 splitDepths; // view-space depth where each cascade ends
    float3 direction;   // world space, from the light toward the scene
    float intensity;
    float3 color;
    float pcfRadius; // filter radius in shadow-map texels (0 = hard 2x2)
    float biasBase;  // depth-compare bias floor; slope-scaled up to 8x
    float mapSize;   // shadow map resolution (texel size = 1/mapSize)
    uint cascadeCount;
    uint debugTint; // non-zero: tint output by cascade for inspection
    uint rtShadows; // non-zero: trace shadow rays instead of sampling maps
    uint reflections; // kReflection* below: reflective objects' source
    uint pad3;
    uint pad4;
};

// LightData.reflections values: where reflective-tagged fragments get
// their reflected color from.
static const uint kReflectionProbe = 0u;  // sample the probe cubemap
static const uint kReflectionTraced = 1u; // fire an inline reflection ray
static const uint kReflectionNone = 2u;   // shade plain (probe capture pass)

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects;
[[vk::binding(1, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState linearSampler;
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras;
[[vk::binding(7, 0)]] StructuredBuffer<LightData> lights;
// Reflection probe cubemap (load-time capture, full mip chain). Only
// sampled when reflections == kReflectionProbe, so the binding may stay
// unwritten on scenes that never captured one (partially bound).
[[vk::binding(18, 0)]] TextureCube probeMap;

// One directional light, glTF metallic-roughness: Lambert diffuse + GGX
// specular (Smith-Schlick visibility, Schlick Fresnel). Scaled by pi so a
// white dielectric matches the old albedo*NdotL model's brightness.
float3 shadeSurface(float3 albedo, float metallic, float roughness, float3 n, float3 v, float3 l,
                    float3 lightColor, float intensity, float shadow) {
    const float ndotl = saturate(dot(n, l));
    if (ndotl <= 0.0f || shadow <= 0.0f) {
        return 0.0f;
    }
    const float3 h = normalize(v + l);
    const float ndotv = max(dot(n, v), 1.0e-4f);
    const float ndoth = saturate(dot(n, h));
    const float vdoth = saturate(dot(v, h));
    const float a = max(roughness * roughness, 1.0e-3f);
    const float a2 = a * a;
    const float dDenom = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    const float d = a2 / (kPi * dDenom * dDenom);
    const float k = a * 0.5f;
    const float gv = ndotv / (ndotv * (1.0f - k) + k);
    const float gl = ndotl / (ndotl * (1.0f - k) + k);
    const float3 f0 = lerp(0.04f, albedo, metallic);
    const float3 f = f0 + (1.0f - f0) * pow(1.0f - vdoth, 5.0f);
    const float3 specular = d * gv * gl * f / (4.0f * ndotv * ndotl + 1.0e-4f);
    const float3 diffuse = albedo * (1.0f - metallic) / kPi;
    return (diffuse + specular) * lightColor * (intensity * ndotl * shadow * kPi);
}

// Hemispherical ambient shared by both paths.
float3 ambientLight(float3 n) {
    return float3(0.30f, 0.32f, 0.36f) * (n.y * 0.2f + 0.5f);
}

// Probe tier: the reflected direction looks up the capture cubemap, with
// roughness selecting a blurrier mip — a cheap stand-in for a real GGX
// prefilter. Infinite probe: no parallax correction, static content.
float3 sampleReflectionProbe(float3 dir, float roughness) {
    float w, h, mips;
    probeMap.GetDimensions(0, w, h, mips);
    return probeMap.SampleLevel(linearSampler, dir, roughness * (mips - 1.0f)).rgb;
}

// Fresnel-weighted mix of a surface's base shading with its reflected
// color, attenuated by roughness (a rough "mirror" barely reflects). One
// helper for both sources so the probe and traced tiers can never drift.
float3 mixReflection(float3 baseColor, float3 reflected, float3 f0, float roughness,
                     float ndotv) {
    const float3 f = f0 + (1.0f - f0) * pow(1.0f - saturate(ndotv), 5.0f);
    return lerp(baseColor, reflected, f * (1.0f - roughness));
}
