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

// One dynamic point light (no shadows): position + falloff radius,
// color + intensity. Rides inside LightData so per-frame CPU updates
// stay a single host-visible memcpy — nothing baked into recordings.
static const uint kMaxPointLights = 16;
struct PointLight {
    float4 positionRadius; // xyz = world position, w = falloff radius
    float4 colorIntensity; // rgb = color, w = intensity (0 = off)
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
    // Volumetric fog box (scene <Fog>): world-space AABB of the media.
    // fogColor.w is the raymarch step count and doubles as the enable
    // flag — 0 (the probe-capture regions' default) disables fog.
    float4 fogBoxMin; // xyz = box min corner, w = density (extinction/m)
    float4 fogBoxMax; // xyz = box max corner, w = anisotropy g (-1..1)
    float4 fogColor;  // rgb = scattering albedo, w = step count (0 = off)
    // Dynamic sky + ambient (day/night control): the sky pass paints
    // skyColor over background pixels every frame; ambientColor replaces
    // the old hardcoded hemisphere constant (its default matches it).
    float4 skyColor;     // rgb = background color, w = active point lights
    float4 ambientColor; // rgb = ambient tint, w = skybox day phase (<0 = flat sky)
    PointLight pointLights[kMaxPointLights];
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
// Per-object world transforms: one region per camera slot (frame slots +
// probe faces), kTransformCapacity entries each, indexed by the object
// index. Scene geometry is world-baked and rides identity; runtime-spawned
// models are placed and moved through these (renderer message queue).
static const uint kTransformCapacity = 4096;
[[vk::binding(19, 0)]] StructuredBuffer<column_major float4x4> objectTransforms;
// Instance rows: SV_InstanceID (which includes the draw's firstInstance)
// indexes here; the row names the object/material row and the transform
// row, so one indirect entry with instanceCount N draws N placements of
// the same geometry. Scene draws ride an identity prefix (row i = {i, i});
// rows past kInstanceRowCapacity are the cull pass's per-slot scratch
// regions — partially visible instanced draws point firstInstance there.
// Scalar members only — see the SkinVertex std430 layout gotcha.
struct InstanceRow {
    uint objectIndex;
    uint transformIndex;
};
[[vk::binding(20, 0)]] StructuredBuffer<InstanceRow> instanceRows;

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

// Hemispherical ambient shared by both paths; the tint rides LightData
// (default 0.30/0.32/0.36 — the old hardcoded constant) so scripts can
// dim it through the night.
float3 ambientLight(float3 n, float3 tint) {
    return tint * (n.y * 0.2f + 0.5f);
}

// Sum of the dynamic point lights (never shadowed): inverse-square with a
// smooth window to zero at each light's radius, through the same BRDF as
// the sun. skyColor.w carries the active count, so scenes without point
// lights skip the loop entirely.
float3 shadePointLights(float3 albedo, float metallic, float roughness, float3 n, float3 v,
                        float3 worldPos, LightData light) {
    const uint count = min((uint)light.skyColor.w, kMaxPointLights);
    float3 sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        const PointLight pl = light.pointLights[i];
        if (pl.colorIntensity.w <= 0.0f) {
            continue;
        }
        const float3 toLight = pl.positionRadius.xyz - worldPos;
        const float d2 = dot(toLight, toLight);
        const float radius = max(pl.positionRadius.w, 1.0e-2f);
        if (d2 >= radius * radius) {
            continue;
        }
        const float d = sqrt(max(d2, 1.0e-4f));
        // Windowed inverse square: physical near the light, rolled
        // smoothly to zero at the radius so the reach never pops.
        const float ratio2 = d2 / (radius * radius);
        float window = saturate(1.0f - ratio2 * ratio2);
        window *= window;
        const float atten = window / max(d2, 1.0e-2f);
        sum += shadeSurface(albedo, metallic, roughness, n, v, toLight / d,
                            pl.colorIntensity.rgb, pl.colorIntensity.w * atten, 1.0f);
    }
    return sum;
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
