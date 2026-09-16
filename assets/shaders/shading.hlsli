// Shared surface-shading vocabulary for the scene passes: the bindless
// data layouts (must match the viewer structs byte for byte), the common
// bindings every shading shader uses, and the one BRDF. Included by
// scene.hlsl and rt_primary.hlsl so the raster and traced paths can never
// drift apart. The including shader declares its own push constants.

#include "backend.hlsli"

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
    // baseColorFactor.rgb (w = 1): multiplies the albedo texture, and
    // tints traced transmission through transparent surfaces. The float4
    // sits at offset 32, so C++ and std430 strides agree (48 B).
    float4 baseColor;
};

// One dynamic point light: position + falloff radius, color + intensity,
// and shadow state. Rides inside LightData so per-frame CPU updates stay
// a single host-visible memcpy — nothing baked into recordings.
static const uint kMaxPointLights = 16;
struct PointLight {
    float4 positionRadius; // xyz = world position, w = falloff radius
    float4 colorIntensity; // rgb = color, w = intensity (0 = off)
    // x = castsShadows (traced paths fire an occlusion ray), y = a baked
    // shadow cube exists at this light's slot in pointShadowMaps (the
    // raster path samples it), zw spare.
    float4 params;
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
    float exposure;   // post pass: scene-color multiplier (1 = neutral)
    uint tonemap;     // post pass: 0 = clamp only, 1 = ACES fitted
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

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects REND_B(0);
[[vk::binding(1, 0)]] Texture2D textures[] REND_T(1);
[[vk::binding(2, 0)]] SamplerState linearSampler REND_S(2);
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras REND_B(6);
[[vk::binding(7, 0)]] StructuredBuffer<LightData> lights REND_B(7);
// Reflection probe cubemap (load-time capture, full mip chain). Only
// sampled when reflections == kReflectionProbe, so the binding may stay
// unwritten on scenes that never captured one (partially bound).
[[vk::binding(18, 0)]] TextureCube probeMap REND_T(18);
// Point-light shadow cubes (load-time capture, one per shadow-casting
// light slot): R32F world distance from the light. Sampled only when the
// light's params.y flags a written slot (partially bound).
[[vk::binding(27, 0)]] TextureCube pointShadowMaps[kMaxPointLights] REND_T(27);
// Per-object world transforms: one region per camera slot (frame slots +
// probe faces), kTransformCapacity entries each, indexed by the object
// index. Scene geometry is world-baked and rides identity; runtime-spawned
// models are placed and moved through these (renderer message queue).
static const uint kTransformCapacity = 4096;
[[vk::binding(19, 0)]] StructuredBuffer<column_major float4x4> objectTransforms REND_B(19);
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
[[vk::binding(20, 0)]] REND_SHARED_BUFFER(InstanceRow) instanceRows REND_U(20);

// One directional light, glTF metallic-roughness: Lambert diffuse + GGX
// specular (Smith-Schlick visibility, Schlick Fresnel). Scaled by pi so a
// white dielectric matches the old albedo*NdotL model's brightness.
// shadow is a per-channel visibility (traced shadow rays tint through
// transparent occluders; the cascade path broadcasts its scalar).
float3 shadeSurface(float3 albedo, float metallic, float roughness, float3 n, float3 v, float3 l,
                    float3 lightColor, float intensity, float3 shadow) {
    const float ndotl = saturate(dot(n, l));
    if (ndotl <= 0.0f || max(shadow.x, max(shadow.y, shadow.z)) <= 0.0f) {
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

// Geometry + attenuation of one point light at a surface point: unit
// direction and distance to the light, and the windowed inverse-square
// attenuation (physical near the light, rolled smoothly to zero at the
// radius so the reach never pops). atten 0 = off or out of range. Shared
// by the unshadowed sum below and the traced variant in rt_common.hlsli.
struct PointLightSample {
    float3 l;
    float dist;
    float atten;
};

PointLightSample samplePointLight(PointLight pl, float3 worldPos) {
    PointLightSample s;
    s.l = 0.0f;
    s.dist = 0.0f;
    s.atten = 0.0f;
    if (pl.colorIntensity.w <= 0.0f) {
        return s;
    }
    const float3 toLight = pl.positionRadius.xyz - worldPos;
    const float d2 = dot(toLight, toLight);
    const float radius = max(pl.positionRadius.w, 1.0e-2f);
    if (d2 >= radius * radius) {
        return s;
    }
    s.dist = sqrt(max(d2, 1.0e-4f));
    s.l = toLight / s.dist;
    const float ratio2 = d2 / (radius * radius);
    float window = saturate(1.0f - ratio2 * ratio2);
    window *= window;
    s.atten = window / max(d2, 1.0e-2f);
    return s;
}

// Sum of the dynamic point lights through the same BRDF as the sun.
// Shadow-casting lights with a baked cube (params.y) compare the
// fragment's distance against the stored light-to-occluder distance —
// the raster shadow tier. skyColor.w carries the active count, so scenes
// without point lights skip the loop entirely.
float3 shadePointLights(float3 albedo, float metallic, float roughness, float3 n, float3 v,
                        float3 worldPos, LightData light) {
    const uint count = min((uint)light.skyColor.w, kMaxPointLights);
    float3 sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        const PointLight pl = light.pointLights[i];
        const PointLightSample s = samplePointLight(pl, worldPos);
        if (s.atten <= 0.0f) {
            continue;
        }
        if (pl.params.x > 0.0f && pl.params.y > 0.0f) {
            const float3 fromLight = worldPos - pl.positionRadius.xyz;
            const float stored =
                pointShadowMaps[NonUniformResourceIndex(i)]
                    .SampleLevel(linearSampler, fromLight, 0.0f)
                    .r;
            // Distance-space bias: flat floor + a slope with range (the
            // cube face's texel footprint grows with distance).
            if (s.dist - (0.05f + s.dist * 0.02f) > stored) {
                continue;
            }
        }
        sum += shadeSurface(albedo, metallic, roughness, n, v, s.l, pl.colorIntensity.rgb,
                            pl.colorIntensity.w * s.atten, 1.0f);
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
