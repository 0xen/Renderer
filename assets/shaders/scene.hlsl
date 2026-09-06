// Scene pass: geometry from the memory pool via indirect draws, camera via
// a per-frame-slot buffer (so a moving camera never touches the static
// command buffers — only the slot index is baked as a push constant),
// materials through the bindless table. Each indirect entry's
// firstInstance is its base row in the instance-row table (dxc maps
// SV_InstanceID to SPIR-V InstanceIndex, which includes firstInstance);
// the row resolves to the object/material index and the transform index,
// so instanced draws (instanceCount > 1) share geometry and materials.

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera buffer
    uint cascade;    // used by the shadow pass; 0 here
};
[[vk::push_constant]] PushConstants pc;

// Data layouts, common bindings (0-2, 6, 7) and the BRDF live in the
// shared include so the raster and traced paths can never drift apart.
#include "shading.hlsli"

[[vk::binding(8, 0)]] Texture2D<float> shadowMaps[4];
[[vk::binding(9, 0)]] SamplerComparisonState shadowSampler;
#if RT_SHADOWS
// Ray-query machinery (compiled only into scene_rt.frag.spv, used on
// RayQuery devices): the scene TLAS + geometry fetch for hybrid traced
// shadows (shadowRay) and per-object reflection rays (traceReflection).
#include "rt_common.hlsli"
#endif

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    uint instanceId : SV_InstanceID;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : POSITION1; // vertices are pre-baked world space
    float viewDepth : DEPTH0;    // clip w = distance along the view axis
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint objectIndex : OBJECT0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    const InstanceRow row = instanceRows[input.instanceId];
    const float4x4 world =
        objectTransforms[pc.cameraSlot * kTransformCapacity + row.transformIndex];
    const float3 worldPos = mul(world, float4(input.position, 1.0f)).xyz;
    output.position = mul(cameras[pc.cameraSlot].viewProj, float4(worldPos, 1.0f));
    output.worldPos = worldPos;
    output.viewDepth = output.position.w;
    // Uniform-scale transforms only (yaw + scale), so the upper 3x3 works
    // for normals after renormalization.
    output.normal = normalize(mul((float3x3)world, input.normal));
    output.uv = input.uv;
    output.objectIndex = row.objectIndex;
    return output;
}

// Visibility from one cascade's map: transform to its light space, 3x3
// PCF over hardware 2x2 compares. The projections bake Vulkan's Y flip,
// so NDC y maps straight to V.
float sampleCascade(float3 worldPos, float3 n, uint cascade, LightData light) {
    const float4 lightClip = mul(light.cascadeViewProj[cascade], float4(worldPos, 1.0f));
    const float2 uv = lightClip.xy * 0.5f + 0.5f;
    // Slope-scaled bias against acne on faces the light grazes.
    const float ndotl = saturate(dot(n, -light.direction));
    const float bias = clamp(light.biasBase / max(ndotl, 0.05f), light.biasBase,
                             light.biasBase * 8.0f);
    const float depth = lightClip.z - bias;
    if (light.pcfRadius <= 0.0f) {
        return shadowMaps[NonUniformResourceIndex(cascade)].SampleCmpLevelZero(shadowSampler, uv,
                                                                               depth);
    }
    const float texel = light.pcfRadius / light.mapSize;
    float sum = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            sum += shadowMaps[NonUniformResourceIndex(cascade)].SampleCmpLevelZero(
                shadowSampler, uv + float2(x, y) * texel, depth);
        }
    }
    return sum / 9.0f;
}

// Fraction of each cascade's depth range that cross-fades into the next
// cascade, hiding the resolution/bias jump at the split as it sweeps
// across geometry while the camera moves.
static const float kCascadeBlendFraction = 0.2f;

// Shadow visibility: pick the cascade by view depth, then blend toward
// the next cascade near the split so the transition never pops.
float shadowFactor(float3 worldPos, float3 n, float viewDepth, LightData light,
                   out uint cascade) {
    cascade = light.cascadeCount - 1;
    for (uint i = 0; i < light.cascadeCount; ++i) {
        if (viewDepth < light.splitDepths[i]) {
            cascade = i;
            break;
        }
    }
    float shadow = sampleCascade(worldPos, n, cascade, light);
    if (cascade + 1 < light.cascadeCount) {
        const float splitStart = cascade > 0 ? light.splitDepths[cascade - 1] : 0.0f;
        const float splitEnd = light.splitDepths[cascade];
        const float band = (splitEnd - splitStart) * kCascadeBlendFraction;
        const float fade = saturate((viewDepth - (splitEnd - band)) / band);
        if (fade > 0.0f) {
            shadow = lerp(shadow, sampleCascade(worldPos, n, cascade + 1, light), fade);
        }
    }
    return shadow;
}

// Tangent-space normal map applied via the screen-space cotangent frame
// (Schueler): derivatives of position and uv rebuild the tangent basis, so
// the vertex layout needs no baked tangents.
float3 applyNormalMap(uint normalIndex, float3 n, float3 worldPos, float2 uv) {
    if (normalIndex == 0) {
        return n;
    }
    const float3 dp1 = ddx(worldPos);
    const float3 dp2 = ddy(worldPos);
    const float2 duv1 = ddx(uv);
    const float2 duv2 = ddy(uv);
    const float3 dp2perp = cross(dp2, n);
    const float3 dp1perp = cross(n, dp1);
    float3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    const float invmax = rsqrt(max(dot(t, t), dot(b, b)) + 1.0e-12f);
    t *= invmax;
    b *= invmax;
    const float3 tn =
        textures[NonUniformResourceIndex(normalIndex)].Sample(linearSampler, uv).xyz * 2.0f -
        1.0f;
    return normalize(t * tn.x + b * tn.y + n * tn.z);
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const ObjectData object = objects[input.objectIndex];
    const float4 albedo =
        textures[NonUniformResourceIndex(object.textureIndex)].Sample(linearSampler, input.uv);
    if ((object.flags & kFlagAlphaMasked) != 0 && albedo.a < object.alphaCutoff) {
        discard;
    }

    const LightData light = lights[pc.cameraSlot];
    float3 n = normalize(input.normal);
    n = applyNormalMap(object.normalIndex, n, input.worldPos, input.uv);
    float metallic = object.metallicFactor;
    float roughness = object.roughnessFactor;
    if (object.mrIndex != 0) {
        const float2 mr =
            textures[NonUniformResourceIndex(object.mrIndex)].Sample(linearSampler, input.uv).gb;
        roughness *= mr.x;
        metallic *= mr.y;
    }
    const float3 l = -normalize(light.direction);
    const float direct = saturate(dot(n, l));
    uint cascade = 0;
    float shadow = 1.0f;
#if RT_SHADOWS
    if (light.rtShadows != 0) {
        shadow = direct > 0.0f ? shadowRay(input.worldPos, n, light) : 0.0f;
    } else
#endif
    if (light.cascadeCount > 0) {
        shadow = direct > 0.0f
                     ? shadowFactor(input.worldPos, n, input.viewDepth, light, cascade)
                     : 0.0f;
    }

    const float3 v = normalize(cameras[pc.cameraSlot].position.xyz - input.worldPos);
    const float3 sun = shadeSurface(albedo.rgb, metallic, roughness, n, v, l, light.color,
                                    light.intensity, shadow);
    float3 color = albedo.rgb * ambientLight(n) + sun;
    // Reflective-flagged fragments mix in a reflected color from the
    // technique the light buffer selects: the probe cubemap everywhere, or
    // (RT variant only) one traced reflection ray whose cost scales with
    // the flagged objects' screen coverage. kReflectionNone shades plain —
    // that's the probe capture pass seeing reflective objects as ordinary
    // surfaces instead of sampling the probe being rendered.
    if ((object.flags & kFlagReflective) != 0 && light.reflections != kReflectionNone) {
        float3 reflected = 0.0f;
        bool haveReflection = false;
#if RT_SHADOWS
        if (light.reflections == kReflectionTraced) {
            const float3 camPos = cameras[pc.cameraSlot].position.xyz;
            reflected = traceReflection(
                input.worldPos + n * 1.0e-3f, reflect(-v, n),
                length(input.worldPos - camPos), cameras[pc.cameraSlot].position.w, light);
            haveReflection = true;
        }
#endif
        if (!haveReflection) {
            reflected = sampleReflectionProbe(reflect(-v, n), roughness);
        }
        const float3 f0 = lerp(0.04f, albedo.rgb, metallic);
        color = mixReflection(color, reflected, f0, roughness, dot(n, v));
    }
    if (light.debugTint != 0) {
        const float3 tints[4] = {float3(1.0f, 0.6f, 0.6f), float3(0.6f, 1.0f, 0.6f),
                                 float3(0.6f, 0.6f, 1.0f), float3(1.0f, 1.0f, 0.6f)};
        color *= tints[cascade];
    }
    // Transparent objects draw in the blend pass, where alpha is the
    // blend factor (glTF blend / transmission approximated as opacity =
    // albedo.a * baseColorFactor.a). The opaque pipeline has blending
    // disabled, so alpha 1 elsewhere costs nothing.
    const float alpha = (object.flags & kFlagTransparent) != 0
                            ? saturate(albedo.a * object.baseAlpha)
                            : 1.0f;
    return float4(color, alpha);
}
