// Shared raster-lighting helpers split out of scene.hlsl so the forward
// pass (scene.hlsl) and the deferred lighting pass (deferred.hlsl) can
// never drift apart: cascade shadow sampling, volumetric fog, and the
// screen-space cotangent-frame normal map. Include AFTER shading.hlsli
// (LightData/CameraData/bindings) — this file declares the shadow-map
// bindings 8/9 itself.

#include "backend.hlsli"

[[vk::binding(8, 0)]] Texture2D<float> shadowMaps[4] REND_T(8);
[[vk::binding(9, 0)]] SamplerComparisonState shadowSampler REND_S(9);

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

// One-tap cascade visibility for a point in AIR: the cascade is picked by
// the sample's own view depth, and with no surface normal there is no
// slope-scaled bias — the flat biasBase floor suffices for volume samples.
// Out-of-map samples read the border (opaque white = lit), so fog past the
// cascade reach is simply unshadowed.
float fogShadow(float3 pos, float viewDepth, LightData light) {
    if (light.cascadeCount == 0) {
        return 1.0f;
    }
    uint cascade = light.cascadeCount - 1;
    for (uint i = 0; i < light.cascadeCount; ++i) {
        if (viewDepth < light.splitDepths[i]) {
            cascade = i;
            break;
        }
    }
    const float4 lightClip = mul(light.cascadeViewProj[cascade], float4(pos, 1.0f));
    const float2 uv = lightClip.xy * 0.5f + 0.5f;
    return shadowMaps[NonUniformResourceIndex(cascade)].SampleCmpLevelZero(
        shadowSampler, uv, lightClip.z - light.biasBase);
}

// Henyey-Greenstein phase function, normalized over the sphere. g > 0
// scatters forward (halos around the sun direction), 0 is isotropic.
float fogPhase(float cosTheta, float g) {
    const float g2 = g * g;
    const float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * kPi * pow(max(denom, 1.0e-4f), 1.5f));
}

// Volumetric fog: march the camera->fragment segment through the scene's
// fog box accumulating Beer-Lambert transmittance and single-scattered,
// cascade-shadowed sunlight — geometry that blocks the sun carves visible
// shafts out of the fog. Runs inside the shading fragment shader (forward
// scene pass or deferred lighting pass): no extra pass, no depth readback,
// nothing baked into static recordings (the box rides the per-slot light
// buffer), and the cost scales with resolution x step count, never with
// scene complexity.
float3 applyFog(float3 color, float3 camPos, float3 worldPos, float viewDepth, float2 pixel,
                LightData light) {
    const float3 toFrag = worldPos - camPos;
    const float dist = length(toFrag);
    const float3 rayDir = toFrag / max(dist, 1.0e-4f);

    // Clip the segment to the fog box (slab test).
    const float3 invDir = 1.0f / select(abs(rayDir) > 1.0e-6f, rayDir, 1.0e-6f);
    const float3 tLo = (light.fogBoxMin.xyz - camPos) * invDir;
    const float3 tHi = (light.fogBoxMax.xyz - camPos) * invDir;
    const float3 tMin3 = min(tLo, tHi);
    const float3 tMax3 = max(tLo, tHi);
    const float tEnter = max(max(tMin3.x, tMin3.y), max(tMin3.z, 0.0f));
    const float tExit = min(min(tMax3.x, tMax3.y), min(tMax3.z, dist));
    if (tExit <= tEnter) {
        return color;
    }

    const uint steps = (uint)light.fogColor.w;
    const float dt = (tExit - tEnter) / steps;
    // Interleaved-gradient jitter per pixel: turns step banding into
    // stable fine-grained noise (no temporal component, no TAA needed).
    const float jitter =
        frac(52.9829189f * frac(dot(pixel, float2(0.06711056f, 0.00583715f))));

    const float sigmaT = light.fogBoxMin.w;
    const float3 sigmaS = sigmaT * light.fogColor.rgb;
    const float3 sunDir = -normalize(light.direction);
    const float phase = fogPhase(dot(rayDir, sunDir), light.fogBoxMax.w);
    const float3 sunLight = light.color * light.intensity;
    // Isotropic ambient in-scatter keeps shadowed fog from going black;
    // matches ambientLight's mid-hemisphere value (and dims with it).
    const float3 ambient = light.ambientColor.rgb * 0.5f;
    const float stepTrans = exp(-sigmaT * dt);

    float transmittance = 1.0f;
    float3 inscatter = 0.0f;
    float t = tEnter + jitter * dt;
    for (uint i = 0; i < steps; ++i) {
        const float3 p = camPos + rayDir * t;
        const float shadow = fogShadow(p, viewDepth * (t / dist), light);
        const float3 scattered = sigmaS * (sunLight * (shadow * phase) + ambient);
        // Analytic integral of the source term across the step keeps the
        // result stable at low step counts.
        inscatter += scattered * (transmittance * (1.0f - stepTrans) / max(sigmaT, 1.0e-4f));
        transmittance *= stepTrans;
        t += dt;
    }
    return color * transmittance + inscatter;
}

// Tangent-space normal map applied via the screen-space cotangent frame
// (Schueler): derivatives of position and uv rebuild the tangent basis, so
// the vertex layout needs no baked tangents. Derivative-dependent — only
// callable while rasterizing the surface's own triangle (forward scene
// pass or the deferred G-buffer pass, never a fullscreen pass).
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
