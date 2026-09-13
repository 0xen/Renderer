// FXAA anti-aliasing module (post-chain stage): one fullscreen triangle
// sampling the LDR intermediate (binding 36 — the post pass's tonemapped
// output) and writing the anti-aliased result to the sRGB swapchain.
// FXAA 3.11-style quality path: luma edge detection, edge-direction
// classification, an 8-step end-of-edge search for the span-based blend
// factor, plus the sub-pixel low-pass. All taps use SampleLevel(0) — the
// bindless sampler (binding 2) is REPEAT + anisotropic, so mip level 0 is
// forced and coordinates are clamped half a texel inside the image.
// Non-temporal and deterministic: static frames stay bit-stable.

struct PushConstants {
    uint cameraSlot; // unused; layout shared with the other post passes
    uint cascade;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(2, 0)]] SamplerState linearSampler;
[[vk::binding(36, 0)]] Texture2D<float4> ldrColor;

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VSOutput output;
    output.position = float4(corners[vertexId], 1.0f, 1.0f);
    output.uv = corners[vertexId] * 0.5f + 0.5f; // +Y-down clip = +V-down uv, already aligned
    return output;
}

static const float kEdgeThreshold = 1.0f / 8.0f;    // relative contrast to bother
static const float kEdgeThresholdMin = 1.0f / 24.0f; // absolute floor (darks)
static const float kSubpixQuality = 0.75f;
static const uint kSearchSteps = 8;
static const float kSearchStepSizes[kSearchSteps] = {1.0f, 1.5f, 2.0f, 2.0f,
                                                     2.0f, 2.0f, 4.0f, 8.0f};

// Perceptual luma from the sampled LINEAR values (the sRGB view decodes):
// sqrt approximates the gamma the algorithm was tuned for.
float lumaAt(float2 uv, float2 clampMin, float2 clampMax) {
    const float3 rgb = ldrColor.SampleLevel(linearSampler, clamp(uv, clampMin, clampMax), 0.0f).rgb;
    return sqrt(dot(rgb, float3(0.299f, 0.587f, 0.114f)));
}

float4 PSMain(VSOutput input) : SV_Target0 {
    float2 dims;
    ldrColor.GetDimensions(dims.x, dims.y);
    const float2 texel = 1.0f / dims;
    // Keep every tap inside the image — the shared sampler wraps.
    const float2 clampMin = 0.5f * texel;
    const float2 clampMax = 1.0f - 0.5f * texel;
    const float2 uv = input.uv;

    const float4 center = ldrColor.SampleLevel(linearSampler, clamp(uv, clampMin, clampMax), 0.0f);
    const float lumaM = sqrt(dot(center.rgb, float3(0.299f, 0.587f, 0.114f)));
    const float lumaN = lumaAt(uv + float2(0.0f, -texel.y), clampMin, clampMax);
    const float lumaS = lumaAt(uv + float2(0.0f, texel.y), clampMin, clampMax);
    const float lumaW = lumaAt(uv + float2(-texel.x, 0.0f), clampMin, clampMax);
    const float lumaE = lumaAt(uv + float2(texel.x, 0.0f), clampMin, clampMax);

    const float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    const float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    const float range = lumaMax - lumaMin;
    if (range < max(kEdgeThresholdMin, lumaMax * kEdgeThreshold)) {
        return center; // flat neighborhood — untouched
    }

    const float lumaNW = lumaAt(uv + float2(-texel.x, -texel.y), clampMin, clampMax);
    const float lumaNE = lumaAt(uv + float2(texel.x, -texel.y), clampMin, clampMax);
    const float lumaSW = lumaAt(uv + float2(-texel.x, texel.y), clampMin, clampMax);
    const float lumaSE = lumaAt(uv + float2(texel.x, texel.y), clampMin, clampMax);

    // Horizontal vs vertical edge from second-derivative energy.
    const float edgeH = abs(lumaNW + lumaNE - 2.0f * lumaN) +
                        2.0f * abs(lumaW + lumaE - 2.0f * lumaM) +
                        abs(lumaSW + lumaSE - 2.0f * lumaS);
    const float edgeV = abs(lumaNW + lumaSW - 2.0f * lumaW) +
                        2.0f * abs(lumaN + lumaS - 2.0f * lumaM) +
                        abs(lumaNE + lumaSE - 2.0f * lumaE);
    const bool horizontal = edgeH >= edgeV;

    // The edge sits between this pixel and the brighter-contrast side.
    const float luma1 = horizontal ? lumaN : lumaW;
    const float luma2 = horizontal ? lumaS : lumaE;
    const float grad1 = luma1 - lumaM;
    const float grad2 = luma2 - lumaM;
    const bool steepest1 = abs(grad1) >= abs(grad2);
    const float gradScaled = 0.25f * max(abs(grad1), abs(grad2));
    const float stepLength = (horizontal ? texel.y : texel.x) * (steepest1 ? -0.5f : 0.5f);
    const float lumaLocalAvg = 0.5f * ((steepest1 ? luma1 : luma2) + lumaM);

    // Walk both directions along the edge to its ends.
    float2 edgeUv = uv;
    if (horizontal) {
        edgeUv.y += stepLength;
    } else {
        edgeUv.x += stepLength;
    }
    const float2 offset = horizontal ? float2(texel.x, 0.0f) : float2(0.0f, texel.y);
    float2 uv1 = edgeUv - offset;
    float2 uv2 = edgeUv + offset;
    float lumaEnd1 = lumaAt(uv1, clampMin, clampMax) - lumaLocalAvg;
    float lumaEnd2 = lumaAt(uv2, clampMin, clampMax) - lumaLocalAvg;
    bool reached1 = abs(lumaEnd1) >= gradScaled;
    bool reached2 = abs(lumaEnd2) >= gradScaled;
    for (uint i = 0; i < kSearchSteps && !(reached1 && reached2); ++i) {
        if (!reached1) {
            uv1 -= offset * kSearchStepSizes[i];
            lumaEnd1 = lumaAt(uv1, clampMin, clampMax) - lumaLocalAvg;
            reached1 = abs(lumaEnd1) >= gradScaled;
        }
        if (!reached2) {
            uv2 += offset * kSearchStepSizes[i];
            lumaEnd2 = lumaAt(uv2, clampMin, clampMax) - lumaLocalAvg;
            reached2 = abs(lumaEnd2) >= gradScaled;
        }
    }

    const float dist1 = horizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    const float dist2 = horizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);
    const bool closer1 = dist1 < dist2;
    const float distFinal = min(dist1, dist2);
    const float edgeLength = dist1 + dist2;
    // Only offset when this pixel sits on the darker/lighter side the
    // nearer end's contrast agrees with (the classic variation check).
    const bool isLumaMSmaller = lumaM < lumaLocalAvg;
    const bool correctVariation =
        ((closer1 ? lumaEnd1 : lumaEnd2) < 0.0f) != isLumaMSmaller;
    float pixelOffset =
        correctVariation ? max(0.0f, 0.5f - distFinal / max(edgeLength, 1e-6f)) : 0.0f;

    // Sub-pixel aliasing low-pass: how far the 3x3 average sits from this
    // pixel relative to the local range.
    const float lumaAvg = (2.0f * (lumaN + lumaS + lumaW + lumaE) +
                           lumaNW + lumaNE + lumaSW + lumaSE) /
                          12.0f;
    const float subpix = saturate(abs(lumaAvg - lumaM) / range);
    const float subpixOffset = subpix * subpix * (3.0f - 2.0f * subpix);
    pixelOffset = max(pixelOffset, subpixOffset * subpixOffset * kSubpixQuality * 0.5f);

    float2 finalUv = uv;
    if (horizontal) {
        finalUv.y += pixelOffset * 2.0f * stepLength;
    } else {
        finalUv.x += pixelOffset * 2.0f * stepLength;
    }
    return float4(
        ldrColor.SampleLevel(linearSampler, clamp(finalUv, clampMin, clampMax), 0.0f).rgb, 1.0f);
}
