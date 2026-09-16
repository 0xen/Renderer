// Scene pass: geometry from the memory pool via indirect draws, camera via
// a per-frame-slot buffer (so a moving camera never touches the static
// command buffers — only the slot index is baked as a push constant),
// materials through the bindless table. Each indirect entry's
// firstInstance is its base row in the instance-row table (dxc maps
// SV_InstanceID to SPIR-V InstanceIndex, which includes firstInstance);
// the row resolves to the object/material index and the transform index,
// so instanced draws (instanceCount > 1) share geometry and materials.

#include "backend.hlsli"

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera buffer
    uint cascade;    // used by the shadow pass; 0 here
};
REND_PUSH(PushConstants, pc);

// Data layouts, common bindings (0-2, 6, 7) and the BRDF live in the
// shared include so the raster and traced paths can never drift apart.
#include "shading.hlsli"

// Cascade shadows, volumetric fog and the cotangent-frame normal map
// (plus bindings 8/9) are shared with the deferred lighting pass.
#include "lighting.hlsli"
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
    const InstanceRow row = instanceRows[rendInstanceIndex(input.instanceId)];
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
    output.position = rendClip(output.position);
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const ObjectData object = objects[input.objectIndex];
    float4 albedo =
        textures[NonUniformResourceIndex(object.textureIndex)].Sample(linearSampler, input.uv);
    if ((object.flags & kFlagAlphaMasked) != 0 && albedo.a < object.alphaCutoff) {
        discard;
    }
    albedo.rgb *= object.baseColor.rgb;

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
    // A zero-intensity sun (scripted night) contributes nothing — skip
    // its shadow work entirely instead of tracing/sampling for it.
    const float direct = light.intensity > 0.0f ? saturate(dot(n, l)) : 0.0f;
    uint cascade = 0;
    float3 shadow = 1.0f;
#if RT_SHADOWS
    if (light.rtShadows != 0) {
        // Traced shadows are per-channel: transparent occluders tint the
        // light that passes through them instead of blocking it.
        shadow = direct > 0.0f ? shadowRay(input.worldPos, n, light) : (float3)0.0f;
    } else
#endif
    if (light.cascadeCount > 0) {
        shadow = direct > 0.0f
                     ? shadowFactor(input.worldPos, n, input.viewDepth, light, cascade).xxx
                     : (float3)0.0f;
    }

    const float3 v = normalize(cameras[pc.cameraSlot].position.xyz - input.worldPos);
    const float3 sun = shadeSurface(albedo.rgb, metallic, roughness, n, v, l, light.color,
                                    light.intensity, shadow);
    float3 color = albedo.rgb * ambientLight(n, light.ambientColor.rgb) + sun;
    // Point lights follow the shadow technique offer: traced when ray
    // traced shadows are selected (one short distance-bounded ray per
    // contributing light), unshadowed otherwise (no maps exist for them).
#if RT_SHADOWS
    if (light.rtShadows != 0) {
        color +=
            shadePointLightsTraced(albedo.rgb, metallic, roughness, n, v, input.worldPos, light);
    } else
#endif
    {
        color += shadePointLights(albedo.rgb, metallic, roughness, n, v, input.worldPos, light);
    }
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
    // Volumetric fog last: attenuates everything shaded above and adds the
    // in-scattered light between the camera and this fragment. Transparent
    // fragments get the same treatment — their fogged color is then scaled
    // by the blend factor (slight in-scatter double-count on glass,
    // accepted for this slice).
    if (light.fogColor.w > 0.0f) {
        color = applyFog(color, cameras[pc.cameraSlot].position.xyz, input.worldPos,
                         input.viewDepth, input.position.xy, light);
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
