// Ray-traced primary visibility (RayQuery devices, ps_6_5): one fullscreen
// triangle; every fragment fires a camera ray through the scene TLAS
// instead of rasterizing the geometry. Hits pull their own triangle
// attributes from the geometry pool (bindings 11/12), shade with the same
// sun + ambient model as scene.hlsl, and fire the shadow ray from the hit
// point. Alpha-masked geometries are non-opaque in the BLAS, so the
// candidate loop alpha-tests them (correct foliage silhouettes, unlike the
// FORCE_OPAQUE shadow rays). Transparent (alpha-blend) surfaces commit,
// contribute opacity-weighted shading, and the ray marches on through them
// with the remaining throughput — front-to-back, no sorting needed.
// Reflective-flagged objects (per-object RT) get one extra reflection
// bounce, Fresnel-mixed into their shading.

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera/light buffers
    uint pad;
};
[[vk::push_constant]] PushConstants pc;

#include "shading.hlsli"
#include "rt_common.hlsli"
#include "lighting.hlsli" // applyFog + the cascade maps its march samples

struct VSOutput {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0; // Vulkan NDC: x right, y down
};

VSOutput VSMain(uint id : SV_VertexID) {
    VSOutput output;
    const float2 pos = float2(id == 1 ? 3.0f : -1.0f, id == 2 ? 3.0f : -1.0f);
    output.position = float4(pos, 0.0f, 1.0f);
    output.ndc = pos;
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const CameraData cam = cameras[pc.cameraSlot];
    const LightData light = lights[pc.cameraSlot];

    float3 origin = cam.position.xyz;
    // Vulkan NDC y points down, camera up points up.
    float3 dir = normalize(cam.forwardAxis.xyz + cam.rightAxis.xyz * input.ndc.x -
                           cam.upAxis.xyz * input.ndc.y);

    float3 color = 0.0f;
    float3 throughput = 1.0f;
    float travelled = 0.0f; // cone width grows across transparency steps
    float fogDistance = 0.0f; // camera -> final surface (or sky) along dir
    [loop] for (uint step = 0; step < kMaxTransparencySteps; ++step) {
        RayQuery<RAY_FLAG_NONE> q;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 1.0e-3f;
        ray.TMax = 1.0e4f;
        q.TraceRayInline(sceneBVH, RAY_FLAG_NONE, 0xff, ray);
        resolveCandidates(q);

        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            // Flat scene sky (matches the raster sky pass, which fog
            // scenes set to the converged in-scatter color); the day-phase
            // skybox mode keeps the traced gradient as before.
            color += throughput * (light.ambientColor.w < 0.0f ? light.skyColor.rgb
                                                               : skyColor(dir));
            fogDistance = travelled + ray.TMax;
            break;
        }

        const TracedHit hit = shadeCommittedHit(
            hitObjectIndex(q.CommittedInstanceID(), q.CommittedGeometryIndex()),
            q.CommittedObjectToWorld3x4(), q.CommittedPrimitiveIndex(),
            q.CommittedTriangleBarycentrics(), origin, dir, q.CommittedRayT(),
            travelled + q.CommittedRayT(), cam.position.w, light);
        travelled += q.CommittedRayT();
        fogDistance = travelled;

        float3 shaded = hit.color;
        if ((hit.flags & kFlagReflective) != 0) {
            shaded = applyReflection(hit, dir, travelled, cam.position.w, light);
        }

        if ((hit.flags & kFlagTransparent) != 0) {
            color += throughput * hit.opacity * shaded;
            throughput *= 1.0f - hit.opacity;
            if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01f) {
                break;
            }
            origin = hit.position + dir * 1.0e-3f;
            continue;
        }
        color += throughput * shaded;
        break;
    }
    // Same march the raster paths run last (deferred.hlsl/scene.hlsl); the
    // segment spans every transparency step since dir never changes. Fog on
    // transparent surfaces is throughput-approximate exactly like raster.
    // The viewer keeps the cascade passes recorded under traced primary
    // when the scene has fog (DrawBatch.fogCascades) so the march's shadow
    // taps read live maps.
    if (light.fogColor.w > 0.0f) {
        const float3 camPos = cam.position.xyz;
        const float viewDepth = dot(dir, cam.forwardAxis.xyz) * fogDistance;
        color = applyFog(color, camPos, camPos + dir * fogDistance, viewDepth,
                         input.position.xy, light);
    }
    return float4(color, 1.0f);
}
