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
            color += throughput * skyColor(dir);
            break;
        }

        const TracedHit hit = shadeCommittedHit(
            q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex(),
            q.CommittedTriangleBarycentrics(), origin, dir, q.CommittedRayT(),
            travelled + q.CommittedRayT(), cam.position.w, light);
        travelled += q.CommittedRayT();

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
    return float4(color, 1.0f);
}
