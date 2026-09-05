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

struct PushConstants {
    uint cameraSlot; // frame-in-flight index into the camera/light buffers
    uint pad;
};
[[vk::push_constant]] PushConstants pc;

// Must match CameraData in the viewer / scene.hlsl. The ray-generation
// axes are premultiplied: right *= tan(fov/2)*aspect, up *= tan(fov/2).
struct CameraData {
    column_major float4x4 viewProj;
    float4 position;
    float4 rightAxis;
    float4 upAxis;
    float4 forwardAxis;
};
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras;

struct ObjectData {
    uint textureIndex; // into the bindless texture array
    uint flags;        // kFlagAlphaMasked | kFlagTransparent
    float alphaCutoff;
    float baseAlpha; // baseColorFactor.a: blend opacity multiplier
};

// Must match LightData in the viewer / scene.hlsl.
struct LightData {
    column_major float4x4 cascadeViewProj[4];
    float4 splitDepths;
    float3 direction;
    float intensity;
    float3 color;
    float pcfRadius;
    float biasBase;
    float mapSize;
    uint cascadeCount;
    uint debugTint;
    uint rtShadows;
    uint pad2;
    uint pad3;
    uint pad4;
};

[[vk::binding(0, 0)]] StructuredBuffer<ObjectData> objects;
[[vk::binding(1, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] SamplerState linearSampler;
[[vk::binding(7, 0)]] StructuredBuffer<LightData> lights;
[[vk::binding(10, 0)]] RaytracingAccelerationStructure sceneBVH;
// The whole geometry pool as raw bytes: interleaved vertices (stride 32:
// pos3f/normal3f/uv2f) and uint32 indices, addressed via geometryInfo.
[[vk::binding(11, 0)]] ByteAddressBuffer geometryBytes;
// Per BLAS-geometry index (== object index): x = firstIndex (uint32 units
// from pool start), y = vertexOffset (vertex-stride units from pool start).
[[vk::binding(12, 0)]] StructuredBuffer<uint2> geometryInfo;

static const uint kFlagAlphaMasked = 1u;
static const uint kFlagTransparent = 2u;
static const uint kVertexStrideBytes = 32u;
static const uint kMaxTransparencySteps = 4u;

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

float3 vertexNormal(uint vertex) {
    return asfloat(geometryBytes.Load3(vertex * kVertexStrideBytes + 12));
}

float2 vertexUv(uint vertex) {
    return asfloat(geometryBytes.Load2(vertex * kVertexStrideBytes + 24));
}

uint3 triangleIndices(uint2 info, uint primitive) {
    return geometryBytes.Load3((info.x + primitive * 3) * 4) + info.y;
}

float2 hitUv(uint objectIndex, uint primitive, float2 bary) {
    const uint3 tri = triangleIndices(geometryInfo[objectIndex], primitive);
    return vertexUv(tri.x) * (1.0f - bary.x - bary.y) + vertexUv(tri.y) * bary.x +
           vertexUv(tri.z) * bary.y;
}

// Candidate filter shared by every trace here: transparent surfaces always
// commit (the outer march blends through them); alpha-masked ones commit
// only where the texture passes the cutoff.
void resolveCandidates(inout RayQuery<RAY_FLAG_NONE> q) {
    while (q.Proceed()) {
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }
        const uint objectIndex = q.CandidateGeometryIndex();
        const ObjectData object = objects[objectIndex];
        if ((object.flags & kFlagTransparent) != 0) {
            q.CommitNonOpaqueTriangleHit();
            continue;
        }
        const float2 uv =
            hitUv(objectIndex, q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics());
        const float alpha = textures[NonUniformResourceIndex(object.textureIndex)]
                                .SampleLevel(linearSampler, uv, 0)
                                .a;
        if (alpha >= object.alphaCutoff) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
}

// Same one-ray sun visibility as the hybrid shadow path (alpha-masked
// casters count as solid — the shared any-hit refinement comes later).
float shadowRay(float3 worldPos, float3 n, LightData light) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray;
    ray.Origin = worldPos + n * 0.02f;
    ray.Direction = -light.direction;
    ray.TMin = 0.0f;
    ray.TMax = 1.0e4f;
    q.TraceRayInline(sceneBVH, RAY_FLAG_NONE, 0xff, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0f : 1.0f;
}

float3 skyColor(float3 dir) {
    // Subtle zenith-to-horizon gradient standing in for the raster path's
    // clear color until a real sky exists.
    return lerp(float3(0.10f, 0.11f, 0.14f), float3(0.02f, 0.02f, 0.04f),
                saturate(dir.y * 0.5f + 0.5f));
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

        const uint objectIndex = q.CommittedGeometryIndex();
        const ObjectData object = objects[objectIndex];
        const float2 bary = q.CommittedTriangleBarycentrics();
        const uint3 tri = triangleIndices(geometryInfo[objectIndex], q.CommittedPrimitiveIndex());
        const float w0 = 1.0f - bary.x - bary.y;
        float3 n = normalize(vertexNormal(tri.x) * w0 + vertexNormal(tri.y) * bary.x +
                             vertexNormal(tri.z) * bary.y);
        if (dot(n, dir) > 0.0f) {
            n = -n; // shade the side the ray sees (geometry is unculled)
        }
        const float2 uv =
            vertexUv(tri.x) * w0 + vertexUv(tri.y) * bary.x + vertexUv(tri.z) * bary.y;
        const float3 hitPos = origin + dir * q.CommittedRayT();

        const float4 albedo =
            textures[NonUniformResourceIndex(object.textureIndex)].SampleLevel(linearSampler, uv, 0);
        const float direct = saturate(dot(n, -normalize(light.direction)));
        const float shadow = direct > 0.0f ? shadowRay(hitPos, n, light) : 0.0f;
        const float3 sun = light.color * (light.intensity * direct * shadow);
        const float3 ambient = float3(0.30f, 0.32f, 0.36f) * (n.y * 0.2f + 0.5f);
        const float3 shaded = albedo.rgb * (ambient + sun);

        if ((object.flags & kFlagTransparent) != 0) {
            const float opacity = saturate(albedo.a * object.baseAlpha);
            color += throughput * opacity * shaded;
            throughput *= 1.0f - opacity;
            if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01f) {
                break;
            }
            origin = hitPos + dir * 1.0e-3f;
            continue;
        }
        color += throughput * shaded;
        break;
    }
    return float4(color, 1.0f);
}
