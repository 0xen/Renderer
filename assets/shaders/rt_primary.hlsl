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
// position.w = per-pixel ray-cone spread angle (traced texture LOD).
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
    uint normalIndex;  // 0 = no normal map (use the vertex normal)
    uint mrIndex;      // 0 = factors only (glTF: B = metallic, G = roughness)
    uint flags;        // kFlagAlphaMasked | kFlagTransparent
    float alphaCutoff;
    float baseAlpha; // baseColorFactor.a: blend opacity multiplier
    float metallicFactor;
    float roughnessFactor;
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
static const float kPi = 3.14159265f;

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

float3 vertexPosition(uint vertex) {
    return asfloat(geometryBytes.Load3(vertex * kVertexStrideBytes));
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

// Ray-cone texture LOD (rays have no screen derivatives, so mip selection
// is manual): the triangle's uv-per-world-unit density, in log2 space.
// Final mip = base + 0.5*log2(texel count) + log2(cone width), stretched
// at grazing incidence. Keeps distant surfaces from mip-0 moire/grain.
float triangleLodBase(uint3 tri) {
    const float3 e1 = vertexPosition(tri.y) - vertexPosition(tri.x);
    const float3 e2 = vertexPosition(tri.z) - vertexPosition(tri.x);
    const float2 d1 = vertexUv(tri.y) - vertexUv(tri.x);
    const float2 d2 = vertexUv(tri.z) - vertexUv(tri.x);
    const float worldArea = max(length(cross(e1, e2)), 1.0e-12f);
    const float uvArea = max(abs(d1.x * d2.y - d2.x * d1.y), 1.0e-12f);
    return 0.5f * log2(uvArea / worldArea);
}

float textureLod(uint textureIndex, float lodBase, float coneWidth, float ndotd) {
    float w, h;
    textures[NonUniformResourceIndex(textureIndex)].GetDimensions(w, h);
    return lodBase + 0.5f * log2(w * h) + log2(coneWidth / max(ndotd, 0.05f));
}

// Tangent-space normal map for a traced hit: no screen derivatives here,
// so the tangent frame comes from the triangle's edges and uv deltas.
float3 applyNormalMap(uint normalIndex, float3 n, uint3 tri, float2 uv, float lod) {
    if (normalIndex == 0) {
        return n;
    }
    const float3 e1 = vertexPosition(tri.y) - vertexPosition(tri.x);
    const float3 e2 = vertexPosition(tri.z) - vertexPosition(tri.x);
    const float2 duv1 = vertexUv(tri.y) - vertexUv(tri.x);
    const float2 duv2 = vertexUv(tri.z) - vertexUv(tri.x);
    const float det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) < 1.0e-8f) {
        return n;
    }
    const float r = 1.0f / det;
    float3 t = (e1 * duv2.y - e2 * duv1.y) * r;
    t = normalize(t - n * dot(n, t)); // orthonormalize against the normal
    const float3 b = cross(n, t) * (det < 0.0f ? -1.0f : 1.0f);
    const float3 tn = textures[NonUniformResourceIndex(normalIndex)]
                          .SampleLevel(linearSampler, uv, lod)
                          .xyz *
                          2.0f -
                      1.0f;
    return normalize(t * tn.x + b * tn.y + n * tn.z);
}

// Same BRDF as scene.hlsl: Lambert diffuse + GGX specular, glTF
// metallic-roughness parameterization, pi-scaled to match.
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
        travelled += q.CommittedRayT();

        // Manual mip selection from the pixel's ray cone at this distance.
        const float coneWidth = max(travelled * cam.position.w, 1.0e-6f);
        const float ndotd = abs(dot(n, dir));
        const float lodBase = triangleLodBase(tri);
        n = applyNormalMap(object.normalIndex, n, tri, uv,
                           textureLod(object.normalIndex, lodBase, coneWidth, ndotd));
        float metallic = object.metallicFactor;
        float roughness = object.roughnessFactor;
        if (object.mrIndex != 0) {
            const float2 mr =
                textures[NonUniformResourceIndex(object.mrIndex)]
                    .SampleLevel(linearSampler, uv,
                                 textureLod(object.mrIndex, lodBase, coneWidth, ndotd))
                    .gb;
            roughness *= mr.x;
            metallic *= mr.y;
        }
        const float4 albedo =
            textures[NonUniformResourceIndex(object.textureIndex)]
                .SampleLevel(linearSampler, uv,
                             textureLod(object.textureIndex, lodBase, coneWidth, ndotd));
        const float3 l = -normalize(light.direction);
        const float direct = saturate(dot(n, l));
        const float shadow = direct > 0.0f ? shadowRay(hitPos, n, light) : 0.0f;
        const float3 sun = shadeSurface(albedo.rgb, metallic, roughness, n, -dir, l, light.color,
                                        light.intensity, shadow);
        const float3 ambient = float3(0.30f, 0.32f, 0.36f) * (n.y * 0.2f + 0.5f);
        const float3 shaded = albedo.rgb * ambient + sun;

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
