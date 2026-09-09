// Ray-query machinery shared by every traced path (ps_6_5 / SM6.5 only):
// the scene TLAS + geometry-pool fetch bindings, the candidate filter for
// non-opaque geometry, the sun shadow ray, ray-cone texture LOD, and the
// full hit shading a committed triangle gets — used by the traced primary
// pass and by per-object reflection rays fired from the raster pass.
// Include after shading.hlsli; the including shader must declare a push
// constant struct `pc` with a `cameraSlot` member.

[[vk::binding(10, 0)]] RaytracingAccelerationStructure sceneBVH;
// The whole geometry pool as raw bytes: interleaved vertices (stride 32:
// pos3f/normal3f/uv2f) and uint32 indices, addressed via geometryInfo.
[[vk::binding(11, 0)]] ByteAddressBuffer geometryBytes;
// Per BLAS-geometry index (== object index): x = firstIndex (uint32 units
// from pool start), y = vertexOffset (vertex-stride units from pool start),
// z = per-slot vertex stride for animated meshes (their y points at the
// posed per-slot regions the BLAS is refitted from; 0 = static), w unused.
[[vk::binding(12, 0)]] StructuredBuffer<uint4> geometryInfo;

static const uint kVertexStrideBytes = 32u;
static const uint kMaxTransparencySteps = 4u;

float3 vertexPosition(uint vertex) {
    return asfloat(geometryBytes.Load3(vertex * kVertexStrideBytes));
}

float3 vertexNormal(uint vertex) {
    return asfloat(geometryBytes.Load3(vertex * kVertexStrideBytes + 12));
}

float2 vertexUv(uint vertex) {
    return asfloat(geometryBytes.Load2(vertex * kVertexStrideBytes + 24));
}

uint3 triangleIndices(uint4 info, uint primitive) {
    // Animated meshes fetch from the current slot's posed copy — the same
    // vertices the refitted BLAS traced against this frame.
    return geometryBytes.Load3((info.x + primitive * 3) * 4) + info.y + pc.cameraSlot * info.z;
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

// One opaque any-hit ray from the surface toward the sun. Alpha-masked
// casters count as solid here (colored/cutout shadow rays are a later
// any-hit refinement).
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

// Distance-bounded occlusion ray toward one point light. The reach stops
// short of the light point: scripts place lamp lights at (or inside) the
// fixture's bulb geometry, which would otherwise occlude everything.
float pointShadowRay(float3 worldPos, float3 n, float3 l, float dist) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray;
    ray.Origin = worldPos + n * 0.02f;
    ray.Direction = l;
    ray.TMin = 0.0f;
    ray.TMax = max(dist - 0.3f, 0.0f);
    q.TraceRayInline(sceneBVH, RAY_FLAG_NONE, 0xff, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0f : 1.0f;
}

// Traced variant of shading.hlsli's shadePointLights: same sample math,
// plus one short shadow ray per light that would actually contribute —
// the attenuation window and a luminance floor gate the rays, so cost
// scales with the lights covering the fragment, not the light count.
static const float kPointShadowMinContribution = 0.004f;
float3 shadePointLightsTraced(float3 albedo, float metallic, float roughness, float3 n, float3 v,
                              float3 worldPos, LightData light) {
    const uint count = min((uint)light.skyColor.w, kMaxPointLights);
    float3 sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        const PointLight pl = light.pointLights[i];
        const PointLightSample s = samplePointLight(pl, worldPos);
        if (s.atten <= 0.0f) {
            continue;
        }
        const float ndotl = dot(n, s.l);
        if (ndotl <= 0.0f) {
            continue;
        }
        const float peak = max(pl.colorIntensity.r, max(pl.colorIntensity.g, pl.colorIntensity.b));
        if (pl.colorIntensity.w * s.atten * ndotl * peak < kPointShadowMinContribution) {
            continue;
        }
        if (pointShadowRay(worldPos, n, s.l, s.dist) <= 0.0f) {
            continue;
        }
        sum += shadeSurface(albedo, metallic, roughness, n, v, s.l, pl.colorIntensity.rgb,
                            pl.colorIntensity.w * s.atten, 1.0f);
    }
    return sum;
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
float3 applyNormalMapTraced(uint normalIndex, float3 n, uint3 tri, float2 uv, float lod) {
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

float3 skyColor(float3 dir) {
    // Subtle zenith-to-horizon gradient standing in for the raster path's
    // clear color until a real sky exists.
    return lerp(float3(0.10f, 0.11f, 0.14f), float3(0.02f, 0.02f, 0.04f),
                saturate(dir.y * 0.5f + 0.5f));
}

// Everything a committed triangle hit shades to: full material fetch with
// ray-cone LOD, sun + traced shadow ray + ambient. travelled = total ray
// distance from the eye (drives the cone width), conePerUnit = the
// camera's per-pixel spread (CameraData.position.w).
struct TracedHit {
    float3 color;   // fully shaded surface color
    float opacity;  // saturate(albedo.a * baseAlpha) — 1 for opaque
    float3 position;
    uint flags;    // the hit object's flags
    float3 normal; // shading normal (normal-mapped, faced toward the ray)
    float3 f0;     // Fresnel reflectance at normal incidence
    float roughness;
};

TracedHit shadeCommittedHit(uint objectIndex, uint primitive, float2 bary, float3 origin,
                            float3 dir, float rayT, float travelled, float conePerUnit,
                            LightData light) {
    const ObjectData object = objects[objectIndex];
    const uint3 tri = triangleIndices(geometryInfo[objectIndex], primitive);
    const float w0 = 1.0f - bary.x - bary.y;
    float3 n = normalize(vertexNormal(tri.x) * w0 + vertexNormal(tri.y) * bary.x +
                         vertexNormal(tri.z) * bary.y);
    if (dot(n, dir) > 0.0f) {
        n = -n; // shade the side the ray sees (geometry is unculled)
    }
    const float2 uv = vertexUv(tri.x) * w0 + vertexUv(tri.y) * bary.x + vertexUv(tri.z) * bary.y;

    TracedHit hit;
    hit.position = origin + dir * rayT;
    hit.flags = object.flags;

    // Manual mip selection from the pixel's ray cone at this distance.
    const float coneWidth = max(travelled * conePerUnit, 1.0e-6f);
    const float ndotd = abs(dot(n, dir));
    const float lodBase = triangleLodBase(tri);
    n = applyNormalMapTraced(object.normalIndex, n, tri, uv,
                             textureLod(object.normalIndex, lodBase, coneWidth, ndotd));
    float metallic = object.metallicFactor;
    float roughness = object.roughnessFactor;
    if (object.mrIndex != 0) {
        const float2 mr = textures[NonUniformResourceIndex(object.mrIndex)]
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
    const float direct = light.intensity > 0.0f ? saturate(dot(n, l)) : 0.0f;
    const float shadow = direct > 0.0f ? shadowRay(hit.position, n, light) : 0.0f;
    const float3 sun = shadeSurface(albedo.rgb, metallic, roughness, n, -dir, l, light.color,
                                    light.intensity, shadow);
    hit.color = albedo.rgb * ambientLight(n, light.ambientColor.rgb) + sun;
    hit.color +=
        shadePointLightsTraced(albedo.rgb, metallic, roughness, n, -dir, hit.position, light);
    hit.opacity =
        (object.flags & kFlagTransparent) != 0 ? saturate(albedo.a * object.baseAlpha) : 1.0f;
    hit.normal = n;
    hit.f0 = lerp(0.04f, albedo.rgb, metallic);
    hit.roughness = roughness;
    return hit;
}

// One reflection bounce for per-object RT: trace from the surface, shade
// what the ray sees, marching through transparent hits like the traced
// primary pass. No recursion — a reflective object seen in a reflection
// shades as a plain surface.
float3 traceReflection(float3 origin, float3 dir, float travelled, float conePerUnit,
                       LightData light) {
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
        const TracedHit hit = shadeCommittedHit(
            q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex(),
            q.CommittedTriangleBarycentrics(), origin, dir, q.CommittedRayT(),
            travelled + q.CommittedRayT(), conePerUnit, light);
        travelled += q.CommittedRayT();
        if ((hit.flags & kFlagTransparent) != 0) {
            color += throughput * hit.opacity * hit.color;
            throughput *= 1.0f - hit.opacity;
            if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01f) {
                break;
            }
            origin = hit.position + dir * 1.0e-3f;
            continue;
        }
        color += throughput * hit.color;
        break;
    }
    return color;
}

// One-bounce traced reflection for a committed hit, Fresnel-mixed into its
// base shading via the shared helper (same math as the probe tier).
float3 applyReflection(TracedHit hit, float3 dir, float travelled, float conePerUnit,
                       LightData light) {
    const float3 r = reflect(dir, hit.normal);
    const float3 reflected =
        traceReflection(hit.position + hit.normal * 1.0e-3f, r, travelled, conePerUnit, light);
    return mixReflection(hit.color, reflected, hit.f0, hit.roughness, dot(hit.normal, -dir));
}
