// Occlusion proxy pass (docs: scratch/memories — GPU occlusion slice 1):
// drawn at the END of the main rendering pass, one instance per draw
// template, 36 unindexed vertices generating the entry's world AABB as a
// cube straight from the bounds table — no vertex buffer, no proxy
// geometry in the pool. Depth test against the frame's finished scene
// depth (LESS_OR_EQUAL, writes off, color masked): any fragment that
// survives means some part of the box is in front of everything drawn, so
// the entry might be visible — the pixel shader marks its slot in the
// visibility buffer, which the NEXT frame's cull dispatch consumes
// (binding 26, per-slot regions, one frame of latency by design).
// [earlydepthstencil] is REQUIRED: a storage write in the PS otherwise
// forces late depth testing and every rasterized fragment would mark
// visible, silently degrading the whole technique to frustum-only.
// Local-bounds entries (bmax.w != 0 — instanced/runtime models) emit
// degenerate positions and are never occlusion-culled.

// Must match CameraData in shading.hlsli (only viewProj read).
struct CameraData {
    column_major float4x4 viewProj;
    float4 position;
    float4 rightAxis;
    float4 upAxis;
    float4 forwardAxis;
};

// Must match ObjectBounds in cull.hlsl / the viewer.
struct ObjectBounds {
    float4 bmin;
    float4 bmax;
};

struct ProxyPush {
    uint slot;     // frame-in-flight index selecting the buffer regions
    uint capacity; // entries per slot region (bounds + visibility)
};
[[vk::push_constant]] ProxyPush push;

[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras;
[[vk::binding(22, 0)]] StructuredBuffer<ObjectBounds> bounds;
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> visibility;

struct VSOutput {
    float4 position : SV_Position;
    nointerpolation uint entry : ENTRYID;
};

// 12 triangles of a unit cube as per-vertex corner selectors (xyz in
// {0,1}); winding is irrelevant — the pipeline culls nothing.
static const uint3 kCubeCorner[36] = {
    // -Z face
    uint3(0, 0, 0), uint3(1, 0, 0), uint3(1, 1, 0),
    uint3(0, 0, 0), uint3(1, 1, 0), uint3(0, 1, 0),
    // +Z face
    uint3(0, 0, 1), uint3(1, 1, 1), uint3(1, 0, 1),
    uint3(0, 0, 1), uint3(0, 1, 1), uint3(1, 1, 1),
    // -X face
    uint3(0, 0, 0), uint3(0, 1, 0), uint3(0, 1, 1),
    uint3(0, 0, 0), uint3(0, 1, 1), uint3(0, 0, 1),
    // +X face
    uint3(1, 0, 0), uint3(1, 1, 1), uint3(1, 1, 0),
    uint3(1, 0, 0), uint3(1, 0, 1), uint3(1, 1, 1),
    // -Y face
    uint3(0, 0, 0), uint3(1, 0, 1), uint3(1, 0, 0),
    uint3(0, 0, 0), uint3(0, 0, 1), uint3(1, 0, 1),
    // +Y face
    uint3(0, 1, 0), uint3(1, 1, 0), uint3(1, 1, 1),
    uint3(0, 1, 0), uint3(1, 1, 1), uint3(0, 1, 1),
};

VSOutput VSMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
    VSOutput output;
    output.entry = instanceId;
    const ObjectBounds b = bounds[push.slot * push.capacity + instanceId];
    // Local-mode or never-written bounds: collapse the box (w = 0 clips
    // every vertex, so the rasterizer drops the whole instance).
    if (b.bmax.w != 0.0f || b.bmin.x > b.bmax.x) {
        output.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return output;
    }
    const uint3 corner = kCubeCorner[vertexId];
    const float3 world = float3(corner.x != 0 ? b.bmax.x : b.bmin.x,
                                corner.y != 0 ? b.bmax.y : b.bmin.y,
                                corner.z != 0 ? b.bmax.z : b.bmin.z);
    output.position = mul(cameras[push.slot].viewProj, float4(world, 1.0f));
    return output;
}

[earlydepthstencil]
void PSMain(VSOutput input) {
    // Plain store, not an atomic: many lanes writing the same 1 is fine.
    visibility[push.slot * push.capacity + input.entry] = 1u;
}
