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

#include "backend.hlsli"

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

// Must match InstanceRow in cull.hlsl / shading.hlsli / the viewer.
struct InstanceRow {
    uint objectIndex;
    uint transformIndex;
};

struct ProxyPush {
    uint slot;     // frame-in-flight index selecting the buffer regions
    uint capacity; // entries per slot region (bounds + templates)
};
REND_PUSH(ProxyPush, push);

// Must match kTransformCapacity / kInstanceRowCapacity in the viewer.
// A visibility region holds capacity entry slots followed by
// kInstanceRowCapacity per-instance slots (region stride = the sum).
static const uint kTransformCapacity = 4096;
static const uint kInstanceRowCapacity = 4096;

[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras REND_B(6);
[[vk::binding(19, 0)]] StructuredBuffer<column_major float4x4> objectTransforms REND_B(19);
[[vk::binding(20, 0)]] REND_SHARED_BUFFER(InstanceRow) instanceRows REND_U(20);
[[vk::binding(22, 0)]] StructuredBuffer<ObjectBounds> bounds REND_B(22);
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> visibility REND_U(26);
// Canonical-row -> draw-entry map (the viewer maintains it beside the
// instance rows): 0xffffffff = the row is dead or belongs to a scene
// entry (covered by the per-entry pass). Lets VSInstances find a row's
// local bounds without any per-frame CPU work.
[[vk::binding(34, 0)]] StructuredBuffer<uint> rowEntries REND_B(34);
// GPU-refined oriented bounding boxes (must match obb.hlsl / cull.hlsl /
// the viewer): row e at 16 + e*64 = float4 center (w = ready flag) +
// three float4 {unit axis, half extent}. Once an entry's row is ready the
// proxy (and the debug overlay pairing this VS) rasterizes the tighter
// oriented box instead of the world AABB — the cull dispatch's
// camera-inside bypass switches with it, keeping test and box matched.
static const uint kObbHeaderBytes = 16u;
static const uint kObbRowBytes = 64u;
[[vk::binding(33, 0)]] REND_SHARED_BYTES obbs REND_U(33);

struct VSOutput {
    float4 position : SV_Position;
    nointerpolation uint entry : ENTRYID;
    // Unit-cube corner (each coord 0 or 1), interpolated across faces —
    // the debug PS turns it into box-edge outlines. PSMain must still
    // TOUCH it (degenerate weight) or dxc strips the input location and
    // the shared proxy.vert pairing trips an interface-mismatch WARN
    // (the point_shadow.hlsl lesson).
    float3 corner : CORNER;
};

// Conservative screen-space dilation: every box vertex is pushed this
// far outward in NDC (scaled by w) away from the box center, so a
// mostly-hidden box's surviving sliver always covers at least one pixel
// center instead of landing on/off centers frame to frame (the
// alternating-frame occlusion pop). ~Half a pixel up to ~1300 px
// viewport height; over-dilation only errs visible (conservative).
static const float kDilateNdc = 0.0015f;

// Dilate a clip-space vertex away from the box's clip-space center.
// Vertices at/behind the eye plane are left alone — the camera-inside
// bypass covers those boxes anyway.
float4 dilate(float4 clip, float4 centerClip) {
    if (clip.w > 0.0f && centerClip.w > 0.0f) {
        const float2 dir = sign(clip.xy / clip.w - centerClip.xy / centerClip.w);
        clip.xy += dir * (kDilateNdc * clip.w);
    }
    return clip;
}

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
    instanceId = rendInstanceIndex(instanceId);
    output.entry = instanceId;
    const ObjectBounds b = bounds[push.slot * push.capacity + instanceId];
    // Local-mode or never-written bounds: collapse the box (w = 0 clips
    // every vertex, so the rasterizer drops the whole instance).
    if (b.bmax.w != 0.0f || b.bmin.x > b.bmax.x) {
        output.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
        output.corner = float3(0.0f, 0.0f, 0.0f);
        return output;
    }
    const uint3 corner = kCubeCorner[vertexId];
    float3 world;
    float3 center;
    const uint row = kObbHeaderBytes + instanceId * kObbRowBytes;
    const float4 obbCenter = asfloat(obbs.Load4(row));
    if (obbCenter.w != 0.0f) {
        const float4 ax = asfloat(obbs.Load4(row + 16));
        const float4 ay = asfloat(obbs.Load4(row + 32));
        const float4 az = asfloat(obbs.Load4(row + 48));
        const float3 s = float3(corner) * 2.0f - 1.0f;
        world = obbCenter.xyz + ax.xyz * (ax.w * s.x) + ay.xyz * (ay.w * s.y) +
                az.xyz * (az.w * s.z);
        center = obbCenter.xyz;
    } else {
        world = float3(corner.x != 0 ? b.bmax.x : b.bmin.x,
                       corner.y != 0 ? b.bmax.y : b.bmin.y,
                       corner.z != 0 ? b.bmax.z : b.bmin.z);
        center = (b.bmin.xyz + b.bmax.xyz) * 0.5f;
    }
    const float4x4 viewProj = cameras[push.slot].viewProj;
    output.position = dilate(mul(viewProj, float4(world, 1.0f)),
                             mul(viewProj, float4(center, 1.0f)));
    output.corner = float3(corner);
    output.position = rendClip(output.position);
    return output;
}

// Per-INSTANCE proxy pass (runtime instanced models): one instance per
// CANONICAL row. Live local-mode rows push their entry's local AABB
// through the row's transform — the rasterized parallelepiped IS the
// instance's oriented box — and mark the visibility region's per-
// instance slot (capacity + row), which the next frame's cull reads in
// its per-instance test. Dead rows and scene (world-mode) rows emit
// degenerate positions; the same PSMain/PSDebug pair the entry pass.
VSOutput VSInstances(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
    VSOutput output;
    instanceId = rendInstanceIndex(instanceId);
    output.entry = push.capacity + instanceId; // per-instance visibility slot
    output.corner = float3(0.0f, 0.0f, 0.0f);
    output.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
    const uint entry = rowEntries[instanceId];
    if (entry == 0xffffffffu) {
        return output; // dead row or scene entry — the entry pass covers it
    }
    const ObjectBounds b = bounds[push.slot * push.capacity + entry];
    if (b.bmax.w == 0.0f || b.bmin.x > b.bmax.x) {
        return output;
    }
    const uint3 corner = kCubeCorner[vertexId];
    const float3 local = float3(corner.x != 0 ? b.bmax.x : b.bmin.x,
                                corner.y != 0 ? b.bmax.y : b.bmin.y,
                                corner.z != 0 ? b.bmax.z : b.bmin.z);
    const float4x4 world = objectTransforms[push.slot * kTransformCapacity +
                                            instanceRows[instanceId].transformIndex];
    const float3 pos = mul(world, float4(local, 1.0f)).xyz;
    const float3 center = mul(world, float4((b.bmin.xyz + b.bmax.xyz) * 0.5f, 1.0f)).xyz;
    const float4x4 viewProj = cameras[push.slot].viewProj;
    output.position = dilate(mul(viewProj, float4(pos, 1.0f)),
                             mul(viewProj, float4(center, 1.0f)));
    output.corner = float3(corner);
    output.position = rendClip(output.position);
    return output;
}

[earlydepthstencil]
void PSMain(VSOutput input) {
    // Plain store, not an atomic: many lanes writing the same 1 is fine.
    // Region stride = capacity entry slots + kInstanceRowCapacity
    // per-instance slots; the entry VS passes an entry id, VSInstances
    // passes capacity + row, so one store serves both passes. The corner
    // term is a degenerate touch (always 0) keeping the input location
    // alive — see the VSOutput comment.
    visibility[push.slot * (push.capacity + kInstanceRowCapacity) + input.entry] =
        1u | (uint)(input.corner.x * 1e-20f);
}

// Debug visualization (Settings -> "Show occlusion boxes"): the SAME
// boxes the proxy pass rasterizes, drawn as outlined translucent shells
// instead of visibility stores — the pipeline pairing this entry keeps
// the proxy's LESS_OR_EQUAL test-only depth state, so the tinted
// fragments are exactly the ones that would mark an entry visible.
// Per-entry color = golden-ratio hash of the entry id; edges detected in
// unit-cube space with fwidth for ~constant screen-space line width
// (solid fills at scene-box counts just compound into fog).
float4 PSDebug(VSOutput input) : SV_Target0 {
    const float3 tint =
        frac((float)(input.entry + 1) *
             float3(0.61803398875f, 0.38196601125f, 0.23606797750f));
    // Distance to the nearest cube face per axis, in pixels. On any face
    // one coord is constant 0/1 (min component ~0 everywhere); an EDGE
    // is where the SECOND-smallest distance is also small.
    const float3 d = min(input.corner, 1.0f - input.corner);
    const float3 px = d / max(fwidth(d), 1e-6f);
    const float second = min(max(px.x, px.y), min(max(px.y, px.z), max(px.z, px.x)));
    const float edge = 1.0f - smoothstep(0.8f, 1.8f, second);
    return float4(tint * (0.55f + 0.45f * edge), lerp(0.04f, 0.85f, edge));
}
