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
    // Unit-cube corner (each coord 0 or 1), interpolated across faces —
    // the debug PS turns it into box-edge outlines. PSMain must still
    // TOUCH it (degenerate weight) or dxc strips the input location and
    // the shared proxy.vert pairing trips an interface-mismatch WARN
    // (the point_shadow.hlsl lesson).
    float3 corner : CORNER;
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
        output.corner = float3(0.0f, 0.0f, 0.0f);
        return output;
    }
    const uint3 corner = kCubeCorner[vertexId];
    const float3 world = float3(corner.x != 0 ? b.bmax.x : b.bmin.x,
                                corner.y != 0 ? b.bmax.y : b.bmin.y,
                                corner.z != 0 ? b.bmax.z : b.bmin.z);
    output.position = mul(cameras[push.slot].viewProj, float4(world, 1.0f));
    output.corner = float3(corner);
    return output;
}

[earlydepthstencil]
void PSMain(VSOutput input) {
    // Plain store, not an atomic: many lanes writing the same 1 is fine.
    // The corner term is a degenerate touch (always 0) keeping the input
    // location alive — see the VSOutput comment.
    visibility[push.slot * push.capacity + input.entry] =
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
