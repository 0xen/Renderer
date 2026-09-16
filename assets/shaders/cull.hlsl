// Compaction + frustum-cull pass for the GPU-driven scene
// (docs/ARCHITECTURE.md): one thread per registered object reads its draw
// template and appends it to THREE compacted indirect lists:
//   - binding 4 (+ counts[slot*4]): every live entry — the shadow passes
//     draw this one, because a caster outside the CAMERA frustum must
//     still cast into the view;
//   - binding 21 (+ counts[slot*4+1]): OPAQUE entries that survive the
//     frustum — the main scene pass draws this one first;
//   - binding 24 (+ counts[slot*4+3]): TRANSPARENT entries that survive
//     the frustum — the blend pass draws this one after the opaques
//     (depth test on, depth write off).
// Culling is PER INSTANCE for instanced entries (bounds mode 1): each
// instance's local AABB is pushed through its transform and tested; the
// survivors' instance rows are compacted into the slot's scratch region
// of the rows buffer (allocated from counts[slot*3+2]) and the emitted
// draw's firstInstance/instanceCount point there. Fully visible entries
// keep their canonical rows (no copies); fully hidden ones are dropped.
// The scene pass then draws via vkCmdDrawIndexedIndirectCount, so no
// visibility decision ever touches the CPU.
// Survivors also pick a level of detail (binding 25): distance to the
// entry's AABB selects a simplified index range baked at scene load, so
// the emitted command's firstIndex/indexCount may differ from the
// template's. The shadow stream always keeps full detail.
// Occlusion (binding 26): entries whose world AABB left no pixel in
// front of last frame's depth (proxy pass, end of the previous frame)
// are dropped from the scene streams — never from the shadow stream.

#include "backend.hlsli"

// Must match rend::gpu::DrawIndexedIndirect (VkDrawIndexedIndirectCommand).
struct DrawCommand {
    uint baseInstance; // == firstInstance; D3D12 root constant (backend.hlsli)
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

// Must match CameraData in shading.hlsli / the viewer (only viewProj read).
struct CameraData {
    column_major float4x4 viewProj;
    float4 position;
    float4 rightAxis;
    float4 upAxis;
    float4 forwardAxis;
};

// Must match ObjectBounds in the viewer. Two modes, picked by bmax.w:
//   0 — bmin/bmax are a WORLD-space AABB, tested once (scene geometry —
//       vertices are world-baked);
//   1 — bmin/bmax are the mesh's LOCAL AABB; each instance's transform
//       (binding 19, this slot's region) takes it to world space and it
//       is tested per instance (runtime models).
// bmin.w is a small-integer bitmask (exact in float): bit 0 marks entries
// the test must never drop (animated meshes — their pose can exceed the
// bind-pose bounds); bit 1 routes the entry to the transparent stream.
struct ObjectBounds {
    float4 bmin;
    float4 bmax;
};

// Must match InstanceRow in shading.hlsli / the viewer.
struct InstanceRow {
    uint objectIndex;
    uint transformIndex;
};

struct CullPush {
    uint drawCount; // live templates this frame (may grow at runtime)
    uint slot;      // frame-in-flight index selecting the buffer regions
    uint capacity;  // entries per slot region — the fixed stride; runtime
                    // model loads change drawCount but never this
    uint flags;     // bit 0: frustum culling, bit 1: LOD selection
    // Screen-size scale for the LOD pick: pixels per world unit at unit
    // distance, divided by the target error in pixels. 0 disables the
    // pick (guards the first frames before the viewer measures it).
    float lodFactor;
};
REND_PUSH(CullPush, push);

static const uint kCullFrustum = 1u;
static const uint kCullLod = 2u;
static const uint kCullOcclusion = 4u;
// Must match kTransformCapacity / kInstanceRowCapacity in the viewer.
static const uint kTransformCapacity = 4096;
static const uint kInstanceRowCapacity = 4096;

static const uint kBoundsAlwaysVisible = 1u;
static const uint kBoundsTransparent = 2u;

[[vk::binding(3, 0)]] StructuredBuffer<DrawCommand> templates REND_B(3);
[[vk::binding(4, 0)]] RWStructuredBuffer<DrawCommand> compacted REND_U(4);
// kCountStride per slot, zeroed before dispatch: [0] shadow-stream count,
// [1] opaque scene-stream count, [2] scratch-row allocator for partial
// entries, [3] transparent-stream count, [4]/[5] INDICES emitted to the
// opaque/transparent streams (post-LOD, x instanceCount — the CPU's
// triangle stat, /3), [6] entries dropped by the occlusion test, [7]
// spare. Must match countRegionStride in the viewer and the count
// offsets in frame_renderer's bindAndDraw.
static const uint kCountStride = 8;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> counts REND_U(5);
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras REND_B(6);
[[vk::binding(19, 0)]] StructuredBuffer<column_major float4x4> objectTransforms REND_B(19);
[[vk::binding(21, 0)]] RWStructuredBuffer<DrawCommand> culled REND_U(21);
[[vk::binding(22, 0)]] StructuredBuffer<ObjectBounds> bounds REND_B(22);
// The instance-row buffer (same VkBuffer as the vertex stage's binding
// 20): rows [0, kInstanceRowCapacity) are canonical; per-slot scratch
// regions above hold the compacted survivors of partially visible draws.
[[vk::binding(23, 0)]] RWStructuredBuffer<InstanceRow> instanceRows REND_U(23);
[[vk::binding(24, 0)]] RWStructuredBuffer<DrawCommand> transparent REND_U(24);

// Discrete LOD chain per draw entry (must match MeshLodTable in the
// viewer). Every level indexes the SAME vertex block as the template —
// simplification only removes triangles — so a level is just another
// firstIndex/indexCount pair into the geometry pool. lods[0] is the full
// mesh; errors are absolute world-space simplification error, ascending.
// lodCount <= 1 (runtime models, animated meshes, <Model lod="off">,
// meshes below the size floor) means the template's ranges stand.
// Static after load: ONE global region, no per-slot copies.
static const uint kMaxMeshLods = 4;
struct MeshLodLevel {
    uint firstIndex;
    uint indexCount;
    float error;
    uint pad;
};
struct MeshLodTable {
    uint lodCount;
    uint pad0;
    uint pad1;
    uint pad2;
    MeshLodLevel lods[kMaxMeshLods];
};
[[vk::binding(25, 0)]] StructuredBuffer<MeshLodTable> meshLods REND_B(25);

// Occlusion visibility (per-slot regions of push.capacity ENTRY slots
// followed by kInstanceRowCapacity per-INSTANCE slots): the proxy passes
// at the END of the previous frame drew every world entry's box AND
// every live canonical instance row's transformed local box against the
// finished scene depth (test only, [earlydepthstencil]) and marked
// survivors 1 — so region [slot ^ 1] holds "had any pixel in front of
// last frame's depth" and region [slot] holds the frame before that
// (it is zeroed AFTER this dispatch reads it, then refilled by this
// frame's proxy passes). Occlusion drops an entry only when BOTH
// regions say hidden: a proxy box whose last sliver of coverage lands
// on/off pixel centers on alternating frames would otherwise oscillate
// visible/occluded every frame (one extra frame of disappear latency,
// no popping). Seeded all-1 at load so frame 0 draws everything.
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> visibility REND_U(26);

// GPU-refined oriented bounding boxes (must match obb.hlsl / proxy.hlsl /
// the viewer): row e at 16 + e*64 = float4 center (w = ready flag) + three
// float4 {unit axis, half extent}. The refine dispatch runs just before
// this one; a ready row is a strictly tighter box than the entry's world
// AABB (the refine pass keeps whichever is smaller), so every test below
// prefers it and falls back to the AABB until it exists. ONE global
// region — rows are load-time constants once written.
static const uint kObbHeaderBytes = 16u;
static const uint kObbRowBytes = 64u;
[[vk::binding(33, 0)]] REND_SHARED_BYTES obbs REND_U(33);

struct Obb {
    float3 center;
    float3 axis0;
    float3 axis1;
    float3 axis2;
    float3 extent; // half extents along the three axes
    bool ready;
};

Obb loadObb(uint entry) {
    const uint row = kObbHeaderBytes + entry * kObbRowBytes;
    const float4 c = asfloat(obbs.Load4(row));
    const float4 x = asfloat(obbs.Load4(row + 16));
    const float4 y = asfloat(obbs.Load4(row + 32));
    const float4 z = asfloat(obbs.Load4(row + 48));
    Obb o;
    o.center = c.xyz;
    o.axis0 = x.xyz;
    o.axis1 = y.xyz;
    o.axis2 = z.xyz;
    o.extent = float3(x.w, y.w, z.w);
    o.ready = c.w != 0.0f;
    return o;
}

// Coordinates of a world point in the box's frame.
float3 obbLocal(Obb o, float3 p) {
    const float3 d = p - o.center;
    return float3(dot(d, o.axis0), dot(d, o.axis1), dot(d, o.axis2));
}

// AABB vs the camera frustum, planes pulled from the slot's viewProj
// (Gribb-Hartmann; clip = M * v, Vulkan z in [0, w]). Each plane points
// inward; the AABB is outside iff its most-positive vertex (p-vertex)
// still sits behind some plane.
bool inFrustum(float3 bmin, float3 bmax, float4x4 m) {
    const float4 planes[6] = {
        m[3] + m[0], // left:   x >= -w
        m[3] - m[0], // right:  x <=  w
        m[3] + m[1], // bottom: y >= -w
        m[3] - m[1], // top:    y <=  w
        m[2],        // near:   z >=  0
        m[3] - m[2], // far:    z <=  w
    };
    [unroll]
    for (uint i = 0; i < 6; ++i) {
        const float4 plane = planes[i];
        const float3 p = float3(plane.x >= 0.0f ? bmax.x : bmin.x,
                                plane.y >= 0.0f ? bmax.y : bmin.y,
                                plane.z >= 0.0f ? bmax.z : bmin.z);
        if (dot(plane.xyz, p) + plane.w < 0.0f) {
            return false;
        }
    }
    return true;
}

// OBB vs the frustum: same planes, projected-radius form — exact for a
// box vs a plane, so still conservative overall (the box may cross two
// planes' corners without truly intersecting the frustum, same as the
// AABB test).
bool obbInFrustum(Obb o, float4x4 m) {
    const float4 planes[6] = {
        m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[2], m[3] - m[2],
    };
    [unroll]
    for (uint i = 0; i < 6; ++i) {
        const float4 plane = planes[i];
        const float r = o.extent.x * abs(dot(plane.xyz, o.axis0)) +
                        o.extent.y * abs(dot(plane.xyz, o.axis1)) +
                        o.extent.z * abs(dot(plane.xyz, o.axis2));
        if (dot(plane.xyz, o.center) + plane.w + r < 0.0f) {
            return false;
        }
    }
    return true;
}

// One instance's visibility: the entry's local AABB through the
// instance's world matrix (center/extent form — exact for the box), then
// the frustum test and — per instance — the occlusion test against the
// previous frame's per-instance proxy pass (visibility slot capacity +
// canonical row; the camera sitting inside the slightly expanded box
// bypasses it, same near-clip false-negative case as world entries).
// countOcclusion: bump the occluded stat only from the counting loop —
// the copy loop re-evaluates the same instances.
bool instanceVisible(ObjectBounds b, uint rowIndex, float4x4 viewProj, bool countOcclusion) {
    const InstanceRow row = instanceRows[rowIndex];
    const float4x4 world =
        objectTransforms[push.slot * kTransformCapacity + row.transformIndex];
    const float3 center = (b.bmin.xyz + b.bmax.xyz) * 0.5f;
    const float3 extent = (b.bmax.xyz - b.bmin.xyz) * 0.5f;
    const float3 wc = mul(world, float4(center, 1.0f)).xyz;
    const float3 we = float3(dot(abs(world[0].xyz), extent), dot(abs(world[1].xyz), extent),
                             dot(abs(world[2].xyz), extent));
    if ((push.flags & kCullFrustum) != 0 && !inFrustum(wc - we, wc + we, viewProj)) {
        return false;
    }
    if ((push.flags & kCullOcclusion) != 0) {
        const float3 camPos = cameras[push.slot].position.xyz;
        const bool cameraInside =
            all(camPos >= wc - we - 0.5f) && all(camPos <= wc + we + 0.5f);
        // kFramesInFlight == 2: the other slot's region is last frame's,
        // this slot's is the frame before (hysteresis — see binding 26).
        const uint visStride = push.capacity + kInstanceRowCapacity;
        const uint visSlot = push.capacity + rowIndex;
        if (!cameraInside &&
            (visibility[(push.slot ^ 1) * visStride + visSlot] |
             visibility[push.slot * visStride + visSlot]) == 0) {
            if (countOcclusion) {
                InterlockedAdd(counts[push.slot * kCountStride + 6], 1);
            }
            return false;
        }
    }
    return true;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= push.drawCount) {
        return;
    }
    const uint base = push.slot * push.capacity;
    DrawCommand cmd = templates[base + id.x];
    if (cmd.instanceCount == 0) {
        return; // hidden
    }
    uint dst;
    InterlockedAdd(counts[push.slot * kCountStride], 1, dst);
    compacted[base + dst] = cmd;

    const ObjectBounds b = bounds[base + id.x];
    const uint boundFlags = (uint)b.bmin.w;
    // Refined OBB, if the load-time refinement got to this entry yet
    // (world-mode only — instanced entries already test tight local boxes
    // through their transforms).
    Obb obb = (Obb)0;
    if (b.bmax.w == 0.0f) {
        obb = loadObb(id.x);
    }
    if ((boundFlags & kBoundsAlwaysVisible) == 0) {
        const float4x4 viewProj = cameras[push.slot].viewProj;
        if (b.bmax.w == 0.0f) {
            // World-space bounds: one frustum test covers the whole entry
            // (occlusion for world entries is the separate block below).
            if ((push.flags & kCullFrustum) != 0 &&
                (obb.ready ? !obbInFrustum(obb, viewProj)
                           : !inFrustum(b.bmin.xyz, b.bmax.xyz, viewProj))) {
                return; // outside the view — shadows above still drew it
            }
        } else if ((push.flags & (kCullFrustum | kCullOcclusion)) != 0) {
            // Local bounds: frustum AND occlusion test each instance
            // (instanceVisible gates each test on its flag). Count
            // survivors first; fully visible entries keep their canonical
            // rows, partial ones compact the survivors into this slot's
            // scratch rows.
            uint visible = 0;
            for (uint k = 0; k < cmd.instanceCount; ++k) {
                visible += instanceVisible(b, cmd.firstInstance + k, viewProj, true) ? 1u : 0u;
            }
            if (visible == 0) {
                return;
            }
            if (visible < cmd.instanceCount) {
                uint rowBase;
                InterlockedAdd(counts[push.slot * kCountStride + 2], visible, rowBase);
                const uint scratch = (1 + push.slot) * kInstanceRowCapacity + rowBase;
                uint written = 0;
                for (uint k = 0; k < cmd.instanceCount; ++k) {
                    if (instanceVisible(b, cmd.firstInstance + k, viewProj, false)) {
                        instanceRows[scratch + written] =
                            instanceRows[cmd.firstInstance + k];
                        ++written;
                    }
                }
                cmd.firstInstance = scratch;
                cmd.baseInstance = scratch;
                cmd.instanceCount = visible;
            }
        }
    }
    // Occlusion test (world-bounds entries only, never the always-visible
    // ones, never the shadow stream above — a caster hidden from the
    // camera still casts into view): the entry is dropped when its box
    // left no pixel in front of LAST frame's depth. One frame of latency
    // by design — a disoccluded object pops in a frame late. The camera
    // sitting inside the (slightly expanded) box bypasses the test: near-
    // plane clipping can wipe out every box face and read as a false
    // "occluded" exactly when the object surrounds the viewer.
    if ((push.flags & kCullOcclusion) != 0 && (boundFlags & kBoundsAlwaysVisible) == 0 &&
        b.bmax.w == 0.0f) {
        const float3 camPos = cameras[push.slot].position.xyz;
        // The inside test must match the box the proxy pass rasterizes —
        // OBB once refined, AABB until then.
        const bool cameraInside =
            obb.ready ? all(abs(obbLocal(obb, camPos)) <= obb.extent + 0.5f)
                      : (all(camPos >= b.bmin.xyz - 0.5f) &&
                         all(camPos <= b.bmax.xyz + 0.5f));
        // kFramesInFlight == 2: the other slot's region is last frame's,
        // this slot's is the frame before (hysteresis — see binding 26).
        const uint visStride = push.capacity + kInstanceRowCapacity;
        if (!cameraInside &&
            (visibility[(push.slot ^ 1) * visStride + id.x] |
             visibility[push.slot * visStride + id.x]) == 0) {
            InterlockedAdd(counts[push.slot * kCountStride + 6], 1);
            return;
        }
    }

    // LOD pick, only for the streams below — the shadow stream above drew
    // the full-detail template (a distance-based swap tied to the CAMERA
    // would move shadow silhouettes as the viewer walks). World-bounds
    // entries only: instanced/runtime entries carry lodCount 0 anyway, and
    // animated meshes must keep the template's per-slot posed vertexOffset
    // pairing untouched. Picks the COARSEST level whose simplification
    // error still projects under the target pixel size at the entry's
    // distance (error * lodFactor <= distance); a camera inside the AABB
    // gives distance 0, so it keeps full detail.
    if ((push.flags & kCullLod) != 0 && push.lodFactor > 0.0f && b.bmax.w == 0.0f) {
        const MeshLodTable lodTable = meshLods[id.x];
        if (lodTable.lodCount > 1) {
            const float3 camPos = cameras[push.slot].position.xyz;
            float dist;
            if (obb.ready) {
                // Distance to the OBB surface: clamp in box space (the
                // orthonormal frame preserves lengths). Tighter box =
                // truer distance = the LOD error metric applied honestly.
                const float3 l = obbLocal(obb, camPos);
                dist = length(max(abs(l) - obb.extent, 0.0f));
            } else {
                const float3 nearest = clamp(camPos, b.bmin.xyz, b.bmax.xyz);
                dist = length(camPos - nearest);
            }
            uint lod = 0;
            for (uint i = 1; i < lodTable.lodCount; ++i) {
                if (lodTable.lods[i].error * push.lodFactor <= dist) {
                    lod = i;
                }
            }
            if (lod != 0) {
                cmd.firstIndex = lodTable.lods[lod].firstIndex;
                cmd.indexCount = lodTable.lods[lod].indexCount;
            }
        }
    }
    if ((boundFlags & kBoundsTransparent) != 0) {
        InterlockedAdd(counts[push.slot * kCountStride + 3], 1, dst);
        transparent[base + dst] = cmd;
        InterlockedAdd(counts[push.slot * kCountStride + 5],
                       cmd.indexCount * cmd.instanceCount);
    } else {
        InterlockedAdd(counts[push.slot * kCountStride + 1], 1, dst);
        culled[base + dst] = cmd;
        InterlockedAdd(counts[push.slot * kCountStride + 4],
                       cmd.indexCount * cmd.instanceCount);
    }
}
