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

// Must match rend::gpu::DrawIndexedIndirect (VkDrawIndexedIndirectCommand).
struct DrawCommand {
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
[[vk::push_constant]] CullPush push;

static const uint kCullFrustum = 1u;
static const uint kCullLod = 2u;
// Must match kTransformCapacity / kInstanceRowCapacity in the viewer.
static const uint kTransformCapacity = 4096;
static const uint kInstanceRowCapacity = 4096;

static const uint kBoundsAlwaysVisible = 1u;
static const uint kBoundsTransparent = 2u;

[[vk::binding(3, 0)]] StructuredBuffer<DrawCommand> templates;
[[vk::binding(4, 0)]] RWStructuredBuffer<DrawCommand> compacted;
// 4 per slot, zeroed before dispatch: [0] shadow-stream count, [1]
// opaque scene-stream count, [2] scratch-row allocator for partial
// entries, [3] transparent-stream count.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras;
[[vk::binding(19, 0)]] StructuredBuffer<column_major float4x4> objectTransforms;
[[vk::binding(21, 0)]] RWStructuredBuffer<DrawCommand> culled;
[[vk::binding(22, 0)]] StructuredBuffer<ObjectBounds> bounds;
// The instance-row buffer (same VkBuffer as the vertex stage's binding
// 20): rows [0, kInstanceRowCapacity) are canonical; per-slot scratch
// regions above hold the compacted survivors of partially visible draws.
[[vk::binding(23, 0)]] RWStructuredBuffer<InstanceRow> instanceRows;
[[vk::binding(24, 0)]] RWStructuredBuffer<DrawCommand> transparent;

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
[[vk::binding(25, 0)]] StructuredBuffer<MeshLodTable> meshLods;

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

// One instance's visibility: the entry's local AABB through the
// instance's world matrix (center/extent form — exact for the box), then
// the frustum test.
bool instanceVisible(ObjectBounds b, uint rowIndex, float4x4 viewProj) {
    const InstanceRow row = instanceRows[rowIndex];
    const float4x4 world =
        objectTransforms[push.slot * kTransformCapacity + row.transformIndex];
    const float3 center = (b.bmin.xyz + b.bmax.xyz) * 0.5f;
    const float3 extent = (b.bmax.xyz - b.bmin.xyz) * 0.5f;
    const float3 wc = mul(world, float4(center, 1.0f)).xyz;
    const float3 we = float3(dot(abs(world[0].xyz), extent), dot(abs(world[1].xyz), extent),
                             dot(abs(world[2].xyz), extent));
    return inFrustum(wc - we, wc + we, viewProj);
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
    InterlockedAdd(counts[push.slot * 4], 1, dst);
    compacted[base + dst] = cmd;

    const ObjectBounds b = bounds[base + id.x];
    const uint boundFlags = (uint)b.bmin.w;
    if ((push.flags & kCullFrustum) != 0 && (boundFlags & kBoundsAlwaysVisible) == 0) {
        const float4x4 viewProj = cameras[push.slot].viewProj;
        if (b.bmax.w == 0.0f) {
            // World-space bounds: one test covers the whole entry.
            if (!inFrustum(b.bmin.xyz, b.bmax.xyz, viewProj)) {
                return; // outside the view — shadows above still drew it
            }
        } else {
            // Local bounds: test each instance. Count survivors first;
            // fully visible entries keep their canonical rows, partial
            // ones compact the survivors into this slot's scratch rows.
            uint visible = 0;
            for (uint k = 0; k < cmd.instanceCount; ++k) {
                visible += instanceVisible(b, cmd.firstInstance + k, viewProj) ? 1u : 0u;
            }
            if (visible == 0) {
                return;
            }
            if (visible < cmd.instanceCount) {
                uint rowBase;
                InterlockedAdd(counts[push.slot * 4 + 2], visible, rowBase);
                const uint scratch = (1 + push.slot) * kInstanceRowCapacity + rowBase;
                uint written = 0;
                for (uint k = 0; k < cmd.instanceCount; ++k) {
                    if (instanceVisible(b, cmd.firstInstance + k, viewProj)) {
                        instanceRows[scratch + written] =
                            instanceRows[cmd.firstInstance + k];
                        ++written;
                    }
                }
                cmd.firstInstance = scratch;
                cmd.instanceCount = visible;
            }
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
            const float3 nearest = clamp(camPos, b.bmin.xyz, b.bmax.xyz);
            const float dist = length(camPos - nearest);
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
        InterlockedAdd(counts[push.slot * 4 + 3], 1, dst);
        transparent[base + dst] = cmd;
    } else {
        InterlockedAdd(counts[push.slot * 4 + 1], 1, dst);
        culled[base + dst] = cmd;
    }
}
