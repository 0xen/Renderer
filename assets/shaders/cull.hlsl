// Compaction + frustum-cull pass for the GPU-driven scene
// (docs/ARCHITECTURE.md): one thread per registered object reads its draw
// template and appends it to TWO compacted indirect lists:
//   - binding 4 (+ counts[slot*2]): every live entry — the shadow passes
//     draw this one, because a caster outside the CAMERA frustum must
//     still cast into the view;
//   - binding 21 (+ counts[slot*2+1]): live entries whose world AABB
//     intersects the camera frustum — the main scene pass draws this one.
// The scene pass then draws via vkCmdDrawIndexedIndirectCount, so the
// visible count never touches the CPU.

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

// Must match ObjectBounds in the viewer: world-space AABB per draw entry
// (runtime models: union over their instances). bmin.w = 1 marks entries
// the frustum test must never drop (animated meshes — their pose can
// exceed the bind-pose bounds).
struct ObjectBounds {
    float4 bmin;
    float4 bmax;
};

struct CullPush {
    uint drawCount; // live templates this frame (may grow at runtime)
    uint slot;      // frame-in-flight index selecting the buffer regions
    uint capacity;  // entries per slot region — the fixed stride; runtime
                    // model loads change drawCount but never this
    uint flags;     // bit 0: frustum culling enabled
};
[[vk::push_constant]] CullPush push;

static const uint kCullFrustum = 1u;

[[vk::binding(3, 0)]] StructuredBuffer<DrawCommand> templates;
[[vk::binding(4, 0)]] RWStructuredBuffer<DrawCommand> compacted;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> counts; // 2 per slot, zeroed before dispatch
[[vk::binding(6, 0)]] StructuredBuffer<CameraData> cameras;
[[vk::binding(21, 0)]] RWStructuredBuffer<DrawCommand> culled;
[[vk::binding(22, 0)]] StructuredBuffer<ObjectBounds> bounds;

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
    InterlockedAdd(counts[push.slot * 2], 1, dst);
    compacted[base + dst] = cmd;

    if ((push.flags & kCullFrustum) != 0) {
        const ObjectBounds b = bounds[base + id.x];
        if (b.bmin.w == 0.0f &&
            !inFrustum(b.bmin.xyz, b.bmax.xyz, cameras[push.slot].viewProj)) {
            return; // outside the view — shadows above still drew it
        }
    }
    InterlockedAdd(counts[push.slot * 2 + 1], 1, dst);
    culled[base + dst] = cmd;
}
