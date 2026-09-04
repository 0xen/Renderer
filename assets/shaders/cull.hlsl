// Compaction pass for the GPU-driven scene (docs/ARCHITECTURE.md): one
// thread per registered object reads its draw template and appends it to
// the compacted indirect list if visible. The scene pass then draws via
// vkCmdDrawIndexedIndirectCount, so the visible count never touches the
// CPU. Visibility today is the template's instanceCount flag; frustum
// culling against per-object bounds slots in here later.

// Must match rend::gpu::DrawIndexedIndirect (VkDrawIndexedIndirectCommand).
struct DrawCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

struct CullPush {
    uint drawCount; // templates per frame slot
    uint slot;      // frame-in-flight index selecting the buffer regions
};
[[vk::push_constant]] CullPush push;

[[vk::binding(3, 0)]] StructuredBuffer<DrawCommand> templates;
[[vk::binding(4, 0)]] RWStructuredBuffer<DrawCommand> compacted;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> counts; // zeroed before dispatch

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= push.drawCount) {
        return;
    }
    const uint base = push.slot * push.drawCount;
    DrawCommand cmd = templates[base + id.x];
    if (cmd.instanceCount == 0) {
        return; // hidden (or later: culled)
    }
    uint dst;
    InterlockedAdd(counts[push.slot], 1, dst);
    compacted[base + dst] = cmd;
}
