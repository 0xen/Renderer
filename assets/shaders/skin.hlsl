// GPU skinning + morph pass: one thread per vertex, reads the bind-pose
// vertex from the geometry pool, applies morph-target deltas and the
// 4-joint skin, and writes the posed vertex into this frame slot's
// destination region of the same pool. The indirect draw stream then
// consumes posed vertices without knowing anything moved — static command
// buffers and GPU compaction stay untouched.

struct PushConstants {
    uint srcVertex;        // bind-pose base, vertex-stride units from pool start
    uint dstVertexBase;    // slot 0 destination base (slot advances by vertexCount)
    uint vertexCount;
    uint skinVertexOffset; // into skinVertices; 0xffffffff = not skinned
    uint jointBase;        // into joints, before the slot region offset
    uint morphBase;        // into morphDeltas, vertex-per-target units
    uint morphTargetCount;
    uint morphWeightOffset; // into morphWeights, before the slot region offset
    uint slot;              // frame-in-flight index (patched at record time)
    uint jointSlotStride;   // total joints per slot region
    uint weightSlotStride;  // total morph weights per slot region
};
[[vk::push_constant]] PushConstants pc;

static const uint kVertexStrideBytes = 32u;
static const uint kNoSkin = 0xffffffffu;

// The whole geometry pool: interleaved pos3f/normal3f/uv2f vertices.
[[vk::binding(13, 0)]] RWByteAddressBuffer vertices;

// Scalar members only: a float4 here would std430-align to 16 and skew
// the stride (32) away from the CPU's tightly packed 24 bytes.
struct SkinVertex {
    uint joints01; // two u16 joint indices
    uint joints23;
    float weight0;
    float weight1;
    float weight2;
    float weight3;
};
[[vk::binding(14, 0)]] StructuredBuffer<SkinVertex> skinVertices;

struct JointMatrix {
    column_major float4x4 m; // model * world[joint] * inverseBind
};
[[vk::binding(15, 0)]] StructuredBuffer<JointMatrix> joints;

// 6 floats per vertex per target: position delta xyz, normal delta xyz.
[[vk::binding(16, 0)]] StructuredBuffer<float> morphDeltas;
[[vk::binding(17, 0)]] StructuredBuffer<float> morphWeights;

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pc.vertexCount) {
        return;
    }
    const uint srcByte = (pc.srcVertex + id.x) * kVertexStrideBytes;
    float3 position = asfloat(vertices.Load3(srcByte));
    float3 normal = asfloat(vertices.Load3(srcByte + 12));

    for (uint t = 0; t < pc.morphTargetCount; ++t) {
        const float w =
            morphWeights[pc.morphWeightOffset + pc.slot * pc.weightSlotStride + t];
        const uint base = (pc.morphBase + t * pc.vertexCount + id.x) * 6;
        position += w * float3(morphDeltas[base], morphDeltas[base + 1], morphDeltas[base + 2]);
        normal += w * float3(morphDeltas[base + 3], morphDeltas[base + 4], morphDeltas[base + 5]);
    }

    if (pc.skinVertexOffset != kNoSkin) {
        const SkinVertex skin = skinVertices[pc.skinVertexOffset + id.x];
        const uint jointSlot = pc.jointBase + pc.slot * pc.jointSlotStride;
        const float4x4 m =
            joints[jointSlot + (skin.joints01 & 0xffffu)].m * skin.weight0 +
            joints[jointSlot + (skin.joints01 >> 16)].m * skin.weight1 +
            joints[jointSlot + (skin.joints23 & 0xffffu)].m * skin.weight2 +
            joints[jointSlot + (skin.joints23 >> 16)].m * skin.weight3;
        position = mul(m, float4(position, 1.0f)).xyz;
        normal = mul((float3x3)m, normal);
    }

    const float len = length(normal);
    if (len > 0.0f) {
        normal /= len;
    }
    const uint dstByte = (pc.dstVertexBase + pc.slot * pc.vertexCount + id.x) * kVertexStrideBytes;
    vertices.Store3(dstByte, asuint(position));
    vertices.Store3(dstByte + 12, asuint(normal));
    vertices.Store2(dstByte + 24, vertices.Load2(srcByte + 24)); // uv passes through
}
