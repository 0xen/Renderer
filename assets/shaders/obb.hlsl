// Incremental GPU OBB refinement (docs: scratch/memories — OBB slice):
// one baked dispatch per frame, recorded once into the static recordings
// BEFORE the cull dispatch, that claims the next few entries from a
// GPU-side counter and fits each an oriented bounding box from its pooled
// vertex data — PCA axes (Jacobi eigenvectors of the position covariance)
// plus exact extents from a second projection pass. The row's ready flag
// flips only when the OBB beats the world AABB's volume, so consumers
// (cull.hlsl frustum/occlusion-bypass/LOD, proxy.hlsl box generation) are
// never worse than the AABB they fall back to. Once the counter passes
// drawCount every later dispatch exits in nanoseconds — the scene renders
// with AABBs from frame 0 and boxes tighten over the first seconds with
// ZERO command-buffer rewrites and zero CPU involvement.
//
// One workgroup = one entry: 256 threads stride the entry's LOD0 index
// range (duplicate vertices weight the covariance by valence — harmless
// for an axis heuristic; extents are min/max so duplicates don't matter).
// Ineligible entries (local-mode/instanced, animated, transparent-only
// flags don't matter — anything without kBoundsObbEligible) burn their
// claim and leave the row unready forever, which is correct: runtime
// model slots recycle only among local-mode resources and animated
// meshes' bounds are CPU-authored per frame.

#include "backend.hlsli"

// Must match rend::gpu::DrawIndexedIndirect / cull.hlsl.
struct DrawCommand {
    uint baseInstance; // == firstInstance; D3D12 root constant (backend.hlsli)
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

// Must match ObjectBounds in cull.hlsl / proxy.hlsl / the viewer.
struct ObjectBounds {
    float4 bmin;
    float4 bmax;
};

struct ObbPush {
    uint drawCount; // live templates (claims past this exit immediately)
    uint slot;      // frame-in-flight index selecting the buffer regions
    uint capacity;  // entries per slot region (templates + bounds)
};
REND_PUSH(ObbPush, push);

// Must match kBoundsObbEligible in the viewer (bounds bmin.w bitmask):
// set only for world-baked, non-animated scene meshes whose pooled
// vertices never change after load.
static const uint kBoundsObbEligible = 4u;

static const uint kVertexStrideBytes = 32u;

[[vk::binding(3, 0)]] StructuredBuffer<DrawCommand> templates REND_U(3);
// The whole geometry pool (same binding the skin pass writes; read-only
// here): uint32 indices and interleaved pos3f/normal3f/uv2f vertices.
[[vk::binding(13, 0)]] ByteAddressBuffer pool REND_U(13);
[[vk::binding(22, 0)]] StructuredBuffer<ObjectBounds> bounds REND_U(22);

// OBB table (must match cull.hlsl / proxy.hlsl / the viewer's buffer):
//   bytes [0,16)  — control: [0] claim counter, rest pad;
//   row e at 16 + e*64 — float4 center (w = ready flag), then three
//   float4 {unit axis xyz, half extent w}.
// ONE global region (rows converge to load-time constants); cross-frame
// ordering rides the recorded barriers around the cull dispatch.
static const uint kObbHeaderBytes = 16u;
static const uint kObbRowBytes = 64u;
[[vk::binding(33, 0)]] RWByteAddressBuffer obbs REND_U(33);

static const uint kThreads = 256u;

groupshared uint gsEntry;
groupshared float gsReduce[kThreads];
groupshared float3 gsAxis0;
groupshared float3 gsAxis1;
groupshared float3 gsAxis2;

// Tree reductions over the shared scratch. Every thread of the group must
// call these together (uniform control flow around the barriers).
float reduceSum(float v, uint tid) {
    gsReduce[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            gsReduce[tid] += gsReduce[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gsReduce[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float reduceMin(float v, uint tid) {
    gsReduce[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            gsReduce[tid] = min(gsReduce[tid], gsReduce[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gsReduce[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float reduceMax(float v, uint tid) {
    gsReduce[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            gsReduce[tid] = max(gsReduce[tid], gsReduce[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gsReduce[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Fetch the t-th referenced vertex position of the entry, offset to keep
// the covariance sums small (float precision).
float3 fetchPosition(DrawCommand cmd, uint t, float3 refPoint) {
    const uint index = pool.Load((cmd.firstIndex + t) * 4);
    const uint vertex = uint(int(index) + cmd.vertexOffset);
    return asfloat(pool.Load3(vertex * kVertexStrideBytes)) - refPoint;
}

// One Jacobi rotation zeroing A[p][q] of the symmetric matrix, the
// rotation accumulated into V (columns converge to the eigenvectors).
void jacobiRotate(inout float3x3 A, inout float3x3 V, uint p, uint q) {
    const float apq = A[p][q];
    if (abs(apq) < 1e-12f) {
        return;
    }
    const float tau = (A[q][q] - A[p][p]) / (2.0f * apq);
    const float t = (tau >= 0.0f ? 1.0f : -1.0f) / (abs(tau) + sqrt(1.0f + tau * tau));
    const float c = rsqrt(1.0f + t * t);
    const float s = t * c;
    [unroll]
    for (uint k = 0; k < 3; ++k) {
        const float akp = A[k][p];
        const float akq = A[k][q];
        A[k][p] = c * akp - s * akq;
        A[k][q] = s * akp + c * akq;
    }
    [unroll]
    for (uint k2 = 0; k2 < 3; ++k2) {
        const float apk = A[p][k2];
        const float aqk = A[q][k2];
        A[p][k2] = c * apk - s * aqk;
        A[q][k2] = s * apk + c * aqk;
    }
    [unroll]
    for (uint k3 = 0; k3 < 3; ++k3) {
        const float vkp = V[k3][p];
        const float vkq = V[k3][q];
        V[k3][p] = c * vkp - s * vkq;
        V[k3][q] = s * vkp + c * vkq;
    }
}

[numthreads(kThreads, 1, 1)]
void CSMain(uint3 groupThread : SV_GroupThreadID) {
    const uint tid = groupThread.x;
    // Claim the next unprocessed entry. The load-then-add keeps the
    // counter from growing forever once refinement is done (a group of
    // racers may overshoot by a few once — harmless).
    if (tid == 0) {
        uint entry = 0xffffffffu;
        if (obbs.Load(0) < push.drawCount) {
            obbs.InterlockedAdd(0, 1, entry);
        }
        gsEntry = entry;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint e = gsEntry;
    if (e >= push.drawCount) {
        return; // nothing left to refine (or lost the boundary race)
    }

    const ObjectBounds b = bounds[push.slot * push.capacity + e];
    if (b.bmax.w != 0.0f || ((uint)b.bmin.w & kBoundsObbEligible) == 0 ||
        b.bmin.x > b.bmax.x) {
        return; // ineligible: the row stays unready, consumers keep the AABB
    }
    const DrawCommand cmd = templates[push.slot * push.capacity + e];
    const uint n = cmd.indexCount;
    if (n == 0) {
        return;
    }
    const float3 refPoint = (b.bmin.xyz + b.bmax.xyz) * 0.5f;

    // Pass 1: mean + covariance of the referenced positions.
    float3 sum = 0.0f;
    float sxx = 0.0f, syy = 0.0f, szz = 0.0f;
    float sxy = 0.0f, sxz = 0.0f, syz = 0.0f;
    for (uint t = tid; t < n; t += kThreads) {
        const float3 v = fetchPosition(cmd, t, refPoint);
        sum += v;
        sxx += v.x * v.x;
        syy += v.y * v.y;
        szz += v.z * v.z;
        sxy += v.x * v.y;
        sxz += v.x * v.z;
        syz += v.y * v.z;
    }
    const float invN = 1.0f / (float)n;
    const float3 mean = float3(reduceSum(sum.x, tid), reduceSum(sum.y, tid),
                               reduceSum(sum.z, tid)) * invN;
    float3x3 cov;
    cov[0][0] = reduceSum(sxx, tid) * invN - mean.x * mean.x;
    cov[1][1] = reduceSum(syy, tid) * invN - mean.y * mean.y;
    cov[2][2] = reduceSum(szz, tid) * invN - mean.z * mean.z;
    cov[0][1] = reduceSum(sxy, tid) * invN - mean.x * mean.y;
    cov[0][2] = reduceSum(sxz, tid) * invN - mean.x * mean.z;
    cov[1][2] = reduceSum(syz, tid) * invN - mean.y * mean.z;
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Diagonalize on one lane (8 cyclic Jacobi sweeps — far past
    // convergence for 3x3) and broadcast the axes. A degenerate
    // covariance leaves V at identity, i.e. the AABB frame.
    if (tid == 0) {
        float3x3 V = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
        for (uint sweep = 0; sweep < 8; ++sweep) {
            jacobiRotate(cov, V, 0, 1);
            jacobiRotate(cov, V, 0, 2);
            jacobiRotate(cov, V, 1, 2);
        }
        gsAxis0 = normalize(float3(V[0][0], V[1][0], V[2][0]));
        gsAxis1 = normalize(float3(V[0][1], V[1][1], V[2][1]));
        gsAxis2 = normalize(float3(V[0][2], V[1][2], V[2][2]));
    }
    GroupMemoryBarrierWithGroupSync();
    const float3 a0 = gsAxis0;
    const float3 a1 = gsAxis1;
    const float3 a2 = gsAxis2;

    // Pass 2: exact extents along the axes — the box CONTAINS every
    // referenced vertex regardless of how good the PCA frame is.
    float3 lo = 1e30f;
    float3 hi = -1e30f;
    for (uint t2 = tid; t2 < n; t2 += kThreads) {
        const float3 v = fetchPosition(cmd, t2, refPoint);
        const float3 l = float3(dot(v, a0), dot(v, a1), dot(v, a2));
        lo = min(lo, l);
        hi = max(hi, l);
    }
    lo = float3(reduceMin(lo.x, tid), reduceMin(lo.y, tid), reduceMin(lo.z, tid));
    hi = float3(reduceMax(hi.x, tid), reduceMax(hi.y, tid), reduceMax(hi.z, tid));

    if (tid == 0) {
        const float3 mid = (lo + hi) * 0.5f;
        // Tiny absolute + relative pad against float rounding in the
        // consumers' plane/projection math.
        const float3 ext = (hi - lo) * 0.5f * 1.0001f + 1e-3f;
        const float3 center = refPoint + a0 * mid.x + a1 * mid.y + a2 * mid.z;
        const float3 aabbHalf = (b.bmax.xyz - b.bmin.xyz) * 0.5f;
        const float obbVol = ext.x * ext.y * ext.z;
        const float aabbVol = aabbHalf.x * aabbHalf.y * aabbHalf.z;
        // Keep whichever box is smaller: ready stays 0 when PCA didn't
        // beat the AABB, so the result is never worse than today.
        if (obbVol < aabbVol * 0.98f) {
            const uint row = kObbHeaderBytes + e * kObbRowBytes;
            obbs.Store4(row, asuint(float4(center, 1.0f)));
            obbs.Store4(row + 16, asuint(float4(a0, ext.x)));
            obbs.Store4(row + 32, asuint(float4(a1, ext.y)));
            obbs.Store4(row + 48, asuint(float4(a2, ext.z)));
        }
    }
}
