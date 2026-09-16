// Backend seam for the shaders: the same HLSL compiles to SPIR-V for the
// Vulkan backend (dxc -spirv, __spirv__ defined) and to DXIL for the
// D3D12 backend. Every shader includes this FIRST. It hides the three
// places the two targets differ:
//
// 1. Resource binding. Vulkan uses the [[vk::binding(N, 0)]] attributes
//    that stay on every declaration (dxc ignores them for DXIL). D3D12
//    needs register assignments: binding N lives in register space N,
//    register 0 (REND_U / REND_T / REND_S append them; the SPIR-V build
//    expands them to nothing). Storage buffers are UAVs (u registers) in
//    EVERY stage on D3D12 so that they sit in one resource state
//    (UNORDERED_ACCESS) for all shader access — the neutral barrier
//    vocabulary cannot name which buffer a dispatch writes, so read-only
//    SRV views would need per-buffer state transitions the backend
//    cannot place. StructuredBuffer/ByteAddressBuffer are therefore
//    macro-mapped to their RW forms for DXIL only; the SPIR-V build keeps
//    the read-only (NonWritable) declarations byte for byte.
// 2. Push constants: a push block on Vulkan, root constants at b0 on D3D12
//    (REND_PUSH). All push structs are scalar-only, so the cbuffer packing
//    matches the push layout dword for dword.
// 3. Clip space. rend::math's projections bake Vulkan's +Y-down clip
//    flip; DXIL output negates clip y in every vertex shader (rendClip)
//    so the D3D12 framebuffer is row-identical to the Vulkan one, and
//    every texture-space convention (shadow-map UVs, G-buffer loads)
//    survives unchanged. Cull mode is NONE everywhere, so the reversed
//    winding is harmless.
//    SV_InstanceID excludes the draw's base instance on D3D12 (Vulkan's
//    InstanceIndex includes it); rendInstanceIndex adds the base back
//    from a root constant at b1 that the backend sets per draw (from the
//    DrawCommand's leading baseInstance word through the ExecuteIndirect
//    command signature, or directly for CPU-issued draws).
#ifndef REND_BACKEND_HLSLI
#define REND_BACKEND_HLSLI

#ifdef __spirv__

#define REND_PUSH(T, name) [[vk::push_constant]] T name
#define REND_U(n)
#define REND_T(n)
#define REND_S(n)

float4 rendClip(float4 clip) { return clip; }
uint rendInstanceIndex(uint svInstanceId) { return svInstanceId; }

#else // DXIL (D3D12)

#define StructuredBuffer RWStructuredBuffer
#define ByteAddressBuffer RWByteAddressBuffer
#define REND_PUSH(T, name) ConstantBuffer<T> name : register(b0, space0)
#define REND_U(n) : register(u0, space##n)
#define REND_T(n) : register(t0, space##n)
#define REND_S(n) : register(s0, space##n)

struct RendDrawConstants {
    uint baseInstance; // the draw's firstInstance (root constant, b1)
};
ConstantBuffer<RendDrawConstants> rendDraw : register(b1, space0);

float4 rendClip(float4 clip) { return float4(clip.x, -clip.y, clip.z, clip.w); }
uint rendInstanceIndex(uint svInstanceId) { return svInstanceId + rendDraw.baseInstance; }

#endif

#endif // REND_BACKEND_HLSLI
