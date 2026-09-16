// Backend seam for the shaders: the same HLSL compiles to SPIR-V for the
// Vulkan backend (dxc -spirv, __spirv__ defined) and to DXIL for the
// D3D12 backend. Every shader includes this FIRST. It hides the places
// the two targets differ:
//
// 1. Resource binding. Vulkan uses the [[vk::binding(N, 0)]] attributes
//    that stay on every declaration (dxc ignores them for DXIL). D3D12
//    needs register assignments: binding N lives in register space N,
//    register 0. The suffix macros append them (the SPIR-V build expands
//    them to nothing):
//      REND_B(N)  read-only buffer  -> register(t0, spaceN)  (SRV)
//      REND_U(N)  RW buffer         -> register(u0, spaceN)  (UAV)
//      REND_T(N)  texture / TLAS    -> register(t0, spaceN)  (SRV)
//      REND_S(N)  sampler           -> register(s0, spaceN)
//    dxc rejects a mismatched register class, so a StructuredBuffer with
//    REND_U or an RWStructuredBuffer with REND_B fails to compile.
//    D3D12 keeps ONE resource state per buffer: a buffer some pass writes
//    through a UAV binding must be a UAV wherever it is read in the same
//    frame (20 aliases 23's rows, 11 and obb's 13 alias the pool the skin
//    pass writes, cull/proxy read the OBB table 33 the refine pass
//    writes). Those read sites declare REND_SHARED_BUFFER(T) /
//    REND_SHARED_BYTES: read-only on SPIR-V (byte-identical to before),
//    RW on DXIL. Everything genuinely read-only (objects, cameras, lights,
//    transforms, templates, bounds, LOD tables, skin inputs) stays an SRV
//    so the hardware can scalar-cache uniform loads — as UAVs those loads
//    cost ~10x (the first D3D12 sponza ran at 56 fps against 648).
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
#define REND_B(n)
#define REND_U(n)
#define REND_T(n)
#define REND_S(n)
#define REND_SHARED_BUFFER(T) StructuredBuffer<T>
#define REND_SHARED_BYTES ByteAddressBuffer

float4 rendClip(float4 clip) { return clip; }
uint rendInstanceIndex(uint svInstanceId) { return svInstanceId; }

#else // DXIL (D3D12)

#define REND_PUSH(T, name) ConstantBuffer<T> name : register(b0, space0)
#define REND_B(n) : register(t0, space##n)
#define REND_U(n) : register(u0, space##n)
#define REND_T(n) : register(t0, space##n)
#define REND_S(n) : register(s0, space##n)
#define REND_SHARED_BUFFER(T) RWStructuredBuffer<T>
#define REND_SHARED_BYTES RWByteAddressBuffer

struct RendDrawConstants {
    uint baseInstance; // the draw's firstInstance (root constant, b1)
};
ConstantBuffer<RendDrawConstants> rendDraw : register(b1, space0);

float4 rendClip(float4 clip) { return float4(clip.x, -clip.y, clip.z, clip.w); }
uint rendInstanceIndex(uint svInstanceId) { return svInstanceId + rendDraw.baseInstance; }

#endif

#endif // REND_BACKEND_HLSLI
