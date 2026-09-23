# Architecture

Renderer is a Windows renderer built around backend-neutral public interfaces: every
layer above the GPU layer talks to `gpu::Buffer`, `gpu::Image`, `gpu::Pipeline` and the
rest without ever naming a graphics-API type, and the GPU layer itself offers the same
surface over either Vulkan 1.3 or D3D12. The guiding split running through the whole
engine is that the game (or, today, the viewer app) describes *intent* — what an object
is, what technique it would prefer — and the renderer decides placement, technique and
timing. A game submission never demands a code path; it queries what the active backend
can offer and picks among the offers.

## Layers

| Layer | Target | Depends on | Role |
|---|---|---|---|
| `engine/core` | `rend::core` | — | Logging, `rend::Result<T>` error handling, minimal math (no GLM), executable-relative paths. No exceptions cross module boundaries. |
| `engine/platform` | `rend::platform` | core | Presentation-target abstraction and windowing backend (SDL3). |
| `engine/gpu` | `rend::gpu` | core | The GPU executor: instance/adapter/device/swapchain, the base GPU object classes, the bindless descriptor table, and the frame loop. Vulkan and D3D12 backends behind one neutral interface. |
| `engine/renderer` | `rend::renderer` | platform, gpu | Cross-process-shaped message queue and the settings registry that mediates every user-facing graphics option. |
| `engine/` (umbrella) | `rend::engine` | all layers above | The engine ships as **one DLL**, `rend.dll`, `WINDOWS_EXPORT_ALL_SYMBOLS`; the per-layer object libraries above are linked into it. Applications link only `rend::engine`, never the internal layer targets. |
| `assetio/` | `rend::assetio` | rend::engine (core utilities only) | A **separate DLL**. Reads scene XML and model/texture files into plain CPU-side structs; the engine is fed the data and never touches a file format itself. |
| `pyhost/` | `rend::pyhost` (`rend_pyhost.dll`) | rend::engine | An optional embedded-CPython DLL that drives the renderer's message queue from scene scripts. Nothing links it; the viewer `LoadLibrary`s it only when a scene declares `<Script>` nodes. |
| `apps/viewer` | `viewer` | rend::engine, rend::assetio | The application. Drives `rend::gpu` directly and owns everything the renderer layer has not yet absorbed — scene setup, instancing, streaming, animation, collision and the settings UI. |

Dependency direction is strict and enforced by CMake link interfaces: core is the only
layer with no engine dependency; platform and gpu both depend on core alone; renderer
depends on platform and gpu; nothing lower ever links upward. `assetio` and `pyhost` are
siblings of the engine DLL, not layers inside it — the engine never links back to either,
so asset formats and the scripting runtime can change without touching `rend.dll`.

## Platform layer

The central concept is a **`PresentationTarget`**, not "a window": a surface that can be
presented to, created from a `TargetDesc` whose `style` selects one of `Decorated`
(a regular OS window), `Borderless` (undecorated), `BorderlessTransparent` (undecorated
with per-pixel alpha; composite-alpha support is driver-dependent), `FullscreenBorderless`
(covers the monitor at desktop resolution), or `FullscreenExclusive` (owns the display
mode; not yet implemented).

`IPlatformBackend` is the per-windowing-system interface: initialize/shutdown,
`createTarget`, `pumpEvents` (returns a flat batch of portable `Event` values, not
callbacks), and `setRelativeMouseMode`. Backend-specific types never leak across it — the
current backend, SDL3, is linked in statically and forward-declared behind a factory
function, so no SDL header reaches a consumer of `rend::platform`. The Vulkan seam is
exactly two methods on the backend: the required instance extensions and surface
creation; a raw HWND is exposed for D3D12 the same way, as a native handle rather than a
distinct interface.

## GPU layer

`rend::gpu` is the executor: instance, adapter selection, device, swapchain, the base GPU
object classes, the bindless descriptor table and the frame loop (`FrameRenderer`). It
carries no policy — no knowledge of models, materials, or rendering technique lives here;
the viewer decides modes and techniques and passes them down through a `DrawBatch`.

**Public headers contain no graphics-API types**, not even forward-declared handles.
Every public class in `engine/gpu/include/rend/gpu/*.h` (`buffer.h`, `image.h`,
`shader.h`, `pipeline.h`, `memory_pool.h`, `acceleration_structure.h`,
`descriptor_table.h`, `frame_renderer.h`, `command_context.h`, `feature_set.h`,
`device.h`, `instance.h`, `swapchain.h`, `transfer.h`, `texture_uploader.h`,
`probe_capture.h`, `api.h`) is an abstract base holding only neutral state. An
`enum class Api { Vulkan, D3D12 }` (`api.h`) selects the backend at `Instance::create`;
every other object has a small static factory that dispatches to the chosen backend.

- **Vulkan backend** (`engine/gpu/src/vulkan/`) loads the API through volk and requires
  Vulkan 1.3. The private header `vulkan_types.h` defines one `VulkanX final : public X`
  per base class, plus `vk()` downcast helpers; this is the only place a Vulkan handle
  is declared next to the public interface.
- **D3D12 backend** (`engine/gpu/src/d3d12/`) mirrors the same shape in `d3d12_types.h`.
  It shares one root signature and one shader-visible descriptor heap across every
  pipeline, tracks buffer and image resource states per command list (D3D12 buffers
  decay to `COMMON` after every submission, so states are reasserted at the start of
  each recording), and reads the same compiled shaders as Vulkan (`.dxil` beside every
  `.spv`). Ray tracing, cube images and reflection-probe capture are not implemented on
  this backend yet; the device reports the acceleration-structure feature absent so the
  application falls back to the raster path automatically.

**Base objects** — `Buffer`, `Image`, `Shader`, `Pipeline`, `MemoryPool`,
`AccelerationStructure` — are small, single-purpose RAII wrappers around one GPU concept
each, with explicit usage flags and memory location and no hidden uploads or policy.
Composition happens above them: a model is buffers plus a pipeline; a residency or
deduplication system, where one exists, manages these objects rather than being one of
them.

**Bindless descriptor table**: one large, mostly-partially-bound descriptor set holds
every resource the frame needs — bindless textures, per-frame buffers, the acceleration
structure, G-buffer targets and consumer-supplied user bindings — so pipelines bind the
table once and index into it rather than rebinding per draw. The storage-buffer pool size
is a fixed, hand-maintained count that every new storage binding must account for on both
backends.

**`FrameRenderer` and static command recording**: the frame loop and all per-frame scene
recording live here. The engine favors **not** rebuilding command buffers per frame:
draws are issued through an indirect stream so that loading, unloading and toggling
visibility touch only per-object data (an instance count, a buffer entry), never the
recorded commands themselves. Where the workload allows it, the whole frame — the scene
pass, culling, shadows and post-process — is recorded once per swapchain image and reused
until something that changes the recording itself (a pipeline, a feature toggle, the
presence of a pass) invalidates it explicitly.

**The GPU-driven model** built on top of that static recording:

- **Indirect draws** — one draw-indirect stream per object, expanded through
  `vkCmdDrawIndexedIndirectCount`/its D3D12 equivalent so an off-screen or unloaded
  object costs a zero instance count rather than a command-buffer edit.
- **Cull compute** — a compute pass frustum-culls, LOD-selects and (optionally)
  occlusion-culls every entry each frame, compacting survivors into the stream the
  indirect draw reads, before the draw is even recorded.
- **LOD** — a mesh's index buffer is a chain of coarser levels built at load time; the
  cull pass picks a level by projected screen error, never by CPU distance checks.
- **Occlusion proxies** — a small proxy geometry pass (one box per object, refined boxes
  once available) marks visibility per entry from the previous frame's depth, and the
  cull pass reads it with a short hysteresis window to avoid single-frame popping.
- **OBB refinement** — eligible world-space entries begin culled against an axis-aligned
  box and are refined in the background, a few per frame, to a tighter oriented box
  computed from their geometry; culling and the occlusion proxy switch to the tighter
  box once it is ready and never revert.

Defragmentation of the geometry memory pool — moving live suballocations to reduce
fragmentation — is not implemented; the pool is sized once and suballocated with a
coalescing free list instead.

## Renderer layer

The renderer layer is the part of the game-facing design that is not yet absorbed into
its own object library — its message schema and settings registry live in
`engine/renderer`, while the majority of the intent/offer machinery is still exercised
directly inside `apps/viewer`.

- **Message queue** (`messages.h`, `message_queue.h`) — a strict-POD schema (fixed-size
  char arrays, no pointers or STL) designed to cross a process or scripting boundary
  cleanly. Commands (load/unload a model, set a transform, set a light, set a setting)
  are pushed by a per-producer `Sender` and committed atomically on `flush()`; events
  (a model becoming ready, a setting's new state, a rejected change) broadcast to every
  registered receiver. Loads are asynchronous by contract: a model does not exist until
  its ready event reports success.
- **Settings registry** (`settings.h`) — the single authority for user-facing graphics
  options. Each setting is a named slot (a set of string option tokens, or a float
  range) with an apply callback that is where the corresponding renderer state actually
  changes; a slot can be overridden by another (selecting a ray-traced primary mode
  overrides the shadow and reflection technique slots beneath it), and an overridden
  slot reports no selectable options rather than disappearing. Both the debug UI and any
  external script route every setting change through the same registry, so they cannot
  disagree with each other.
- **Intent, not implementation** — submissions carry a small semantic vocabulary (an
  object is opaque, transparent, or an overlay) describing what it *is*; the renderer
  maps that vocabulary onto whatever pass structure it currently uses. Ray-tracing every
  opaque object instead of rasterizing it is a renderer-side decision that changes no
  game-side submission.
- **Offer-based technique negotiation** — the game queries what the active backend and
  hardware can currently do and chooses among the offers; it never demands a feature
  directly. In practice this is the settings registry's option lists, each backed by the
  device's negotiated feature set: primary rays as raster or ray-traced, shadows as
  cascaded or ray-traced, reflections as a baked probe or ray-traced. An offer that stops
  being available (unsupported hardware, a feature that failed to enable) removes itself
  from the list rather than being silently accepted and failing later.

## Asset IO

`rend::assetio` is a separate DLL from the engine and links only its core utilities
(logging, `Result`) — the engine never links back to it, so the choice of file formats
never reaches `rend.dll`. It has three entry points: parsing a scene XML file into a
`SceneDesc` and, if requested, importing every model it references into a `LoadedScene`;
importing a single model file into a backend-agnostic `ModelData` through an
`ImporterRegistry` (new formats implement `IModelImporter` and register with it, nothing
else in the engine changes); and loading a single image file into `TextureData`. Textures
are not decoded by scene loading — a material carries texture paths only, and the
consumer (the viewer) decodes and streams them on its own schedule. The one built-in
model importer reads glTF and glb through cgltf; scene XML is read through pugixml and is
documented in `docs/SCENE_FORMAT.md`.

## Shaders

Every shader is written once, in HLSL, and compiled offline by dxc to both SPIR-V (for
Vulkan) and DXIL (for D3D12) — there is no runtime shader compilation on either backend.
A shared header, `assets/shaders/backend.hlsli`, is included first by every shader and
hides the difference between the two targets: it maps a single binding declaration to
both a Vulkan descriptor binding and a D3D12 register in a matching register space,
supplies a push-constant macro that becomes a root constant on D3D12, and normalizes clip
space and instance indexing so the same source produces a row-identical framebuffer on
both APIs. A binding declared without going through this seam compiles cleanly but binds
the wrong slot on D3D12, so the seam is the only supported way to add one.

## Python host

`pyhost` embeds CPython to let scene scripts drive the renderer's message queue —
loading and moving models, setting lights and camera, reading back settings and
animation state — entirely through the same asynchronous command/event schema any other
producer uses. It is optional, built as its own DLL, and loaded by the viewer only when a
scene needs it. The full scripting surface is documented in `docs/SCRIPTING.md`.

## Conventions

- C++20 throughout, `/W4 /permissive-` on MSVC (`-Wall -Wextra -Wpedantic` elsewhere),
  enforced through a shared `rend_warnings` interface target.
- `.clang-format` (LLVM-derived, 4-space indent) governs formatting.
- Errors are returned as `rend::Result<T>` (a `Result<void>` specialization covers
  operations with no value); exceptions never cross a module boundary.
