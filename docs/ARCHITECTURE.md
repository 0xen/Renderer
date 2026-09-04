# Architecture

A Windows-only Vulkan renderer built around **modularity**: windowing backends, Vulkan
feature sets, and render techniques all live behind interfaces we own.

## Layers

| Layer | Target | Depends on | Role |
|---|---|---|---|
| `engine/core` | `rend::core` | — | Logging, `rend::Result<T>` error handling, shared utilities. No exceptions cross module boundaries. |
| `engine/platform` | `rend::platform` | core | Presentation-target abstraction + windowing backends. |
| `engine/gpu` | `rend::gpu` | core | Vulkan layer (instance, adapter selection, device, swapchain, feature negotiation). *Placeholder.* |
| `engine/renderer` | `rend::renderer` | platform, gpu | High-level rendering (frame graph, passes). *Placeholder.* |
| `engine/` (umbrella) | `rend::engine` | all layers | The engine ships as **one DLL** (`rend.dll`); the per-layer object libraries above are linked into it. |
| `apps/viewer` | `viewer` | rend::engine | Milestone test-bed, growing into a scene viewer: `viewer <scene.xml>` loads an XML scene description (shaders, scene setup, models — loader lands with the renderer layer; pugixml planned). |

Dependency direction is strict and enforced by CMake link interfaces — nothing lower
links upward. Applications link only `rend::engine` (the DLL), never the internal
layer targets.

## Platform layer

The central concept is a **`PresentationTarget`**, not "a window": a surface that can be
presented to. `TargetDesc.style` selects one of:

- `Decorated` — regular OS window
- `Borderless` — undecorated, launcher-style (first-class requirement)
- `BorderlessTransparent` — undecorated + per-pixel alpha (Vulkan composite-alpha is driver-dependent)
- `FullscreenBorderless` — covers the monitor at desktop resolution
- `FullscreenExclusive` — owns the display mode (`VK_EXT_full_screen_exclusive`, future)

`IPlatformBackend` is the per-windowing-system interface: initialize/shutdown,
`createTarget`, `pumpEvents` (translated to portable `Event` values). Backend types
(SDL, Win32) never leak across it. Current backend: **SDL3** (static, FetchContent,
pinned tag). A raw Win32 backend is planned as a modularity proof.

The platform layer will expose exactly one Vulkan seam when the gpu layer lands:
required instance extensions + `createSurface`.

## Renderer layer (planned design, agreed 2026-09-04)

The game-facing side. Core principle: **the game describes resources; the renderer owns
placement, upload timing, and residency.** The two sides stay maximally separated.

- **Rendering instance** — the active backend (Vulkan first; the design keeps OpenGL/DX
  possible). Lives in `engine/gpu` as the explicit executor.
- **Render object descriptors** — the game defines each resource (texture, vertex
  buffer, generic memory pool, …) as data: type, format, usage, and *how to get its
  bytes* (a **data provider**: asset/file reference or callback — pull model, so
  registering a whole scene is cheap and bytes load only on demand).
- **Opaque handles** — the game holds small IDs, never pointers into renderer memory,
  so the renderer can lazily load, evict, defragment, and hot-reload freely.
- **Residency** — explicit states (Unloaded → CPU-resident → GPU-resident). The game
  sends *hints* (always-resident, priority-by-distance); the renderer owns the policy
  (e.g. load a building only when in range or requested for draw).
- **Minimal version fallback** — a descriptor may include a "minimal" representation of
  the object (lowest LOD mesh, top mips of a texture) that is small enough to keep
  resident eagerly. If a draw requests an object whose full data is still streaming,
  the renderer draws the minimal version instead — frames never stall on IO, and
  objects refine from coarse to full rather than popping from a placeholder. Optional
  per descriptor (a raw memory pool has no minimal form; it just reports non-resident).
  A per-descriptor flag (`keepMinimalResident`) requests that the minimal version be
  uploaded to VRAM at registration and pinned — guaranteeing the object can always draw
  immediately; without it, the minimal version streams on demand like any other data.
- **Deduplication** — the renderer never holds multiple VRAM copies of the same data.
  Each descriptor's data provider supplies a stable identity (asset ID / content hash);
  the renderer keeps an identity → resource table, so registering the same object twice
  returns the same underlying GPU resource, reference-counted (freed only when the last
  handle is released). Resources are distinct from *instances*: many scene instances
  (transform + per-instance data) share one set of GPU resources.
- **Intent, not implementation** — submissions are tagged with a small *semantic* stage
  vocabulary (opaque, transparent, overlay, …) describing what an object *is*, never how
  or in what order it is drawn. The renderer maps stages onto its current architecture
  (deferred, forward, ray-traced, …); it may, e.g., ray-trace all opaque geometry
  without any change to game-side submissions. Pass structure, techniques, and formats
  are the rendering implementation's dictate.
- **Offer-based capability negotiation** — the game *queries* available techniques
  (e.g. shadows → {ray-traced, raycast}, with quality/cost hints) and selects among
  what the renderer offers; it never demands features. Offers derive from the active
  backend's negotiated FeatureSet on the actual hardware — one chain of truth flowing
  upward. The game's selection is a preference the renderer honors; everything beneath
  it stays renderer-owned.
- **Two mechanisms, one philosophy** — message-style API for resource lifetime and
  streaming (create/destroy/hint: infrequent, async-friendly); structured draw
  lists / frame graph for per-frame submission (typed, batch-oriented — not generic
  messages).

## GPU layer (planned)

Plain C Vulkan API loaded via **volk**. Feature/extension requirements declared as data
(`FeatureSet`); physical-device selection scores adapters against it, so supporting
multiple feature tiers stays declarative.

### Base object classes (agreed 2026-09-04)

GPU primitives are small, logical, isolated classes — `Buffer`, `Texture`, and later
`Sampler`, `Shader`, `Pipeline`, `CommandContext`. Each wraps exactly one Vulkan
concept, owns its handle/memory (RAII), and carries **no policy**: no knowledge of
models, materials, streaming, residency, or rendering technique. They stay close to
the metal (explicit usage flags, memory location, no hidden uploads).

All composition happens above: a model is made of buffers (renderer layer); the
residency/dedup systems manage these objects; and a rendering path — forward,
deferred, ray tracing — is just a different arrangement of the same primitives. This
isolation is what keeps the future dynamic pipeline system highly adaptable: the
technique layer changes, the base vocabulary doesn't.

### Static command buffer / GPU-driven rendering model (agreed direction, 2026-09-04)

Goal: **avoid rebuilding command buffers** — ideally record once and reuse every frame
(screen resize excepted), with the CPU only writing buffers.

- **Indirect scene pass** — one `vkCmdDrawIndexedIndirect` stream with an entry per
  unique object. Loading/unloading an object toggles its `instanceCount` 0↔1 and never
  touches the command buffer. Each entry's `firstInstance` indexes a big per-object
  SSBO (in Vulkan `gl_InstanceIndex` starts at `firstInstance`), so every object finds
  its transforms/material data with no per-draw binds.
- **Evolution: compaction** — `vkCmdDrawIndexedIndirectCount` + a compute pass that
  compacts live/visible entries and writes the count; removes the zero-instance draw
  overhead at scale, command buffer still static.
- **Geometry mega-buffer** — all vertex/index data suballocated (free lists, size
  buckets) from one large device-local buffer, bound once; indirect entries carry
  `firstIndex`/`vertexOffset`. Defrag is a background task on the async transfer queue
  (move slices, patch indirect entries) — investigated as its own experiment, kept rare
  via suballocation rather than run per-frame.
- **Bindless textures** — descriptor indexing (core 1.2): one descriptor array bound
  once; per-object data holds texture indices. Prerequisite for static buffers.
- **Few pipelines** — über-shader-leaning, draws grouped per pipeline, since pipeline
  switches are recorded commands.
- **Sync model** — per-frame-in-flight regions for CPU-written data (indirect + object
  SSBO); the guarding barriers are identical each frame and live in the prerecorded
  buffer.
- **No secondary command buffers by default** — everything (scene pass, post-process)
  is prerecorded into the reused primary command buffer. Since nothing is re-recorded
  per frame, secondaries buy little and carry driver-dependent overhead; they are only
  revisited if the experiments surface a concrete need.
- **Feature requirements** (declared via FeatureSet): `multiDrawIndirect`,
  `drawIndirectFirstInstance`, `shaderDrawParameters`, descriptor indexing,
  draw-indirect-count.

An early rendering-test milestone validates this end-to-end: N objects in the
mega-buffer, add/remove/toggle without re-recording, measured against a naive
re-record-per-frame baseline.

## Conventions

- C++20, MSVC `/W4 /permissive-`, `.clang-format` (LLVM-derived, 4-space).
- Errors as `rend::Result<T>`; exceptions never cross module boundaries.
- `Scratch/` (gitignored) holds generated working material: `memories/`, `skills/`,
  `screenshots/`, `artifacts/`.
