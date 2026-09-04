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
| `apps/sandbox` | `sandbox` | all | Test-bed exercising each milestone. |

Dependency direction is strict and enforced by CMake link interfaces — nothing lower
links upward.

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
- **Two mechanisms, one philosophy** — message-style API for resource lifetime and
  streaming (create/destroy/hint: infrequent, async-friendly); structured draw
  lists / frame graph for per-frame submission (typed, batch-oriented — not generic
  messages).

## GPU layer (planned)

Plain C Vulkan API loaded via **volk**. Feature/extension requirements declared as data
(`FeatureSet`); physical-device selection scores adapters against it, so supporting
multiple feature tiers stays declarative.

## Conventions

- C++20, MSVC `/W4 /permissive-`, `.clang-format` (LLVM-derived, 4-space).
- Errors as `rend::Result<T>`; exceptions never cross module boundaries.
- `Scratch/` (gitignored) holds generated working material: `memories/`, `skills/`,
  `screenshots/`, `artifacts/`.
