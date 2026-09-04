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

## GPU layer (planned)

Plain C Vulkan API loaded via **volk**. Feature/extension requirements declared as data
(`FeatureSet`); physical-device selection scores adapters against it, so supporting
multiple feature tiers stays declarative.

## Conventions

- C++20, MSVC `/W4 /permissive-`, `.clang-format` (LLVM-derived, 4-space).
- Errors as `rend::Result<T>`; exceptions never cross module boundaries.
- `Scratch/` (gitignored) holds generated working material: `memories/`, `skills/`,
  `screenshots/`, `artifacts/`.
