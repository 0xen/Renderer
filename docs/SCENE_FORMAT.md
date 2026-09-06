# Scene & Pipeline XML Format

The viewer loads a scene description XML (`viewer <scene.xml>`); scene nodes (e.g. a
model) reference **pipeline definition files** like the example below. Loader lands
with the renderer layer (roadmap #8, pugixml).

## Pipeline definition

Defines *what the shader consumes*: vertex input layout, named descriptor table
layouts, and stages pointing at precompiled SPIR-V. Deliberately absent — blend,
depth/stencil, cull, attachment formats: those follow from the semantic stage and the
active rendering technique (see the intent/offer model in ARCHITECTURE.md), so the
renderer supplies them, not the asset.

```xml
<?xml version="1.0"?>
<Pipeline type="Graphics" topology="Triangle">
    <VertexBindings>
        <Binding binding="0" stride="8"  rate="INPUT_RATE_VERTEX" />   <!-- World Position -->
        <Binding binding="1" stride="24" rate="INPUT_RATE_VERTEX" />   <!-- UV/SpriteSheetID -->
        <Binding binding="2" stride="8"  rate="INPUT_RATE_INSTANCE" /> <!-- Chunk Position -->
    </VertexBindings>

    <VertexInputAttributes>
        <Attribute location="0" binding="0" format="R32G32_SFLOAT" offset="0" />
        <Attribute location="1" binding="1" format="R32G32_SFLOAT" offset="0" />
        <Attribute location="5" binding="2" format="R32G32_SFLOAT" offset="0" />
    </VertexInputAttributes>

    <Descriptors>
        <!-- Names reference engine-registered resource table layouts; the XML
             never defines binding internals. -->
        <Descriptor name="CameraResourceTableLayout" />
        <Descriptor name="GlobalSamplerArrayResourceTableLayout" />
    </Descriptors>

    <Stages>
        <!-- Logical, backend-agnostic paths: no extension. The active backend
             resolves the artifact (.spv for Vulkan, .dxil for DirectX, ...). -->
        <Stage entrypoint="main" stage="Vertex"   path="data/Shaders/Terrain/vert" />
        <Stage entrypoint="main" stage="Fragment" path="data/Shaders/Terrain/frag" />
    </Stages>
</Pipeline>
```

- Formats/rates use Vulkan-style names (`R32G32_SFLOAT`, `INPUT_RATE_VERTEX`,
  `INPUT_RATE_INSTANCE`) as a *neutral vocabulary* describing data layout; each
  backend maps them mechanically (e.g. → `DXGI_FORMAT_R32G32_FLOAT`). Topology
  values: `Triangle`, … (extended as needed).
- Shader artifacts are compiled **offline at build time**, one binary per backend.
  Planned toolchain: single-source HLSL through **dxc** (emits both SPIR-V and DXIL
  natively), with **SPIRV-Cross** available for converting existing SPIR-V to other
  backends' source. No runtime shader compilation.
- Planned: SPIR-V reflection (SPIRV-Reflect) at load validates that vertex layout and
  descriptor references match the shader — mismatches are load-time errors, not
  silent corruption.

## Scene file

Loaded by the standalone `assetio` project (pugixml). Mesh paths resolve against the
scene file's directory; `Shader` paths resolve against the engine's data root. Vector
attributes are whitespace-separated floats. `Camera` and `Transform` are optional and
default sensibly. Model formats are dispatched by extension through assetio's importer
registry (currently glTF 2.0 via cgltf; new formats = new importer, nothing else changes).

```xml
<Scene name="crytek_sponza">
    <Camera position="-8 1.7 0" target="0 1.7 0" fovDegrees="60" />
    <Light type="directional" direction="0.35 -1.0 0.2" color="1.0 0.96 0.88"
           intensity="1.0" castsShadows="true" />
    <Model name="sponza">
        <Shader path="pipelines/opaque.xml" />
        <Mesh path="model/Sponza.gltf" />
        <Transform position="0 0 0" rotation="0 0 0" scale="1 1 1" />
    </Model>
</Scene>
```

`Model` takes an optional `reflective="true"` attribute — a semantic surface tag in
the intent-model sense: it says the object *is* mirror-like, never how to render it.
The renderer offers reflection techniques for such objects (reflection probe on every
device, ray traced where ray queries exist — cost scales with screen coverage); the
technique is a user/renderer choice, superseded entirely when primary visibility is
ray traced.

`ReflectionProbe position="x y z"` (optional, zero or more) marks where a probe-based
renderer captures its environment cubemap at load time. Like lights it describes the
scene, not the technique. Only the first probe is used for now (single probe, no
blending); with none listed the renderer defaults to the scene AABB's center. The
capture is static: animated meshes bake at the bind pose, lighting never re-captures,
and there is no parallax correction — probes are approximate by design, ray traced
reflections are the exact tier above.

`Script path="scripts/foo.py"` (optional, zero or more) names Python scripts that run
alongside the scene, resolved against the scene file's directory like mesh paths. The
viewer feeds them to the optional embedded Python host (`rend_pyhost.dll`), which runs
them in order on the interpreter's own thread; scripts `import rend` and speak the
renderer message queue (`load_model` / `set_transform` / `unload_model`, `poll_events`
/ `wait_model_ready`, `should_quit`, `log`). A scene that lists no scripts involves no
Python at all — the host DLL is never even loaded, so it and the CPython runtime may
be absent. assetio only parses the paths; it never executes code.

`Scene` takes an optional `loading` attribute describing how the app should
*present* asset loading, never how it schedules it (loading is nonblocking to the
renderer either way): `wait` (the default) holds the scene behind a loading screen
with a progress bar until every asset has landed; `streaming` shows the scene
immediately, assets popping in as they arrive (textures sample a neutral white
until theirs lands), with a small inline progress bar.

`Light` describes what the light *is* (direction points from the light toward the
scene), never the rendering technique — shadow maps vs ray tracing is the renderer's
offer, per the intent/offer model in ARCHITECTURE.md. `type` is `directional` only for
now; every attribute is optional with sensible defaults. Multiple lights are parsed
into a list (the GPU consumes lights as a buffer, so more types scale without new
passes).

Example scene: `C:\github\scenes\crytek_sponza\crytek_sponza.xml` (kept outside this
repo; Crytek Sponza in the Khronos glTF conversion).
