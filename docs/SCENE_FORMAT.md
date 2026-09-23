# Scene & Pipeline XML Format

The viewer loads a scene description XML (`viewer <scene.xml>`); scene nodes (e.g. a
model) reference **pipeline definition files** like the example below.

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
default sensibly. `Camera` also takes an optional fly-in: `flyFrom="x y z"` spawns the
camera there and eases it into `position` over `flySeconds` (default 8) while looking
at `target`; free flight takes over when it lands. Model formats are dispatched by extension through assetio's importer
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

`Model` also takes an optional `lod="off"` attribute (default on). With LOD on, the
viewer bakes simplified index chains for the model's meshes at load and the GPU cull
pass picks a level per frame from each mesh's distance — full detail up close, fewer
triangles far away, chosen so the simplification error stays under about a pixel.
`lod="off"` locks the model to full detail (hero assets, close-up subjects); animated
meshes and meshes under a size floor are always full detail regardless.

`Model` may carry an optional `<Animation clip="name"/>` child naming which animation
clip an animated model starts on:

```xml
<Model name="xbot">
    <Shader path="pipelines/opaque.xml" />
    <Mesh path="model/Xbot/Xbot.gltf" />
    <Animation clip="idle">
        <Clip name="sad_pose" loop="false" />
    </Animation>
</Model>
```

`clip` is a clip name as it appears in the source file (glTF `animations[].name`);
omitting the element — or naming a clip the model does not have, which warns — starts
the first clip, which is what every scene did before this element existed.

Clips loop by default. A `<Clip name="..." loop="false"/>` child marks one that must
**not**: it plays once and holds its last pose. This matters because glTF stores no
loop flag, so a one-shot gesture, or a two-keyframe A-to-B pose clip, is
indistinguishable from a cycle until it wraps — and looping one makes it snap back to
its first pose every cycle. A 0.07 s Mixamo pose clip looped this way judders about
fifteen times a second. The viewer WARNS at load for any looping clip whose first and
last rotation keys disagree, naming the clip and the `<Clip>` line to add, so this
does not have to be discovered by eye.

At runtime the number keys `1`-`9` switch the viewer to the first nine clips of
every animated model, cross-fading over 0.25 s; the load log lists each animated
model's clips with their key, duration and an `once` marker for the non-looping ones.
Python scripts switch clips and override loop mode through `rend.set_animation()` and
`rend.set_animation_loop()` (see `docs/SCRIPTING.md`).

`ReflectionProbe position="x y z"` (optional, zero or more) marks where a probe-based
renderer captures its environment cubemap at load time. Like lights it describes the
scene, not the technique. Only the first probe is used for now (single probe, no
blending); with none listed the renderer defaults to the scene AABB's center. The
capture is static: animated meshes bake at the bind pose, lighting never re-captures,
and there is no parallax correction — probes are approximate by design, ray traced
reflections are the exact tier above.

`Fog position="x y z" size="x y z" density="0.03" color="r g b" anisotropy="0.6"
steps="24"` (optional, at most one) fills a world-space box of participating media:
`position` is the box center, `size` its full extents. Like lights it describes what
the media *is*, never the technique — the current renderer integrates it per shaded
fragment (a camera-to-surface raymarch sampling the sun's cascade shadow maps, so
occluders carve visible light shafts), but a froxel-based integrator would consume the
same element. `density` is the extinction coefficient per meter inside the box
(0.02-0.05 reads as heavy morning fog at street scale), `color` the scattering albedo,
`anisotropy` the Henyey-Greenstein g in -1..1 (positive = forward scattering, brighter
looking toward the sun), and `steps` the per-pixel raymarch budget. The raster
background clear color is matched to the fog's converged in-scatter so sky pixels read
as fog all the way out. The traced-primary path (`--rtprimary`) runs the same march
after its hit loop (fog scenes keep the cascade passes recorded for it, and its sky
miss uses the scene sky color). Current gap: in-scatter on transparent surfaces is
scaled by their blend factor (both paths).

`Script path="scripts/foo.py"` (optional, zero or more) names Python scripts that run
alongside the scene, resolved against the scene file's directory like mesh paths. The
viewer feeds them to the optional embedded Python host (`rend_pyhost.dll`), which runs
them in order on the interpreter's own thread; scripts `import rend` and speak the
renderer message queue (`load_model` / `set_transform` / `unload_model`, `poll_events`
/ `wait_model_ready`, `should_quit`, `log`, and the lighting controls `set_sun` /
`set_sky_color` / `set_ambient` / `set_point_light` — see `docs/SCRIPTING.md`). A scene that lists no scripts involves no
Python at all — the host DLL is never even loaded, so it and the CPython runtime may
be absent. assetio only parses the paths; it never executes code.

`Scene` takes an optional `loading` attribute describing how the app should
*present* asset loading, never how it schedules it (loading is nonblocking to the
renderer either way): `wait` (the default) holds the scene behind a loading screen
with a progress bar until every asset has landed; `streaming` shows the scene
immediately, assets popping in as they arrive (textures sample a neutral white
until theirs lands), with a small inline progress bar.

`Light` describes what the light *is*, never the rendering technique — shadow maps vs
ray tracing is the renderer's offer, per the intent/offer model in ARCHITECTURE.md.
`type="directional"` (the default) takes `direction` (from the light toward the
scene), `color`, `intensity`, `castsShadows` (default true); the first directional
light is the sun. `type="point"` takes `position`, `radius` (the falloff reach —
lighting rolls smoothly to zero there), `color`, `intensity`, and `castsShadows`
(default FALSE — each shadow-casting point light costs a load-time shadow capture in
the raster path and occlusion rays in the traced paths; today's renderer supports 16
point lights, extras are dropped with a warning). Every attribute is optional with
sensible defaults. Scripts can fade all point lights together
(`rend.set_point_light_scale`) or replace slots wholesale (`rend.set_point_light`).

