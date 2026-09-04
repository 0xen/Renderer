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
        <Stage entrypoint="main" stage="Vertex"   path="data/Shaders/Terrain/vert.spv" />
        <Stage entrypoint="main" stage="Fragment" path="data/Shaders/Terrain/frag.spv" />
    </Stages>
</Pipeline>
```

- Formats/rates use the Vulkan vocabulary (`R32G32_SFLOAT`, `INPUT_RATE_VERTEX`,
  `INPUT_RATE_INSTANCE`); topology values: `Triangle`, … (extended as needed).
- `path` points at offline-compiled SPIR-V; a build step (glslc from the SDK) will
  compile shader source into these artifacts.
- Planned: SPIR-V reflection (SPIRV-Reflect) at load validates that vertex layout and
  descriptor references match the shader — mismatches are load-time errors, not
  silent corruption.

## Scene file (planned shape)

The scene hierarchy contains nodes such as models; a model references a pipeline
definition file plus its buffers/textures. Exact schema to be designed at roadmap #8.
```xml
<Scene>
    <Model name="terrain">
        <Shader path="pipelines/TerrainBackground.xml" />
        <!-- meshes, textures, transforms ... -->
    </Model>
</Scene>
```
