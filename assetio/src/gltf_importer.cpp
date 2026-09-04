#include "gltf_importer.h"

#include "rend/core/log.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <numeric>
#include <string>

namespace rend::assetio {

namespace {

const char* resultName(cgltf_result r) {
    switch (r) {
    case cgltf_result_success: return "success";
    case cgltf_result_data_too_short: return "data too short";
    case cgltf_result_unknown_format: return "unknown format";
    case cgltf_result_invalid_json: return "invalid JSON";
    case cgltf_result_invalid_gltf: return "invalid glTF";
    case cgltf_result_out_of_memory: return "out of memory";
    case cgltf_result_file_not_found: return "file not found";
    case cgltf_result_io_error: return "IO error";
    default: return "error";
    }
}

// glTF image URIs are percent-encoded paths relative to the .gltf file.
std::filesystem::path resolveTexture(const cgltf_texture_view& view,
                                     const std::filesystem::path& baseDir) {
    if (!view.texture || !view.texture->image || !view.texture->image->uri) {
        return {};
    }
    std::string uri = view.texture->image->uri;
    cgltf_decode_uri(uri.data());
    uri.resize(std::strlen(uri.c_str()));
    return baseDir / std::filesystem::path(uri);
}

MaterialData convertMaterial(const cgltf_material& m, const std::filesystem::path& baseDir) {
    MaterialData out;
    out.name = m.name ? m.name : "";
    if (m.has_pbr_metallic_roughness) {
        const auto& pbr = m.pbr_metallic_roughness;
        out.baseColorFactor = {pbr.base_color_factor[0], pbr.base_color_factor[1],
                               pbr.base_color_factor[2], pbr.base_color_factor[3]};
        out.metallicFactor = pbr.metallic_factor;
        out.roughnessFactor = pbr.roughness_factor;
        out.baseColorTexture = resolveTexture(pbr.base_color_texture, baseDir);
        out.metallicRoughnessTexture = resolveTexture(pbr.metallic_roughness_texture, baseDir);
    }
    out.normalTexture = resolveTexture(m.normal_texture, baseDir);
    out.alphaMasked = m.alpha_mode == cgltf_alpha_mode_mask;
    out.alphaCutoff = m.alpha_cutoff;
    return out;
}

// Column-major 4x4 point transform.
void transformPoints(const float m[16], std::vector<float>& xyz) {
    for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
        const float x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
        xyz[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
        xyz[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
        xyz[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
}

// Normals need the inverse-transpose of the upper 3x3 (= cofactor matrix /
// determinant; the determinant divides out under normalization).
void transformNormals(const float m[16], std::vector<float>& xyz) {
    const float a00 = m[0], a01 = m[4], a02 = m[8];
    const float a10 = m[1], a11 = m[5], a12 = m[9];
    const float a20 = m[2], a21 = m[6], a22 = m[10];
    const float c00 = a11 * a22 - a12 * a21, c01 = a12 * a20 - a10 * a22, c02 = a10 * a21 - a11 * a20;
    const float c10 = a02 * a21 - a01 * a22, c11 = a00 * a22 - a02 * a20, c12 = a01 * a20 - a00 * a21;
    const float c20 = a01 * a12 - a02 * a11, c21 = a02 * a10 - a00 * a12, c22 = a00 * a11 - a01 * a10;
    for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
        const float x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
        float nx = c00 * x + c10 * y + c20 * z;
        float ny = c01 * x + c11 * y + c21 * z;
        float nz = c02 * x + c12 * y + c22 * z;
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 0.0f) {
            nx /= len;
            ny /= len;
            nz /= len;
        }
        xyz[i] = nx;
        xyz[i + 1] = ny;
        xyz[i + 2] = nz;
    }
}

bool unpackFloats(const cgltf_accessor* accessor, std::size_t components, std::vector<float>& out) {
    out.resize(accessor->count * components);
    return cgltf_accessor_unpack_floats(accessor, out.data(), out.size()) == out.size();
}

Result<void> convertPrimitive(const cgltf_data& data, const cgltf_node& node,
                              const cgltf_primitive& prim, std::size_t primIndex,
                              const float world[16], ModelData& model) {
    if (prim.type != cgltf_primitive_type_triangles) {
        log::warn("Skipping non-triangle primitive in '{}'", model.name);
        return {};
    }

    MeshData mesh;
    const char* nodeName = node.name ? node.name : (node.mesh->name ? node.mesh->name : "mesh");
    mesh.name = node.mesh->primitives_count > 1 ? std::format("{}:{}", nodeName, primIndex)
                                                : nodeName;

    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        if (attr.index != 0) {
            continue; // only the first UV/color/... set for now
        }
        bool ok = true;
        switch (attr.type) {
        case cgltf_attribute_type_position: ok = unpackFloats(attr.data, 3, mesh.positions); break;
        case cgltf_attribute_type_normal: ok = unpackFloats(attr.data, 3, mesh.normals); break;
        case cgltf_attribute_type_texcoord: ok = unpackFloats(attr.data, 2, mesh.uvs); break;
        default: break;
        }
        if (!ok) {
            return Error{std::format("Failed to unpack attribute data in '{}'", mesh.name)};
        }
    }
    if (mesh.positions.empty()) {
        return Error{std::format("Primitive '{}' has no positions", mesh.name)};
    }

    transformPoints(world, mesh.positions);
    transformNormals(world, mesh.normals);

    if (prim.indices) {
        mesh.indices.resize(prim.indices->count);
        for (cgltf_size i = 0; i < prim.indices->count; ++i) {
            mesh.indices[i] = static_cast<std::uint32_t>(cgltf_accessor_read_index(prim.indices, i));
        }
    } else {
        mesh.indices.resize(mesh.vertexCount());
        std::iota(mesh.indices.begin(), mesh.indices.end(), 0u);
    }

    // Material index maps 1:1 onto data.materials; a missing material gets
    // the defaults slot appended after all real ones.
    mesh.materialIndex = prim.material
                             ? static_cast<std::uint32_t>(prim.material - data.materials)
                             : static_cast<std::uint32_t>(data.materials_count);

    model.meshes.push_back(std::move(mesh));
    return {};
}

Result<void> convertNode(const cgltf_data& data, const cgltf_node& node, ModelData& model) {
    if (node.mesh) {
        float world[16];
        cgltf_node_transform_world(&node, world);
        for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
            if (auto r = convertPrimitive(data, node, node.mesh->primitives[p], p, world, model);
                !r) {
                return r;
            }
        }
    }
    for (cgltf_size c = 0; c < node.children_count; ++c) {
        if (auto r = convertNode(data, *node.children[c], model); !r) {
            return r;
        }
    }
    return {};
}

} // namespace

bool GltfImporter::supports(const std::filesystem::path& file) const {
    const auto ext = file.extension();
    return ext == ".gltf" || ext == ".glb";
}

Result<ModelData> GltfImporter::import(const std::filesystem::path& file) const {
    const std::string pathUtf8 = file.string();

    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result parsed = cgltf_parse_file(&options, pathUtf8.c_str(), &data);
    if (parsed != cgltf_result_success) {
        return Error{std::format("glTF parse failed for '{}': {}", pathUtf8, resultName(parsed))};
    }

    // External buffers (.bin) and GLB-embedded chunks resolve here.
    cgltf_result buffers = cgltf_load_buffers(&options, data, pathUtf8.c_str());
    if (buffers != cgltf_result_success) {
        cgltf_free(data);
        return Error{
            std::format("glTF buffer load failed for '{}': {}", pathUtf8, resultName(buffers))};
    }

    ModelData model;
    model.name = file.stem().string();
    const std::filesystem::path baseDir = file.parent_path();

    for (cgltf_size i = 0; i < data->materials_count; ++i) {
        model.materials.push_back(convertMaterial(data->materials[i], baseDir));
    }

    const cgltf_scene* scene = data->scene ? data->scene : data->scenes;
    if (!scene) {
        cgltf_free(data);
        return Error{std::format("glTF '{}' contains no scene", pathUtf8)};
    }
    for (cgltf_size n = 0; n < scene->nodes_count; ++n) {
        if (auto r = convertNode(*data, *scene->nodes[n], model); !r) {
            cgltf_free(data);
            return r.error();
        }
    }
    cgltf_free(data);

    // Slot for primitives that reference no material (see convertPrimitive).
    const bool needsDefault = std::ranges::any_of(
        model.meshes, [&](const MeshData& m) { return m.materialIndex >= model.materials.size(); });
    if (needsDefault) {
        model.materials.push_back({.name = "default"});
    }

    return model;
}

} // namespace rend::assetio
