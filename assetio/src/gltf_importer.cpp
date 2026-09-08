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
    } else if (m.has_pbr_specular_glossiness) {
        // KHR_materials_pbrSpecularGlossiness mapped onto metallic-
        // roughness: diffuse becomes base color, roughness = 1 - gloss,
        // dielectric. The spec-gloss texture's channels don't line up
        // with an MR texture (specular RGB + gloss A), so only the
        // factors convert — per-material constant roughness.
        const auto& sg = m.pbr_specular_glossiness;
        out.baseColorFactor = {sg.diffuse_factor[0], sg.diffuse_factor[1], sg.diffuse_factor[2],
                               sg.diffuse_factor[3]};
        out.metallicFactor = 0.0f;
        out.roughnessFactor = 1.0f - sg.glossiness_factor;
        out.baseColorTexture = resolveTexture(sg.diffuse_texture, baseDir);
    }
    out.normalTexture = resolveTexture(m.normal_texture, baseDir);
    out.alphaMasked = m.alpha_mode == cgltf_alpha_mode_mask;
    out.alphaCutoff = m.alpha_cutoff;
    out.transparent = m.alpha_mode == cgltf_alpha_mode_blend;
    // KHR_materials_transmission (glass) approximated as alpha blending
    // until real refraction lands: opacity = 1 - transmission.
    if (m.has_transmission) {
        out.transparent = true;
        out.baseColorFactor[3] = std::min(
            out.baseColorFactor[3],
            std::max(1.0f - m.transmission.transmission_factor, 0.05f));
    }
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
    // The cofactor matrix is det(M) * M^-T; normalization removes |det| but
    // not its sign, so mirrored transforms (det < 0) still need a flip.
    const float det = a00 * c00 + a01 * c01 + a02 * c02;
    const float flip = det < 0.0f ? -1.0f : 1.0f;
    for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
        const float x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
        float nx = (c00 * x + c01 * y + c02 * z) * flip;
        float ny = (c10 * x + c11 * y + c12 * z) * flip;
        float nz = (c20 * x + c21 * y + c22 * z) * flip;
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

// --- Animated-model support: node hierarchy, skins, keyframe channels ---

// Rotation part of a (assumed TRS) matrix as a quaternion, for the rare
// node that stores a matrix instead of decomposed TRS.
std::array<float, 4> matrixToQuaternion(const float r[9]) {
    std::array<float, 4> q{0.0f, 0.0f, 0.0f, 1.0f};
    const float trace = r[0] + r[4] + r[8];
    if (trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q = {(r[5] - r[7]) / s, (r[6] - r[2]) / s, (r[1] - r[3]) / s, 0.25f * s};
    } else if (r[0] > r[4] && r[0] > r[8]) {
        const float s = std::sqrt(1.0f + r[0] - r[4] - r[8]) * 2.0f;
        q = {0.25f * s, (r[3] + r[1]) / s, (r[6] + r[2]) / s, (r[5] - r[7]) / s};
    } else if (r[4] > r[8]) {
        const float s = std::sqrt(1.0f + r[4] - r[0] - r[8]) * 2.0f;
        q = {(r[3] + r[1]) / s, 0.25f * s, (r[7] + r[5]) / s, (r[6] - r[2]) / s};
    } else {
        const float s = std::sqrt(1.0f + r[8] - r[0] - r[4]) * 2.0f;
        q = {(r[6] + r[2]) / s, (r[7] + r[5]) / s, 0.25f * s, (r[1] - r[3]) / s};
    }
    return q;
}

SkeletonNode convertNodePose(const cgltf_node& node) {
    SkeletonNode out;
    out.name = node.name ? node.name : "";
    if (node.has_matrix) {
        // Decompose: translation straight out, scale = column lengths,
        // rotation from the normalized upper 3x3.
        const float* m = node.matrix;
        out.translation = {m[12], m[13], m[14]};
        float rot[9];
        for (int c = 0; c < 3; ++c) {
            const float len = std::sqrt(m[c * 4] * m[c * 4] + m[c * 4 + 1] * m[c * 4 + 1] +
                                        m[c * 4 + 2] * m[c * 4 + 2]);
            out.scale[c] = len;
            const float inv = len > 0.0f ? 1.0f / len : 0.0f;
            rot[c * 3] = m[c * 4] * inv;
            rot[c * 3 + 1] = m[c * 4 + 1] * inv;
            rot[c * 3 + 2] = m[c * 4 + 2] * inv;
        }
        out.rotation = matrixToQuaternion(rot);
        return out;
    }
    if (node.has_translation) {
        out.translation = {node.translation[0], node.translation[1], node.translation[2]};
    }
    if (node.has_rotation) {
        out.rotation = {node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]};
    }
    if (node.has_scale) {
        out.scale = {node.scale[0], node.scale[1], node.scale[2]};
    }
    return out;
}

// Whole node hierarchy, reordered parents-first; remap[cgltf index] = new
// index. One flat skeleton per model keeps channel targets trivial.
void buildSkeleton(const cgltf_data& data, SkeletonData& skeleton,
                   std::vector<std::uint32_t>& remap) {
    remap.assign(data.nodes_count, 0);
    skeleton.nodes.reserve(data.nodes_count);
    // Iterative DFS from every root so parents land before children.
    std::vector<const cgltf_node*> stack;
    for (cgltf_size n = data.nodes_count; n > 0; --n) {
        if (!data.nodes[n - 1].parent) {
            stack.push_back(&data.nodes[n - 1]);
        }
    }
    while (!stack.empty()) {
        const cgltf_node* node = stack.back();
        stack.pop_back();
        const auto cgltfIndex = static_cast<std::size_t>(node - data.nodes);
        remap[cgltfIndex] = static_cast<std::uint32_t>(skeleton.nodes.size());
        SkeletonNode pose = convertNodePose(*node);
        pose.parent = node->parent
                          ? static_cast<std::int32_t>(
                                remap[static_cast<std::size_t>(node->parent - data.nodes)])
                          : -1;
        skeleton.nodes.push_back(std::move(pose));
        for (cgltf_size c = node->children_count; c > 0; --c) {
            stack.push_back(node->children[c - 1]);
        }
    }
}

Result<void> convertSkin(const cgltf_data& data, const cgltf_skin& skin,
                         const std::vector<std::uint32_t>& remap, SkeletonData& skeleton) {
    skeleton.jointNodes.reserve(skin.joints_count);
    for (cgltf_size j = 0; j < skin.joints_count; ++j) {
        skeleton.jointNodes.push_back(
            remap[static_cast<std::size_t>(skin.joints[j] - data.nodes)]);
    }
    skeleton.inverseBind.resize(skin.joints_count);
    if (skin.inverse_bind_matrices) {
        std::vector<float> all;
        if (!unpackFloats(skin.inverse_bind_matrices, 16, all)) {
            return Error{"Failed to unpack inverse bind matrices"};
        }
        for (cgltf_size j = 0; j < skin.joints_count; ++j) {
            std::copy_n(all.begin() + j * 16, 16, skeleton.inverseBind[j].begin());
        }
    } else {
        for (auto& m : skeleton.inverseBind) {
            m = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        }
    }
    return {};
}

Result<void> convertAnimations(const cgltf_data& data, const std::vector<std::uint32_t>& remap,
                               ModelData& model) {
    for (cgltf_size a = 0; a < data.animations_count; ++a) {
        const cgltf_animation& src = data.animations[a];
        AnimationData anim;
        anim.name = src.name ? src.name : std::format("animation{}", a);
        for (cgltf_size c = 0; c < src.channels_count; ++c) {
            const cgltf_animation_channel& channel = src.channels[c];
            if (!channel.target_node || !channel.sampler) {
                continue;
            }
            AnimationChannelData out;
            out.node = remap[static_cast<std::size_t>(channel.target_node - data.nodes)];
            std::size_t components = 0;
            switch (channel.target_path) {
            case cgltf_animation_path_type_translation:
                out.path = AnimationPath::Translation;
                components = 3;
                break;
            case cgltf_animation_path_type_rotation:
                out.path = AnimationPath::Rotation;
                components = 4;
                break;
            case cgltf_animation_path_type_scale:
                out.path = AnimationPath::Scale;
                components = 3;
                break;
            case cgltf_animation_path_type_weights:
                out.path = AnimationPath::Weights;
                components = channel.target_node->mesh
                                 ? channel.target_node->mesh->primitives[0].targets_count
                                 : 0;
                break;
            default:
                continue;
            }
            if (components == 0) {
                continue;
            }
            switch (channel.sampler->interpolation) {
            case cgltf_interpolation_type_step:
                out.interpolation = AnimationInterpolation::Step;
                break;
            case cgltf_interpolation_type_cubic_spline:
                out.interpolation = AnimationInterpolation::CubicSpline;
                components *= 3; // in-tangent, value, out-tangent per key
                break;
            default:
                out.interpolation = AnimationInterpolation::Linear;
                break;
            }
            // Output accessors are vec3/vec4 per key for T/R/S but SCALAR
            // with count = keys*targets for weights — size from the
            // accessor itself, then validate against the expected shape.
            out.values.resize(channel.sampler->output->count *
                              cgltf_num_components(channel.sampler->output->type));
            if (!unpackFloats(channel.sampler->input, 1, out.times) ||
                cgltf_accessor_unpack_floats(channel.sampler->output, out.values.data(),
                                             out.values.size()) != out.values.size()) {
                return Error{std::format("Failed to unpack animation '{}'", anim.name)};
            }
            if (out.values.size() != out.times.size() * components) {
                return Error{std::format("Animation '{}': {} values for {} keys x {} components",
                                         anim.name, out.values.size(), out.times.size(),
                                         components)};
            }
            if (!out.times.empty()) {
                anim.duration = std::max(anim.duration, out.times.back());
            }
            anim.channels.push_back(std::move(out));
        }
        model.animations.push_back(std::move(anim));
    }
    return {};
}

// remap non-null = animated model: geometry stays in mesh space and the
// mesh records its node plus skinning/morph payloads instead of baking.
Result<void> convertPrimitive(const cgltf_data& data, const cgltf_node& node,
                              const cgltf_primitive& prim, std::size_t primIndex,
                              const float world[16], const std::vector<std::uint32_t>* remap,
                              ModelData& model) {
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
        case cgltf_attribute_type_joints:
            if (remap) {
                mesh.joints.resize(attr.data->count * 4);
                for (cgltf_size v = 0; v < attr.data->count; ++v) {
                    cgltf_uint j[4] = {};
                    ok = ok && cgltf_accessor_read_uint(attr.data, v, j, 4);
                    for (int k = 0; k < 4; ++k) {
                        mesh.joints[v * 4 + k] = static_cast<std::uint16_t>(j[k]);
                    }
                }
            }
            break;
        case cgltf_attribute_type_weights:
            if (remap) {
                ok = unpackFloats(attr.data, 4, mesh.weights);
            }
            break;
        default: break;
        }
        if (!ok) {
            return Error{std::format("Failed to unpack attribute data in '{}'", mesh.name)};
        }
    }
    if (mesh.positions.empty()) {
        return Error{std::format("Primitive '{}' has no positions", mesh.name)};
    }

    if (remap) {
        mesh.sourceNode = (*remap)[static_cast<std::size_t>(&node - data.nodes)];
        mesh.skinned = node.skin != nullptr && !mesh.joints.empty() && !mesh.weights.empty();
        for (cgltf_size t = 0; t < prim.targets_count; ++t) {
            MorphTargetData target;
            for (cgltf_size a = 0; a < prim.targets[t].attributes_count; ++a) {
                const cgltf_attribute& attr = prim.targets[t].attributes[a];
                bool ok = true;
                if (attr.type == cgltf_attribute_type_position) {
                    ok = unpackFloats(attr.data, 3, target.positionDeltas);
                } else if (attr.type == cgltf_attribute_type_normal) {
                    ok = unpackFloats(attr.data, 3, target.normalDeltas);
                }
                if (!ok) {
                    return Error{std::format("Failed to unpack morph target in '{}'", mesh.name)};
                }
            }
            mesh.morphTargets.push_back(std::move(target));
        }
        const cgltf_float* restWeights =
            node.weights_count > 0 ? node.weights : node.mesh->weights;
        const cgltf_size restCount =
            node.weights_count > 0 ? node.weights_count : node.mesh->weights_count;
        for (cgltf_size w = 0; w < std::min<cgltf_size>(restCount, prim.targets_count); ++w) {
            mesh.morphWeights.push_back(restWeights[w]);
        }
        mesh.morphWeights.resize(prim.targets_count, 0.0f);
    } else {
        transformPoints(world, mesh.positions);
        transformNormals(world, mesh.normals);
    }

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

Result<void> convertNode(const cgltf_data& data, const cgltf_node& node,
                         const std::vector<std::uint32_t>* remap, ModelData& model) {
    if (node.mesh) {
        float world[16];
        cgltf_node_transform_world(&node, world);
        for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
            if (auto r = convertPrimitive(data, node, node.mesh->primitives[p], p, world, remap,
                                          model);
                !r) {
                return r;
            }
        }
    }
    for (cgltf_size c = 0; c < node.children_count; ++c) {
        if (auto r = convertNode(data, *node.children[c], remap, model); !r) {
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

    // Animated models keep their node hierarchy live (nothing baked); the
    // skeleton is the whole node set so channels target nodes directly.
    const bool animated = data->animations_count > 0 || data->skins_count > 0;
    std::vector<std::uint32_t> remap;
    if (animated) {
        buildSkeleton(*data, model.skeleton, remap);
        if (data->skins_count > 0) {
            if (data->skins_count > 1) {
                log::warn("'{}': {} skins, only the first is used", model.name,
                          data->skins_count);
            }
            if (auto r = convertSkin(*data, data->skins[0], remap, model.skeleton); !r) {
                cgltf_free(data);
                return r.error();
            }
        }
        if (auto r = convertAnimations(*data, remap, model); !r) {
            cgltf_free(data);
            return r.error();
        }
    }

    for (cgltf_size n = 0; n < scene->nodes_count; ++n) {
        if (auto r = convertNode(*data, *scene->nodes[n], animated ? &remap : nullptr, model);
            !r) {
            cgltf_free(data);
            return r.error();
        }
    }
    cgltf_free(data);

    if (animated) {
        std::size_t morphTargets = 0;
        for (const MeshData& mesh : model.meshes) {
            morphTargets += mesh.morphTargets.size();
        }
        log::info("'{}' animated: {} nodes, {} joints, {} animations, {} morph targets",
                  model.name, model.skeleton.nodes.size(), model.skeleton.jointNodes.size(),
                  model.animations.size(), morphTargets);
    }

    // Slot for primitives that reference no material (see convertPrimitive).
    const bool needsDefault = std::ranges::any_of(
        model.meshes, [&](const MeshData& m) { return m.materialIndex >= model.materials.size(); });
    if (needsDefault) {
        model.materials.push_back({.name = "default"});
    }

    return model;
}

} // namespace rend::assetio
