#include "scene/scene.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>

#include "fmt/core.h"
#include "mikktspace.h"

namespace {

template <typename T>
bool GetAttribute(
    const tinygltf::Model &model,
    const tinygltf::Primitive &prim,
    const std::string &name,
    std::vector<T> &data
) {
    auto attrib_it = prim.attributes.find(name);
    if (attrib_it == prim.attributes.end()) {
        return false;
    }

    const auto &acc = model.accessors[attrib_it->second];
    const auto &buffer_view = model.bufferViews[acc.bufferView];
    const auto &buffer = model.buffers[buffer_view.buffer];
    const uint8_t *buffer_data = &buffer.data[acc.byteOffset + buffer_view.byteOffset];

    if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
        if (buffer_view.byteStride == 0) {
            const auto typed_buffer_data = reinterpret_cast<const T *>(buffer_data);
            std::copy(typed_buffer_data, typed_buffer_data + acc.count, std::back_inserter(data));
        } else {
            for (size_t i = 0; i < acc.count; i++) {
                data.push_back(*reinterpret_cast<const T *>(buffer_data + i * buffer_view.byteStride));
            }
        }
    } else {
        throw std::runtime_error("Unknown attribute format");
    }

    return true;
}

struct MikktData {
    const Mesh *mesh;
    const uint32_t *indices;
    const pcm::Vec3 *positions;
    const pcm::Vec3 *normals;
    const pcm::Vec2 *uvs;
    pcm::Vec4 *tangents;
};

int MikktGetNumFaces(const SMikkTSpaceContext *ctx) {
    const auto data = static_cast<MikktData *>(ctx->m_pUserData);
    return data->mesh->num_indices / 3;
}

int MikktGetNumVerticesOfFace(const SMikkTSpaceContext *ctx, const int face_id) {
    return 3;
}

void MikktGetPosition(const SMikkTSpaceContext *ctx, float *out_pos, const int face_id, const int vert_id) {
    const auto data = static_cast<MikktData *>(ctx->m_pUserData);
    uint32_t index = data->indices[data->mesh->first_index + face_id * 3 + vert_id];
    const pcm::Vec3 &pos = data->positions[data->mesh->vertex_offset + index];
    out_pos[0] = pos.X();
    out_pos[1] = pos.Y();
    out_pos[2] = pos.Z();
}

void MikktGetNormal(const SMikkTSpaceContext *ctx, float *out_norm, const int face_id, const int vert_id) {
    const auto data = static_cast<MikktData *>(ctx->m_pUserData);
    uint32_t index = data->indices[data->mesh->first_index + face_id * 3 + vert_id];
    const pcm::Vec3 &norm = data->normals[data->mesh->vertex_offset + index];
    out_norm[0] = norm.X();
    out_norm[1] = norm.Y();
    out_norm[2] = norm.Z();
}

void MikktGetTexcoord(const SMikkTSpaceContext *ctx, float *out_uv, const int face_id, const int vert_id) {
    const auto data = static_cast<MikktData *>(ctx->m_pUserData);
    uint32_t index = data->indices[data->mesh->first_index + face_id * 3 + vert_id];
    const pcm::Vec2 &uv = data->uvs[data->mesh->vertex_offset + index];
    out_uv[0] = uv.X();
    out_uv[1] = uv.Y();
}

void MikktSetTspaceBasic(
    const SMikkTSpaceContext *ctx,
    const float *tan,
    const float sign,
    const int face_id,
    const int vert_id
) {
    auto data = static_cast<MikktData *>(ctx->m_pUserData);
    uint32_t index = data->indices[data->mesh->first_index + face_id * 3 + vert_id];
    data->tangents[index] = pcm::Vec4(tan[0], tan[1], tan[2], sign);
}

}

Scene::Scene(const std::string &path) {
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    tinygltf::Model model;
    if (!loader.LoadASCIIFromFile(&model, &err, &warn, path)) {
        return;
    }
    if (!err.empty()) {
        fmt::print(stderr, "GLTF parse error: {}\n", err);
    }
    if (!warn.empty()) {
        fmt::print(stderr, "GLTF parse warning: {}\n", warn);
    }

    size_t num_vertices = 0;
    size_t num_indices = 0;
    size_t num_meshes = 0;
    std::vector<std::vector<size_t>> mesh_index_map;
    mesh_index_map.reserve(model.meshes.size());
    for (const auto &mesh : model.meshes) {
        std::vector<size_t> temp_map;
        temp_map.reserve(mesh.primitives.size());
        for (const auto &prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }
            const auto &pos_acc = model.accessors[prim.attributes.find("POSITION")->second];
            num_vertices += pos_acc.count;
            const auto &index_acc = model.accessors[prim.indices];
            num_indices += index_acc.count;
            temp_map.push_back(num_meshes++);
        }
        mesh_index_map.emplace_back(temp_map);
    }

    positions_.reserve(num_vertices);
    normals_.reserve(num_vertices);
    tangents_.reserve(num_vertices);
    uvs_.reserve(num_vertices);
    indices_.reserve(num_indices);

    std::vector<uint32_t> temp_index_buffer_u32;
    std::vector<uint16_t> temp_index_buffer_u16;
    std::vector<uint8_t> temp_index_buffer_u8;

    meshes_.reserve(num_meshes);
    for (const auto &mesh : model.meshes) {
        for (const auto &prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }
            Mesh mesh{};
            mesh.material_index = std::max(0, prim.material);
            mesh.first_index = indices_.size();
            mesh.vertex_offset = positions_.size();

            { // indices
                const auto &index_acc = model.accessors[prim.indices];
                const auto &buffer_view = model.bufferViews[index_acc.bufferView];
                const auto &buffer = model.buffers[buffer_view.buffer];
                mesh.num_indices = index_acc.count;
                switch (index_acc.componentType) {
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                    temp_index_buffer_u32.resize(index_acc.count);
                    memcpy(
                        temp_index_buffer_u32.data(),
                        &buffer.data[index_acc.byteOffset + buffer_view.byteOffset],
                        index_acc.count * sizeof(uint32_t)
                    );
                    std::ranges::copy(temp_index_buffer_u32, std::back_inserter(indices_));
                    break;
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                    temp_index_buffer_u16.resize(index_acc.count);
                    memcpy(
                        temp_index_buffer_u16.data(),
                        &buffer.data[index_acc.byteOffset + buffer_view.byteOffset],
                        index_acc.count * sizeof(uint16_t)
                    );
                    std::ranges::copy(temp_index_buffer_u16, std::back_inserter(indices_));
                    break;
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                    temp_index_buffer_u8.resize(index_acc.count);
                    memcpy(
                        temp_index_buffer_u8.data(),
                        &buffer.data[index_acc.byteOffset + buffer_view.byteOffset],
                        index_acc.count * sizeof(uint8_t)
                    );
                    std::ranges::copy(temp_index_buffer_u8, std::back_inserter(indices_));
                    break;
                default:
                    throw std::runtime_error("Unknown index format");
                }
            }

            { // positions
                GetAttribute(model, prim, "POSITION", positions_);
                const auto &pos_acc = model.accessors[prim.attributes.find("POSITION")->second];
                mesh.num_vertices = pos_acc.count;
                if (!pos_acc.minValues.empty()) {
                    mesh.bounding_min = pcm::Vec3(pos_acc.minValues[0], pos_acc.minValues[1], pos_acc.minValues[2]);
                } else {
                    mesh.bounding_min = std::reduce(
                        std::next(positions_.begin(), mesh.vertex_offset + 1),
                        positions_.end(),
                        positions_[mesh.vertex_offset],
                        [](const pcm::Vec3 &a, const pcm::Vec3 &b) { return a.Min(b); }
                    );
                }
                if (!pos_acc.maxValues.empty()) {
                    mesh.bounding_max = pcm::Vec3(pos_acc.maxValues[0], pos_acc.maxValues[1], pos_acc.maxValues[2]);
                } else {
                    mesh.bounding_max = std::reduce(
                        std::next(positions_.begin(), mesh.vertex_offset + 1),
                        positions_.end(),
                        positions_[mesh.vertex_offset],
                        [](const pcm::Vec3 &a, const pcm::Vec3 &b) { return a.Max(b); }
                    );
                }
            }

            { // normals
                if (!GetAttribute(model, prim, "NORMAL", normals_)) {
                    std::vector<pcm::Vec3> norm_sum(mesh.num_vertices);
                    std::ranges::fill(norm_sum, pcm::Vec3::Zero());
                    for (size_t i = 0; i < mesh.num_indices; i += 3) {
                        uint32_t i0 = indices_[mesh.first_index + i];
                        uint32_t i1 = indices_[mesh.first_index + i + 1];
                        uint32_t i2 = indices_[mesh.first_index + i + 2];
                        const pcm::Vec3 p0 = positions_[mesh.vertex_offset + i0];
                        const pcm::Vec3 p1 = positions_[mesh.vertex_offset + i1];
                        const pcm::Vec3 p2 = positions_[mesh.vertex_offset + i2];
                        const pcm::Vec3 norm = (p1 - p0).Cross(p2 - p0).Normalize();
                        norm_sum[i0] += norm;
                        norm_sum[i1] += norm;
                        norm_sum[i2] += norm;
                    }
                    for (pcm::Vec3 &norm : norm_sum) {
                        norm = norm.Normalize();
                    }
                    std::ranges::copy(norm_sum, std::back_inserter(normals_));
                }
            }

            { // uvs
                if (!GetAttribute(model, prim, "TEXCOORD_0", uvs_)) {
                    std::fill_n(std::back_inserter(uvs_), mesh.num_vertices, pcm::Vec2::Zero());
                }
            }

            { // tangents
                if (!GetAttribute(model, prim, "TANGENT", tangents_)) {
                    std::vector<pcm::Vec4> temp_tangents(mesh.num_vertices);
                    MikktData data {
                        .mesh = &mesh,
                        .indices = indices_.data(),
                        .positions = positions_.data(),
                        .normals = normals_.data(),
                        .uvs = uvs_.data(),
                        .tangents = temp_tangents.data(),
                    };
                    SMikkTSpaceInterface mikkt_int {
                        .m_getNumFaces = MikktGetNumFaces,
                        .m_getNumVerticesOfFace = MikktGetNumVerticesOfFace,
                        .m_getPosition = MikktGetPosition,
                        .m_getNormal = MikktGetNormal,
                        .m_getTexCoord = MikktGetTexcoord,
                        .m_setTSpaceBasic = MikktSetTspaceBasic,
                        .m_setTSpace = nullptr,
                    };
                    SMikkTSpaceContext mikkt_ctx {
                        .m_pInterface = &mikkt_int,
                        .m_pUserData = &data,
                    };
                    genTangSpaceDefault(&mikkt_ctx);
                    std::ranges::copy(temp_tangents, std::back_inserter(tangents_));
                }
            }

            meshes_.emplace_back(mesh);
        }
    }

    bounding_min_ = pcm::Vec3(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
    bounding_max_ = pcm::Vec3(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    );
    for (int node_ind : model.scenes[std::max(0, model.defaultScene)].nodes) {
        const auto &node = model.nodes[node_ind];
        ProcessNode(model, node, pcm::Mat4::Identity(), mesh_index_map);
    }

    materials_.reserve(std::max<size_t>(1, model.materials.size()));
    for (const auto &in_mat : model.materials) {
        Material mat;
        mat.alpha_cutoff = in_mat.alphaCutoff;
        mat.alpha_mode = in_mat.alphaMode == "MASK" ? 1 : (in_mat.alphaMode == "BLEND" ? 2 : 0);
        mat.double_sided = in_mat.doubleSided;
        mat.emissive = pcm::Vec3(in_mat.emissiveFactor[0], in_mat.emissiveFactor[1], in_mat.emissiveFactor[2]);
        mat.emissive_tex = in_mat.emissiveTexture.index;
        mat.normal_tex = in_mat.normalTexture.index;
        mat.normal_tex_scale = in_mat.normalTexture.scale;
        mat.occlusion_tex = in_mat.occlusionTexture.index;
        mat.occlusion_tex_strength = in_mat.occlusionTexture.strength;
        const auto &in_pbr = in_mat.pbrMetallicRoughness;
        mat.base_color = pcm::Vec4(
            in_pbr.baseColorFactor[0],
            in_pbr.baseColorFactor[1],
            in_pbr.baseColorFactor[2],
            in_pbr.baseColorFactor[3]
        );
        mat.base_color_tex = in_pbr.baseColorTexture.index;
        mat.metallic = in_pbr.metallicFactor;
        mat.roughness = in_pbr.roughnessFactor;
        mat.metallic_roughness_tex = in_pbr.metallicRoughnessTexture.index;
        materials_.emplace_back(mat);
    }
    if (materials_.empty()) {
        materials_.emplace_back();
    }

    textures_ = std::move(model.textures);
    samplers_ = std::move(model.samplers);
    images_ = std::move(model.images);
    if (images_.empty()) {
        tinygltf::Image default_image = {};
        default_image.name = std::string("default");
        default_image.width = 1;
        default_image.height = 1;
        default_image.component = 4;
        default_image.bits = 8;
        default_image.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
        default_image.image = {255, 255, 255, 255};
        images_.emplace_back(default_image);
    }
}

void Scene::ProcessNode(
    const tinygltf::Model &model,
    const tinygltf::Node &node,
    const pcm::Mat4 &parent_model,
    const std::vector<std::vector<size_t>> &mesh_index_map
) {
    pcm::Mat4 curr_mat = pcm::Mat4::Identity();
    if (!node.translation.empty()) {
        curr_mat = curr_mat * pcm::Translate(node.translation[0], node.translation[1], node.translation[2]);
    }
    if (!node.scale.empty()) {
        curr_mat = curr_mat * pcm::Scale(node.scale[0], node.scale[1], node.scale[2]);
    }
    if (!node.rotation.empty()) {
        curr_mat = curr_mat
            * pcm::Rotate(pcm::Vec4(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]));
    }
    if (!node.matrix.empty()) {
        size_t col = 0;
        size_t row = 0;
        pcm::Mat4 temp_mat;
        for (size_t i = 0; i < 16; i++) {
            temp_mat[col][row] = node.matrix[i];
            if (row == 3) {
                ++col;
                row = 0;
            } else {
                ++row;
            }
        }
        curr_mat = curr_mat * temp_mat;
    }
    const pcm::Mat4 model_mat = parent_model * curr_mat;

    if (node.mesh >= 0) {
        const auto &mesh = model.meshes[node.mesh];
        size_t prim_ind = 0;
        for (const auto &prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }
            DrawableNode drawable {
                .model = model_mat,
                .mesh_index = mesh_index_map[node.mesh][prim_ind],
            };
            bounding_min_ =
                bounding_min_.Min(pcm::Vec3(model_mat * pcm::Vec4(meshes_[drawable.mesh_index].bounding_min, 1.0f)));
            bounding_max_ =
                bounding_max_.Max(pcm::Vec3(model_mat * pcm::Vec4(meshes_[drawable.mesh_index].bounding_max, 1.0f)));
            drawables_.emplace_back(drawable);
            ++prim_ind;
        }
    }

    for (int child : node.children) {
        const auto &child_node = model.nodes[child];
        ProcessNode(model, child_node, model_mat, mesh_index_map);
    }
}