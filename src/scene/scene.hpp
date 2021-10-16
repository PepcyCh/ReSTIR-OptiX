#pragma once

#include "tiny_gltf.h"

#include "scene/structs.hpp"

class Scene {
public:
    Scene(const std::string &path);

    const std::vector<Mesh> &Meshes() const {
        return meshes_;
    }

    const std::vector<Material> &Materials() const {
        return materials_;
    }

    const std::vector<DrawableNode> &Drawables() const {
        return drawables_;
    }

    const std::vector<tinygltf::Texture> &Textures() const {
        return textures_;
    }

    const std::vector<tinygltf::Sampler> &Samplers() const {
        return samplers_;
    }
    
    const std::vector<tinygltf::Image> &Images() const {
        return images_;
    }

    const std::vector<pcm::Vec3> &Positions() const {
        return positions_;
    }

    const std::vector<pcm::Vec3> &Normals() const {
        return normals_;
    }

    const std::vector<pcm::Vec4> &Tangents() const {
        return tangents_;
    }

    const std::vector<pcm::Vec2> &Uvs() const {
        return uvs_;
    }

    const std::vector<uint32_t> &Indices() const {
        return indices_;
    }

    const pcm::Vec3 &BoundingMin() const {
        return bounding_min_;
    }

    const pcm::Vec3 &BoundingMax() const {
        return bounding_max_;
    }

    pcm::Vec3 Center() const {
        return (bounding_min_ + bounding_max_) * 0.5f;
    }

private:
    void ProcessNode(
        const tinygltf::Model &model,
        const tinygltf::Node &node,
        const pcm::Mat4 &parent_model,
        const std::vector<std::vector<size_t>> &mesh_index_map
    );

    std::vector<Mesh> meshes_;
    std::vector<Material> materials_;
    std::vector<DrawableNode> drawables_;

    std::vector<tinygltf::Texture> textures_;
    std::vector<tinygltf::Sampler> samplers_;
    std::vector<tinygltf::Image> images_;

    std::vector<pcm::Vec3> positions_;
    std::vector<pcm::Vec3> normals_;
    std::vector<pcm::Vec4> tangents_;
    std::vector<pcm::Vec2> uvs_;
    std::vector<uint32_t> indices_;

    pcm::Vec3 bounding_min_;
    pcm::Vec3 bounding_max_;
};