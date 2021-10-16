#pragma once

#include "pcmath/pcmath.hpp"

class Camera {
public:
    Camera(
        float near,
        float fov,
        float aspect,
        float phi,
        float theta,
        float radius,
        const pcm::Vec3 &center
    );

    void Rotate(float delta_phi, float delta_theta);

    void Stretch(float delta_radius);

    void Move(const pcm::Vec3 &delta_center);

    void UpdateAspect(float aspect);

    const pcm::Mat4 &ProjMatrix() const {
        return proj_;
    }

    const pcm::Mat4 &ViewMatrix() const {
        return view_;
    }

    pcm::Mat4 ProjViewMatrix() const {
        return proj_ * view_;
    }

    const pcm::Vec3 &Position() const {
        return pos_;
    }

private:
    float near_;
    float fov_;
    float aspect_;

    float phi_;
    float theta_;
    float radius_;
    pcm::Vec3 center_;

    pcm::Mat4 proj_;
    pcm::Mat4 view_;
    pcm::Vec3 pos_;
};