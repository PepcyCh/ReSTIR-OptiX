#include "misc/camera.hpp"

#include <algorithm>
#include <numbers>

Camera::Camera(
    float near,
    float fov,
    float aspect,
    float phi,
    float theta,
    float radius,
    const pcm::Vec3 &center
) : near_(near), fov_(fov), aspect_(aspect), phi_(phi), theta_(theta), radius_(radius), center_(center) {
    proj_ = pcm::PerspectiveInfReverseZ(fov, aspect, near, true);

    pos_ = pcm::Vec3(
        radius * std::sin(phi) * std::cos(theta),
        radius * std::cos(phi),
        radius * std::sin(phi) * std::sin(theta)
    ) + center;
    view_ = pcm::LookAt(pos_, center, pcm::Vec3::UnitY());
}

void Camera::Rotate(float delta_phi, float delta_theta) {
    phi_ = std::clamp(phi_ + delta_phi, 0.1f, std::numbers::pi_v<float> - 0.1f);
    theta_ += delta_theta;

    pos_ = pcm::Vec3(
        radius_ * std::sin(phi_) * std::cos(theta_),
        radius_ * std::cos(phi_),
        radius_ * std::sin(phi_) * std::sin(theta_)
    ) + center_;
    view_ = pcm::LookAt(pos_, center_, pcm::Vec3::UnitY());
}

void Camera::Move(const pcm::Vec3 &delta_center) {
    center_ += delta_center;
    pos_ += delta_center;
    
    view_ = pcm::LookAt(pos_, center_, pcm::Vec3::UnitY());
}

void Camera::Stretch(float delta_radius) {
    radius_ = std::max(0.1f, radius_ + delta_radius);

    pos_ = pcm::Vec3(
        radius_ * std::sin(phi_) * std::cos(theta_),
        radius_ * std::cos(phi_),
        radius_ * std::sin(phi_) * std::sin(theta_)
    ) + center_;
    view_ = pcm::LookAt(pos_, center_, pcm::Vec3::UnitY());
}

void Camera::UpdateAspect(float aspect) {
    aspect_ = aspect;

    proj_ = pcm::PerspectiveInfReverseZ(fov_, aspect_, near_, true);
}