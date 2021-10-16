#pragma once

#include <vector>

#include "misc/light_data.hpp"
#include "scene/scene.hpp"

class Lights {
public:
    Lights(const Scene *scene);

    const std::vector<LightData> &LightsData() const {
        return lights_;
    }

private:
    std::vector<LightData> lights_;
};