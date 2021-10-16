#pragma once

#include <string>
#include <vector>

#include "optix.h"

struct OptixProgramConfig {
    std::string ptx_path;
    std::string launch_params;
    uint32_t max_trace_depth;
    std::string raygen;
    std::vector<std::string> miss;
    std::vector<std::string> closesthit;
    std::vector<std::string> anyhit;
};

class OptixProgram {
public:
    OptixProgram(const OptixProgramConfig &config, const OptixDeviceContext context);

    const OptixPipeline Pipeline() const {
        return pipeline_;
    }

    const OptixProgramGroup RaygenProgram() const {
        return raygen_programs_;
    }

    const std::vector<OptixProgramGroup> &MissPrograms() const {
        return miss_programs_;
    }

    const std::vector<OptixProgramGroup> &HitgroupPrograms() const {
        return hitgroup_programs_;
    }

private:
    OptixModule module_;
    OptixProgramGroup raygen_programs_;
    std::vector<OptixProgramGroup> miss_programs_;
    std::vector<OptixProgramGroup> hitgroup_programs_;
    OptixPipeline pipeline_;
};