#include "cuda_utils/optix_program.hpp"

#include <fstream>

#include "optix_stubs.h"

#include "misc/check_macros.hpp"

OptixProgram::OptixProgram(const OptixProgramConfig &config, const OptixDeviceContext context) {
    char log_info[2048];
    size_t log_size = sizeof(log_info);

    // module
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = 50;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "optix_launch_params";

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = config.max_trace_depth;

    std::ifstream ptx_fin(config.ptx_path);
    const std::string ptx_source((std::istreambuf_iterator<char>(ptx_fin)), std::istreambuf_iterator<char>());
    OPTIX_CHECK(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_source.c_str(),
        ptx_source.size(),
        log_info,
        &log_size,
        &module_
    ));
    if (log_size > 1) {
        fmt::print("Create device programs module: {}\n", log_info);
    }
    log_size = sizeof(log_info);

    // raygen
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module = module_;
    pg_desc.raygen.entryFunctionName = config.raygen.c_str();

    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &pg_desc,
        1,
        &pg_options,
        log_info,
        &log_size,
        &raygen_programs_
    ));
    if (log_size > 1) {
        fmt::print("Create raygen programs: {}\n", log_info);
    }
    log_size = sizeof(log_info);

    // miss
    miss_programs_.resize(config.miss.size());
    for (size_t i = 0; i < config.miss.size(); i++) {
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = module_;
        pg_desc.miss.entryFunctionName = config.miss[i].c_str();

        OPTIX_CHECK(optixProgramGroupCreate(
            context,
            &pg_desc,
            1,
            &pg_options,
            log_info,
            &log_size,
            &miss_programs_[i]
        ));
        if (log_size > 1) {
            fmt::print("Create miss programs: {}\n", log_info);
        }
        log_size = sizeof(log_info);
    }

    // hit group
    hitgroup_programs_.resize(config.closesthit.size());
    for (size_t i = 0; i < config.closesthit.size(); i++) {
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = config.closesthit[i].c_str();
        pg_desc.hitgroup.moduleAH = module_;
        pg_desc.hitgroup.entryFunctionNameAH = config.anyhit[i].c_str();

        OPTIX_CHECK(optixProgramGroupCreate(
            context,
            &pg_desc,
            1,
            &pg_options,
            log_info,
            &log_size,
            &hitgroup_programs_[i]
        ));
        if (log_size > 1) {
            fmt::print("Create hitgroup programs: {}\n", log_info);
        }
        log_size = sizeof(log_info);
    }

    // pipeline
    std::vector<OptixProgramGroup> groups;
    groups.reserve(1 + miss_programs_.size() + hitgroup_programs_.size());
    groups.push_back(raygen_programs_);
    std::ranges::copy(miss_programs_, std::back_inserter(groups));
    std::ranges::copy(hitgroup_programs_, std::back_inserter(groups));

    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        groups.data(),
        static_cast<int>(groups.size()),
        log_info,
        &log_size,
        &pipeline_
    ));
    if (log_size > 1) {
        fmt::print("Create pipeline: {}\n", log_info);
    }

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        2 * 1024,
        2 * 1024,
        2 * 1024,
        1
    ));
}