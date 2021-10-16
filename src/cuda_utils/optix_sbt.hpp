#pragma once

#include "cuda_utils/cuda_buffer.hpp"
#include "cuda_utils/optix_program.hpp"
#include "misc/check_macros.hpp"

class OptixSbt {
public:
    template <typename T>
    OptixSbt(const OptixProgram *program, const std::vector<T> &hitgroup_data) {
        EmptySbtRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(program->RaygenProgram(), &raygen_record));
        raygen_records_.AllocAndUpload(&raygen_record, sizeof(raygen_record));
        sbt_.raygenRecord = raygen_records_.DevicePtr();

        std::vector<EmptySbtRecord> miss_records(program->MissPrograms().size());
        for (size_t i = 0; i < program->MissPrograms().size(); i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(program->MissPrograms()[i], &miss_records[i]));
        }
        miss_records_.AllocAndUpload(miss_records.data(), miss_records.size() * sizeof(EmptySbtRecord));
        sbt_.missRecordBase = miss_records_.DevicePtr();
        sbt_.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt_.missRecordCount = miss_records.size();

        std::vector<TypedSbtRecord<T>> hitgroup_records;
        hitgroup_records.reserve(hitgroup_data.size() * program->HitgroupPrograms().size());
        for (size_t i = 0; i < hitgroup_data.size(); i++) {
            for (const OptixProgramGroup pg : program->HitgroupPrograms()) {
                TypedSbtRecord<T> rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
                rec.data = hitgroup_data[i];
                hitgroup_records.emplace_back(rec);
            }
        }
        hitgroup_records_.AllocAndUpload(
            hitgroup_records.data(),
            hitgroup_records.size() * sizeof(TypedSbtRecord<T>)
        );
        sbt_.hitgroupRecordBase = hitgroup_records_.DevicePtr();
        sbt_.hitgroupRecordStrideInBytes = sizeof(TypedSbtRecord<T>);
        sbt_.hitgroupRecordCount = hitgroup_records.size();
    }

    template <>
    OptixSbt(const OptixProgram *program, const std::vector<std::nullptr_t> &hitgroup_data) {
        EmptySbtRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(program->RaygenProgram(), &raygen_record));
        raygen_records_.AllocAndUpload(&raygen_record, sizeof(raygen_record));
        sbt_.raygenRecord = raygen_records_.DevicePtr();

        std::vector<EmptySbtRecord> miss_records(program->MissPrograms().size());
        for (size_t i = 0; i < program->MissPrograms().size(); i++) {
            OPTIX_CHECK(optixSbtRecordPackHeader(program->MissPrograms()[i], &miss_records[i]));
        }
        miss_records_.AllocAndUpload(miss_records.data(), miss_records.size() * sizeof(EmptySbtRecord));
        sbt_.missRecordBase = miss_records_.DevicePtr();
        sbt_.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt_.missRecordCount = miss_records.size();

        std::vector<EmptySbtRecord> hitgroup_records;
        hitgroup_records.reserve(hitgroup_data.size() * program->HitgroupPrograms().size());
        for (size_t i = 0; i < hitgroup_data.size(); i++) {
            for (const OptixProgramGroup pg : program->HitgroupPrograms()) {
                EmptySbtRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
                hitgroup_records.emplace_back(rec);
            }
        }
        hitgroup_records_.AllocAndUpload(
            hitgroup_records.data(),
            hitgroup_records.size() * sizeof(EmptySbtRecord)
        );
        sbt_.hitgroupRecordBase = hitgroup_records_.DevicePtr();
        sbt_.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt_.hitgroupRecordCount = hitgroup_records.size();
    }

    const OptixShaderBindingTable *SbtPtr() const {
        return &sbt_;
    }

private:
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    template <typename T>
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) TypedSbtRecord {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    OptixShaderBindingTable sbt_ = {};
    CudaBuffer raygen_records_;
    CudaBuffer miss_records_;
    CudaBuffer hitgroup_records_;
};