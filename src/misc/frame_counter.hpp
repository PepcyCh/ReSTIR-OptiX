#pragma once

#include <chrono>

class FrameCounter {
public:
    void Reset();

    void Tick();

    double TotalTime() const;

    double DeltaTime() const;

    double Fps() const;

    double Mspf() const;

private:
    uint32_t frame_count_ = 0;
    double time_elapsed_ = 0.0;
    double fps_ = 0.0;
    double mspf_ = 0.0;

    std::chrono::time_point<std::chrono::high_resolution_clock> base_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> prev_time_;

    double delta_time_ = 0.0;
};