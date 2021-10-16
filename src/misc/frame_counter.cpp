#include "misc/frame_counter.hpp"

namespace chrono = std::chrono;

void FrameCounter::Reset() {
    frame_count_ = 0;

    base_time_ = chrono::high_resolution_clock::now();
    prev_time_ = base_time_;
    delta_time_ = 0.0;
}

void FrameCounter::Tick() {
    frame_count_ += 1;

    const auto curr_time = chrono::high_resolution_clock::now();
    const chrono::duration<double> delta = curr_time - prev_time_;
    delta_time_ = delta.count();
    prev_time_ = curr_time;

    time_elapsed_ += delta_time_;
    if (time_elapsed_ >= 1.0) {
        fps_ = frame_count_ / time_elapsed_;
        mspf_ = 1000.0 * time_elapsed_ / frame_count_;
        frame_count_ = 0;
        time_elapsed_ = 0.0;
    }
}

double FrameCounter::TotalTime() const {
    const chrono::duration<double> delta = prev_time_ - base_time_;
    return delta.count();
}

double FrameCounter::DeltaTime() const {
    return delta_time_;
}

double FrameCounter::Fps() const {
    return fps_;
}

double FrameCounter::Mspf() const {
    return mspf_;
}