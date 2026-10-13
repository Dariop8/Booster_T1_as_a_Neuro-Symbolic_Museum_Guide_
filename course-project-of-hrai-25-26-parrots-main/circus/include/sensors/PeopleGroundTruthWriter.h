#pragma once

#include <mujoco/mujoco.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "sensors/ImageSharedMemoryWriter.h"

namespace spqr {

/**
 * Publishes the ground-truth footprint of every scene object tagged
 * "role: person" into shared memory, once per simulation step.
 *
 * The Booster T1 carries an RGB-D head camera and nothing else — no lidar, no
 * person detector, no tracker. The guide's obstacle avoidance therefore has no
 * sensor that could tell it a visitor is standing in the way. This channel
 * stands in for that missing hardware: the positions come straight out of
 * MuJoCo, exactly as a perfect sensor would report them.
 *
 * What is published here is the raw truth and nothing else. The SENSOR MODEL —
 * range, line of sight, how long a track is remembered after it stops being
 * seen — lives on the Python side in people_sensor.py, where it can be tuned
 * without rebuilding this binary. Truth is measured here; what the robot is
 * allowed to know is decided there.
 *
 * Deliberately NOT the same thing as nav_map.get_perceived_objects(), which
 * reads the same visitors out of the scene YAML: that one is the benchmarks'
 * answer key and must never reach the planner, or a perception result would be
 * scored against the very file it was supposed to discover. This is a runtime
 * measurement channel; that one is an offline yardstick.
 *
 * Reading body positions from mjData rather than echoing back the YAML values
 * costs nothing today, since the visitors are static, and is the whole point
 * tomorrow: the moment a person gets a freejoint or an actuator, this channel
 * reports where they actually are without a line changing here.
 *
 * Payload — float32, tightly packed, kPersonStride values per person:
 *     x, y, z, radius, height
 * One person per "pixel column", so a reader recovers the count from the
 * shared-memory header's width field. Carrier is the same
 * ImageSharedMemoryWriter BoosterT1 already uses for rgb/depth/pose, for the
 * same reason: one IPC mechanism in the codebase, not three.
 */
class PeopleGroundTruthWriter {
    public:
        struct Person {
                std::string bodyName;
                float radius = 0.f;  // meters, footprint radius on the floor
                float height = 0.f;  // meters, full height (not half-extent)
        };

        static constexpr int kPersonStride = 5;

        /**
         * Resolves each person's MuJoCo body and opens the shared-memory
         * channel. A name that does not resolve is skipped rather than
         * fatal: an object can be commented out of the scene YAML mid-session
         * and that should cost one absent track, not a scene that won't load.
         */
        void configure(const std::string& path, const std::vector<Person>& people, const mjModel* model) {
            people_.clear();
            body_ids_.clear();
            values_.clear();
            buffer_.clear();
            if (people.empty() || model == nullptr) return;

            for (const Person& p : people) {
                const int id = mj_name2id(model, mjOBJ_BODY, p.bodyName.c_str());
                if (id < 0) continue;
                people_.push_back(p);
                body_ids_.push_back(id);
            }
            if (people_.empty()) return;

            writer_.configure(path, static_cast<int>(people_.size()), 1, kPersonStride * sizeof(float));
            values_.assign(people_.size() * kPersonStride, 0.f);
            buffer_.assign(values_.size() * sizeof(float), 0);
        }

        void update(const mjData* data) {
            if (people_.empty() || data == nullptr || !writer_.isReady()) return;

            for (size_t i = 0; i < people_.size(); ++i) {
                // xpos is the body frame origin in world coordinates — for the
                // fixed bodies SceneParser builds that is the YAML position,
                // and for a body given a freejoint it is wherever physics put
                // it this step.
                const mjtNum* xpos = data->xpos + 3 * body_ids_[i];
                float* out = values_.data() + i * kPersonStride;
                out[0] = static_cast<float>(xpos[0]);
                out[1] = static_cast<float>(xpos[1]);
                out[2] = static_cast<float>(xpos[2]);
                out[3] = people_[i].radius;
                out[4] = people_[i].height;
            }
            std::memcpy(buffer_.data(), values_.data(), buffer_.size());
            writer_.write(buffer_);
        }

    private:
        ImageSharedMemoryWriter writer_;
        std::vector<Person> people_;
        std::vector<int> body_ids_;
        std::vector<float> values_;
        std::vector<uint8_t> buffer_;
};

}  // namespace spqr
