#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace spqr {

/**
 * Moves the scene objects that declare a `motion:` route (SceneParser.h's
 * MotionSpec) along it, once per simulation step.
 *
 * Why kinematic rather than physical: a museum visitor is not a ragdoll. What
 * the navigation stack needs from them is an obstacle that goes where the
 * scene author said it would, keeps going after the robot bumps into it, and
 * never falls over — none of which a freejoint body with legs it does not have
 * would do. MuJoCo already has exactly this: a mocap body is excluded from the
 * integrator and placed by writing data->mocap_pos, while still colliding with
 * everything dynamic. SceneParser builds the body; this class writes the pose.
 *
 * The robot learns none of this from here. Where a walker actually is reaches
 * Python through the same channel the static visitors always used —
 * PeopleGroundTruthWriter reads body positions out of mjData, so a person that
 * moves is published as a person that moves without a line changing there —
 * and people_sensor.py still decides, by range and line of sight, how much of
 * that the robot is allowed to know. Making a visitor walk therefore does not
 * hand the robot a better sensor, only a harder room.
 *
 * Route semantics: waypoints are followed at a constant speed, dwelling
 * pause_s on arrival at each. `loop` cycles back to the first waypoint after
 * the last (a circuit); otherwise the route is walked back the way it came (a
 * ping-pong, which is what "pacing between two paintings" looks like).
 *
 * The body is also TURNED to face where it is going. That mattered not at all
 * while the visitors were cylinders and matters a great deal now that they are
 * human meshes: a figure sliding sideways across the gallery reads as a bug.
 * The heading is chased at kTurnRateRadPerSec rather than snapped, so the turn
 * at a waypoint happens visibly during the dwell — a person pivoting, not a
 * sprite flipping. `forward_yaw` accounts for the model's own front not being
 * +x (see MotionSpec).
 */
class PeopleWalker {
    public:
        struct Walker {
                std::string bodyName;
                std::vector<std::array<float, 2>> waypoints;
                float speed = 0.3f;
                float pause_s = 0.0f;
                bool loop = false;
                float forward_yaw = 0.0f;  // yaw of the model's own front when unrotated
        };

        // How fast a walker pivots onto a new heading. A person turning a
        // corner takes well under a second for a right angle; the routes in the
        // museum scene dwell 1.5-2 s at each waypoint, so the turn completes
        // during the pause and the walker sets off already facing the right way.
        static constexpr double kTurnRateRadPerSec = 2.5;

        /**
         * Resolves each walker's mocap slot. A body that does not exist, or
         * exists but was not built as a mocap body, is skipped with a warning
         * rather than being fatal: the scene should still load and run, one
         * visitor short, exactly as PeopleGroundTruthWriter::configure treats
         * a name it cannot resolve.
         */
        void configure(const std::vector<Walker>& walkers, const mjModel* model) {
            state_.clear();
            lastTime_ = -1.0;
            if (walkers.empty() || model == nullptr) return;

            for (const Walker& w : walkers) {
                const int bodyId = mj_name2id(model, mjOBJ_BODY, w.bodyName.c_str());
                if (bodyId < 0) {
                    std::cerr << "PeopleWalker: no body named '" << w.bodyName << "' — not animated." << std::endl;
                    continue;
                }
                const int mocapId = model->body_mocapid[bodyId];
                if (mocapId < 0) {
                    std::cerr << "PeopleWalker: body '" << w.bodyName << "' is not a mocap body — not animated." << std::endl;
                    continue;
                }
                State s;
                s.spec = w;
                s.mocapId = mocapId;
                // Height is authored once in the YAML and never varies along
                // the route: these are people walking on a flat floor, and
                // taking z from the model keeps the one number that says how
                // tall they stand in a single place.
                s.z = model->body_pos[3 * bodyId + 2];
                s.pauseLeft = w.pause_s;  // dwell at the starting waypoint first
                s.yaw = headingOf(s);     // start facing the first leg, don't spin on frame 1
                state_.push_back(std::move(s));
            }
        }

        /** Advances every walker by the time elapsed since the previous call. */
        void update(mjData* data) {
            if (state_.empty() || data == nullptr) return;

            const double now = data->time;
            // First call, or the simulation was reset back to t=0: re-anchor
            // instead of applying a nonsensical dt.
            if (lastTime_ < 0.0 || now < lastTime_) {
                lastTime_ = now;
                for (State& s : state_) write(data, s);
                return;
            }
            const double dt = now - lastTime_;
            lastTime_ = now;
            if (dt <= 0.0) return;  // paused: mj_step isn't running, so nobody walks

            for (State& s : state_) {
                advance(s, dt);
                turn(s, dt);
                write(data, s);
            }
        }

    private:
        struct State {
                Walker spec;
                int mocapId = -1;
                double z = 0.0;
                std::size_t from = 0;   // index of the waypoint just left
                std::size_t to = 1;     // index of the waypoint being walked to
                int dir = 1;            // ping-pong direction; unused when looping
                double travelled = 0.0; // meters covered on the current segment
                double pauseLeft = 0.0;
                double yaw = 0.0;       // current body yaw, chased toward the heading
        };

        /** World yaw the body should hold to face along the current segment. */
        static double headingOf(const State& s) {
            const auto& a = s.spec.waypoints[s.from];
            const auto& b = s.spec.waypoints[s.to];
            const double dx = static_cast<double>(b[0]) - a[0], dy = static_cast<double>(b[1]) - a[1];
            if (std::abs(dx) < 1e-9 && std::abs(dy) < 1e-9) return s.yaw;  // degenerate leg: hold
            return std::atan2(dy, dx) - s.spec.forward_yaw;
        }

        static double wrapPi(double a) {
            while (a > M_PI) a -= 2.0 * M_PI;
            while (a < -M_PI) a += 2.0 * M_PI;
            return a;
        }

        /** Chase the heading at a bounded rate, the short way round. */
        static void turn(State& s, double dt) {
            const double error = wrapPi(headingOf(s) - s.yaw);
            const double step = kTurnRateRadPerSec * dt;
            s.yaw = wrapPi(s.yaw + (std::abs(error) <= step ? error : std::copysign(step, error)));
        }

        static double segmentLength(const State& s) {
            const auto& a = s.spec.waypoints[s.from];
            const auto& b = s.spec.waypoints[s.to];
            return std::hypot(static_cast<double>(b[0]) - a[0], static_cast<double>(b[1]) - a[1]);
        }

        /** Waypoint reached: pick the next leg and start the dwell. */
        static void arrive(State& s) {
            const std::size_t n = s.spec.waypoints.size();
            s.from = s.to;
            if (s.spec.loop) {
                s.to = (s.to + 1) % n;
            } else {
                // Ping-pong: turn around at either end of the route.
                if ((s.dir > 0 && s.from + 1 >= n) || (s.dir < 0 && s.from == 0)) s.dir = -s.dir;
                s.to = static_cast<std::size_t>(static_cast<long>(s.from) + s.dir);
            }
            s.travelled = 0.0;
            s.pauseLeft = s.spec.pause_s;
        }

        static void advance(State& s, double dt) {
            if (s.pauseLeft > 0.0) {
                s.pauseLeft -= dt;
                return;
            }
            // Loop rather than a single step because a route may contain
            // segments shorter than one step's worth of travel; it terminates
            // because every iteration either consumes all of `remaining` or
            // retires a whole segment.
            double remaining = static_cast<double>(s.spec.speed) * dt;
            while (remaining > 0.0) {
                const double segLen = segmentLength(s);
                if (segLen < 1e-9) {  // duplicated waypoints: nothing to walk
                    arrive(s);
                    break;
                }
                const double left = segLen - s.travelled;
                if (remaining < left) {
                    s.travelled += remaining;
                    break;
                }
                remaining -= left;
                arrive(s);
                if (s.pauseLeft > 0.0) break;  // dwell here before setting off again
            }
        }

        void write(mjData* data, const State& s) const {
            const auto& a = s.spec.waypoints[s.from];
            const auto& b = s.spec.waypoints[s.to];
            const double segLen = segmentLength(s);
            const double t = segLen > 1e-9 ? s.travelled / segLen : 0.0;
            mjtNum* pos = data->mocap_pos + 3 * s.mocapId;
            pos[0] = a[0] + t * (static_cast<double>(b[0]) - a[0]);
            pos[1] = a[1] + t * (static_cast<double>(b[1]) - a[1]);
            pos[2] = s.z;

            // Upright by construction: these are people walking on a flat
            // floor, so the orientation is one yaw and never a full attitude.
            // Written every step because a mocap body takes its whole pose from
            // here — the body's own `euler` in the XML is ignored once it does.
            mjtNum* quat = data->mocap_quat + 4 * s.mocapId;
            quat[0] = std::cos(s.yaw * 0.5);
            quat[1] = 0.0;
            quat[2] = 0.0;
            quat[3] = std::sin(s.yaw * 0.5);
        }

        std::vector<State> state_;
        double lastTime_ = -1.0;
};

}  // namespace spqr
