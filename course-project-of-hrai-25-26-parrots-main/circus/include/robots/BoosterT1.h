#pragma once

#include <mujoco/mujoco.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/types.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>
#include <cmath>
#include <cstring>
#include <memory>
#include <msgpack.hpp>
#include <msgpack/v3/object_fwd_decl.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "MujocoContext.h"
#include "robots/Robot.h"
#include "sensors/CameraDepth.h"
#include "sensors/CameraRGB.h"
#include "sensors/ImageSharedMemoryWriter.h"
#include "sensors/Imu.h"
#include "sensors/Joint.h"
#include "sensors/Pose.h"

#define MAX_MSG_SIZE 1048576  // 1MB
namespace spqr {

struct Team;  // Forward declaration

class BoosterT1 : public Robot {
    public:
        Pose* pose = nullptr;
        Imu* imu = nullptr;
        Joints* joints = nullptr;
        CameraRGB* rgbCamera;
        CameraDepth* depthCamera;

        BoosterT1(const std::string& name, const std::string& type, uint8_t number, const Eigen::Vector3d& initPosition,
                  const Eigen::Vector3d& initOrientation, const std::tuple<int, int, int> color, const std::shared_ptr<Team>& team)
            : Robot(name, type, number, initPosition, initOrientation, color, team),
              joint_map{{JointValue::HEAD_YAW, name + "_AAHead_yaw"},
                        {JointValue::HEAD_PITCH, name + "_Head_pitch"},
                        {JointValue::SHOULDER_LEFT_PITCH, name + "_Left_Shoulder_Pitch"},
                        {JointValue::SHOULDER_LEFT_ROLL, name + "_Left_Shoulder_Roll"},
                        {JointValue::ELBOW_LEFT_PITCH, name + "_Left_Elbow_Pitch"},
                        {JointValue::ELBOW_LEFT_YAW, name + "_Left_Elbow_Yaw"},
                        {JointValue::SHOULDER_RIGHT_PITCH, name + "_Right_Shoulder_Pitch"},
                        {JointValue::SHOULDER_RIGHT_ROLL, name + "_Right_Shoulder_Roll"},
                        {JointValue::ELBOW_RIGHT_PITCH, name + "_Right_Elbow_Pitch"},
                        {JointValue::ELBOW_RIGHT_YAW, name + "_Right_Elbow_Yaw"},
                        {JointValue::WAIST, name + "_Waist"},
                        {JointValue::HIP_LEFT_PITCH, name + "_Left_Hip_Pitch"},
                        {JointValue::HIP_LEFT_ROLL, name + "_Left_Hip_Roll"},
                        {JointValue::HIP_LEFT_YAW, name + "_Left_Hip_Yaw"},
                        {JointValue::KNEE_LEFT_PITCH, name + "_Left_Knee_Pitch"},
                        {JointValue::ANKLE_LEFT_PITCH, name + "_Left_Ankle_Pitch"},
                        {JointValue::ANKLE_LEFT_ROLL, name + "_Left_Ankle_Roll"},
                        {JointValue::HIP_RIGHT_PITCH, name + "_Right_Hip_Pitch"},
                        {JointValue::HIP_RIGHT_ROLL, name + "_Right_Hip_Roll"},
                        {JointValue::HIP_RIGHT_YAW, name + "_Right_Hip_Yaw"},
                        {JointValue::KNEE_RIGHT_PITCH, name + "_Right_Knee_Pitch"},
                        {JointValue::ANKLE_RIGHT_PITCH, name + "_Right_Ankle_Pitch"},
                        {JointValue::ANKLE_RIGHT_ROLL, name + "_Right_Ankle_Roll"}} {
            shm_dir_ = "/dev/shm/circus_ipc";
        }

        void bindMujoco(MujocoContext* mujCtx) override {
            pose = new Pose(mujCtx->model, mujCtx->data, (name + "_position").c_str(), (name + "_orientation").c_str());
            imu = new Imu(mujCtx->model, mujCtx->data, (name + "_linear-acceleration").c_str(), (name + "_angular-velocity").c_str());
            joints = new Joints(mujCtx->model, mujCtx->data, joint_map);

            joints->set_position({{JointValue::HEAD_YAW, 0},
                                  {JointValue::HEAD_PITCH, 0},
                                  {JointValue::SHOULDER_LEFT_PITCH, 0},
                                  {JointValue::SHOULDER_LEFT_ROLL, 0},
                                  {JointValue::ELBOW_LEFT_PITCH, 0},
                                  {JointValue::ELBOW_LEFT_YAW, 0},
                                  {JointValue::SHOULDER_RIGHT_PITCH, 0},
                                  {JointValue::SHOULDER_RIGHT_ROLL, 0},
                                  {JointValue::ELBOW_RIGHT_PITCH, 0},
                                  {JointValue::ELBOW_RIGHT_YAW, 0},
                                  {JointValue::WAIST, 0},
                                  {JointValue::HIP_LEFT_PITCH, 0},
                                  {JointValue::HIP_LEFT_ROLL, 0},
                                  {JointValue::HIP_LEFT_YAW, 0},
                                  {JointValue::KNEE_LEFT_PITCH, 0},
                                  {JointValue::ANKLE_LEFT_PITCH, 0},
                                  {JointValue::ANKLE_LEFT_ROLL, 0},
                                  {JointValue::HIP_RIGHT_PITCH, 0},
                                  {JointValue::HIP_RIGHT_ROLL, 0},
                                  {JointValue::HIP_RIGHT_YAW, 0},
                                  {JointValue::KNEE_RIGHT_PITCH, 0},
                                  {JointValue::ANKLE_RIGHT_PITCH, 0},
                                  {JointValue::ANKLE_RIGHT_ROLL, 0}});

            rgbCamera = new CameraRGB(mujCtx, (name + "_rgb_cam").c_str());
            // Use RGB viewpoint for simulated depth to provide aligned depth-to-color.
            depthCamera = new CameraDepth(mujCtx, (name + "_rgb_cam").c_str());

            const int width = rgbCamera->getWidth();
            const int height = rgbCamera->getHeight();
            rgb_writer_.configure(shmFilePath_("rgb"), width, height, 3);
            depth_writer_.configure(shmFilePath_("depth"), width, height, 1);

            // Ground-truth base pose for path planning (nav_map.py /
            // path_planner.py / locomotion.py): the room's own map is
            // already fully known (authored by hand in the scene YAML), so
            // the only missing piece for navigation was localization — no
            // SLAM/LIDAR needed in simulation, MuJoCo already knows exactly
            // where the robot is. Reuses ImageSharedMemoryWriter as a
            // generic byte-buffer channel (6 floats: x,y,z,roll,pitch,yaw),
            // same IPC mechanism as rgb/depth above, just a much smaller
            // payload.
            pose_writer_.configure(shmFilePath_("pose"), 1, 1, 6 * sizeof(float));
        }

        void receiveMessage(const std::map<std::string, msgpack::object>& message) override {
            auto it = message.find("joint_torques");
            if (it == message.end()) {
                throw std::runtime_error("Error: 'joint_torques' key not found in message");
                return;
            }

            std::vector<double> joint_torques = it->second.as<std::vector<double>>();

            if (joint_torques.size() != joint_map.size()) {
                throw std::runtime_error("Error: joint_torques size (" + std::to_string(joint_torques.size()) + ") doesn't match number of joints ("
                                         + std::to_string(joint_map.size()) + ")");
            }

            std::unordered_map<JointValue, mjtNum> torque_map;
            size_t i = 0;
            for (const auto& [joint_value, joint_name] : joint_map) {
                torque_map[joint_value] = joint_torques[i++];
            }

            joints->set_torque(torque_map);
        }

        std::map<std::string, msgpack::object> sendMessage() override {
            buffer_zone_.clear();
            std::map<std::string, msgpack::object> msg;
            msg["robot_name"] = msgpack::object(name, buffer_zone_);
            msg["pose"] = pose->serialize(buffer_zone_);
            msg["imu"] = imu->serialize(buffer_zone_);
            msg["joints"] = joints->serialize(buffer_zone_);

            return msg;
        }

        std::map<std::string, Sensor*> getSensors() override {
            std::map<std::string, Sensor*> sensors;
            sensors["pose"] = pose;
            sensors["imu"] = imu;
            sensors["joints"] = joints;
            sensors["rgb_camera"] = rgbCamera;
            sensors["depth_camera"] = depthCamera;
            return sensors;
        }

        void update() override {
            pose->update();
            imu->update();
            joints->update();
            rgbCamera->update();
            depthCamera->update();
            rgb_writer_.write(rgbCamera->getImage());
            depth_writer_.write(depthCamera->getDepth8bit());

            const Eigen::Vector3d& p = pose->getPosition();
            const Eigen::Vector3d& e = pose->getEulerOrientation();
            // Yaw specifically is NOT taken from getEulerOrientation(): Eigen's
            // eulerAngles() is not continuous — it can jump by roughly pi
            // between two very similar physical orientations (representation
            // ambiguity near its internal branch cuts). Observed live: a
            // one-burst yaw jump of ~2.8 rad with no corresponding real turn,
            // which sent locomotion.py's heading controller into a feedback
            // oscillation. Yaw (rotation about world Z) is computed directly
            // from the quaternion instead — continuous regardless of
            // roll/pitch, unaffected by that ambiguity. Quaternion is
            // (w, x, y, z) per MuJoCo's framequat sensor convention.
            const Eigen::Vector4d& q = pose->getQuatOrientation();
            const double yaw = std::atan2(2.0 * (q(0) * q(3) + q(1) * q(2)),
                                           1.0 - 2.0 * (q(2) * q(2) + q(3) * q(3)));
            const float pose6[6] = {
                static_cast<float>(p(0)), static_cast<float>(p(1)), static_cast<float>(p(2)),
                static_cast<float>(e(0)), static_cast<float>(e(1)), static_cast<float>(yaw),
            };
            std::vector<uint8_t> pose_bytes(sizeof(pose6));
            std::memcpy(pose_bytes.data(), pose6, sizeof(pose6));
            pose_writer_.write(pose_bytes);
        }

        ~BoosterT1() = default;

    private:
        std::string shmFilePath_(const std::string& camera) const {
            return shm_dir_ + "/" + name + "_" + camera + ".shm";
        }

        std::map<JointValue, std::string> joint_map;
        std::string shm_dir_;
        ImageSharedMemoryWriter rgb_writer_;
        ImageSharedMemoryWriter depth_writer_;
        ImageSharedMemoryWriter pose_writer_;
};

}  // namespace spqr
