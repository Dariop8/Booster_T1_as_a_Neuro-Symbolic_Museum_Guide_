#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <booster/idl/b1/LowCmd.h>
#include <booster/idl/b1/MotorCmd.h>
#include <booster/idl/b1/LowState.h>

#include <booster/robot/channel/channel_publisher.hpp>
#include <booster/robot/channel/channel_subscriber.hpp>
#include <booster/robot/b1/b1_api_const.hpp>


static const std::string kTopicArmSDK = "rt/joint_ctrl";
static const std::string kTopicLowState = "rt/low_state";

std::mutex g_state_mutex;
std::atomic<bool> g_have_state{false};
booster_interface::msg::LowState g_latest_state;

void LowStateCallback(const void *msg) {
  const auto *state = static_cast<const booster_interface::msg::LowState *>(msg);
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_latest_state = *state;
  }
  g_have_state.store(true);
}

struct Options {
  std::string network_interface = "lo";
  std::string hpp_path;
  bool wait_for_enter = false;
  float kp = 80.0f;
  float kd = 2.0f;
  float weight = 1.0f;
  float control_dt = 0.02f;
  float initial_ramp_time = 2.0f;
  float segment_time = 0.05f;
  float waypoint_hold_time = 0.0f;
  float final_hold_time = 1.0f;
  bool hold_forever = false;
  bool loop = false;
};

static void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " networkInterface trajectory.hpp [options]\n\n"
            << "Example:\n"
            << "  " << argv0 << " lo /app/exchange/perception_test/recorded_demos/point_left.hpp --wait\n\n"
            << "Options:\n"
            << "  --wait                         wait for ENTER before starting\n"
            << "  --kp VALUE                      proportional gain, default 80\n"
            << "  --kd VALUE                      derivative gain, default 2\n"
            << "  --weight VALUE                  command authority, default 1\n"
            << "  --control-dt VALUE              command period in seconds, default 0.02\n"
            << "  --initial-ramp-time VALUE       seconds from current pose to first setpoint, default 2.0\n"
            << "  --segment-time VALUE            seconds between consecutive setpoints, default 0.05\n"
            << "  --waypoint-hold-time VALUE      pause at each setpoint, default 0.0\n"
            << "  --final-hold-time VALUE         hold final pose then exit, default 1.0\n"
            << "  --hold-forever                  keep final pose until Ctrl+C\n"
            << "  --loop                          repeat trajectory until Ctrl+C\n";
}

static float ParseFloatArg(int &i, int argc, const char *argv[], const std::string &name) {
  if (i + 1 >= argc) {
    std::cerr << "ERROR: missing value after " << name << std::endl;
    std::exit(1);
  }
  ++i;
  return std::stof(argv[i]);
}

static Options ParseOptions(int argc, const char *argv[]) {
  Options opt;
  if (argc < 3) {
    PrintUsage(argv[0]);
    std::exit(1);
  }

  opt.network_interface = argv[1];
  opt.hpp_path = argv[2];

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--wait") {
      opt.wait_for_enter = true;
    } else if (arg == "--loop") {
      opt.loop = true;
    } else if (arg == "--hold-forever") {
      opt.hold_forever = true;
    } else if (arg == "--kp") {
      opt.kp = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--kd") {
      opt.kd = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--weight") {
      opt.weight = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--control-dt") {
      opt.control_dt = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--initial-ramp-time") {
      opt.initial_ramp_time = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--segment-time") {
      opt.segment_time = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--waypoint-hold-time") {
      opt.waypoint_hold_time = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--final-hold-time") {
      opt.final_hold_time = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "ERROR: unknown argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      std::exit(1);
    }
  }

  if (opt.control_dt <= 0.0f || opt.segment_time <= 0.0f || opt.initial_ramp_time < 0.0f || opt.final_hold_time < 0.0f) {
    std::cerr << "ERROR: invalid timing parameter." << std::endl;
    std::exit(1);
  }

  return opt;
}

static std::string ReadTextFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("could not open file: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string StripCppLineComments(const std::string &text) {
  std::stringstream input(text);
  std::ostringstream output;
  std::string line;
  while (std::getline(input, line)) {
    const size_t comment_pos = line.find("//");
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    output << line << '\n';
  }
  return output.str();
}

static std::vector<float> ExtractFloats(const std::string &text) {
  static const std::regex number_re(R"(([+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?)[fF]?)");
  std::vector<float> values;
  auto begin = std::sregex_iterator(text.begin(), text.end(), number_re);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    values.push_back(std::stof((*it)[1].str()));
  }
  return values;
}

static std::vector<std::array<float, 8>> LoadArmTrajectoryFromHpp(const std::string &path) {
  std::string text = StripCppLineComments(ReadTextFile(path));

  size_t pos = text.find("kArmTrajectoryTargets");
  if (pos == std::string::npos) {
    throw std::runtime_error("kArmTrajectoryTargets not found in: " + path);
  }

  size_t vec_open = text.find('{', pos);
  if (vec_open == std::string::npos) {
    throw std::runtime_error("opening brace for kArmTrajectoryTargets not found in: " + path);
  }

  std::vector<std::array<float, 8>> trajectory;
  int depth = 0;
  std::string current_inner;

  for (size_t i = vec_open; i < text.size(); ++i) {
    char c = text[i];

    if (c == '{') {
      ++depth;
      if (depth == 2) {
        current_inner.clear();
      }
      continue;
    }

    if (c == '}') {
      if (depth == 2) {
        const std::vector<float> values = ExtractFloats(current_inner);
        if (values.size() == 8) {
          std::array<float, 8> q{};
          for (int j = 0; j < 8; ++j) q[j] = values[j];
          trajectory.push_back(q);
        } else if (!values.empty()) {
          std::ostringstream err;
          err << "found an inner array with " << values.size()
              << " numbers instead of 8 in: " << path;
          throw std::runtime_error(err.str());
        }
      }
      --depth;
      if (depth == 0) break;
      continue;
    }

    if (depth == 2) {
      current_inner.push_back(c);
    }
  }

  if (trajectory.empty()) {
    throw std::runtime_error("no 8-joint arm setpoints parsed from: " + path);
  }

  return trajectory;
}

static float SmoothStep(float x) {
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return x * x * (3.0f - 2.0f * x);
}

static void PrintArmPose(const std::string &title, const std::array<float, 8> &q) {
  std::cout << title << std::endl;
  std::cout << "  0 Left_Shoulder_Pitch  : " << q[0] << std::endl;
  std::cout << "  1 Left_Shoulder_Roll   : " << q[1] << std::endl;
  std::cout << "  2 Left_Elbow_Pitch     : " << q[2] << std::endl;
  std::cout << "  3 Left_Elbow_Yaw       : " << q[3] << std::endl;
  std::cout << "  4 Right_Shoulder_Pitch : " << q[4] << std::endl;
  std::cout << "  5 Right_Shoulder_Roll  : " << q[5] << std::endl;
  std::cout << "  6 Right_Elbow_Pitch    : " << q[6] << std::endl;
  std::cout << "  7 Right_Elbow_Yaw      : " << q[7] << std::endl;
}

static void FillArmCommand(
    booster_interface::msg::LowCmd &cmd_msg,
    const std::array<booster::robot::b1::JointIndex, 8> &arm_joints,
    const std::array<float, 8> &q_arm,
    const Options &opt) {
  const float dq = 0.0f;
  const float tau_ff = 0.0f;

  for (int j = 0; j < static_cast<int>(arm_joints.size()); ++j) {
    auto idx = static_cast<int>(arm_joints.at(j));
    cmd_msg.motor_cmd().at(idx).q(q_arm[j]);
    cmd_msg.motor_cmd().at(idx).dq(dq);
    cmd_msg.motor_cmd().at(idx).kp(opt.kp);
    cmd_msg.motor_cmd().at(idx).kd(opt.kd);
    cmd_msg.motor_cmd().at(idx).tau(tau_ff);
    cmd_msg.motor_cmd().at(idx).weight(opt.weight);
  }
}

static void SendPoseForDuration(
    booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> &publisher,
    booster_interface::msg::LowCmd &cmd_msg,
    const std::array<booster::robot::b1::JointIndex, 8> &arm_joints,
    const std::array<float, 8> &q_arm,
    const Options &opt,
    float duration_s) {
  if (duration_s <= 0.0f) return;
  const auto sleep_time = std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f));
  int steps = std::max(1, static_cast<int>(duration_s / opt.control_dt));
  for (int i = 0; i < steps; ++i) {
    FillArmCommand(cmd_msg, arm_joints, q_arm, opt);
    publisher->Write(&cmd_msg);
    std::this_thread::sleep_for(sleep_time);
  }
}

static void SendInterpolatedSegment(
    booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> &publisher,
    booster_interface::msg::LowCmd &cmd_msg,
    const std::array<booster::robot::b1::JointIndex, 8> &arm_joints,
    const std::array<float, 8> &q0,
    const std::array<float, 8> &q1,
    const Options &opt,
    float duration_s) {
  const auto sleep_time = std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f));
  int steps = std::max(1, static_cast<int>(duration_s / opt.control_dt));

  for (int i = 0; i <= steps; ++i) {
    float raw_alpha = static_cast<float>(i) / static_cast<float>(steps);
    float alpha = SmoothStep(raw_alpha);

    std::array<float, 8> q{};
    for (int j = 0; j < 8; ++j) {
      q[j] = (1.0f - alpha) * q0[j] + alpha * q1[j];
    }

    FillArmCommand(cmd_msg, arm_joints, q, opt);
    publisher->Write(&cmd_msg);
    std::this_thread::sleep_for(sleep_time);
  }
}

int main(int argc, char const *argv[]) {
  const Options opt = ParseOptions(argc, argv);

  std::vector<std::array<float, 8>> trajectory;
  try {
    trajectory = LoadArmTrajectoryFromHpp(opt.hpp_path);
  } catch (const std::exception &e) {
    std::cerr << "ERROR while loading trajectory: " << e.what() << std::endl;
    return 1;
  }

  booster::robot::ChannelFactory::Instance()->Init(0, opt.network_interface);

  std::array<booster::robot::b1::JointIndex, 8> arm_joints = {
      booster::robot::b1::JointIndex::kLeftShoulderPitch,
      booster::robot::b1::JointIndex::kLeftShoulderRoll,
      booster::robot::b1::JointIndex::kLeftElbowPitch,
      booster::robot::b1::JointIndex::kLeftElbowYaw,
      booster::robot::b1::JointIndex::kRightShoulderPitch,
      booster::robot::b1::JointIndex::kRightShoulderRoll,
      booster::robot::b1::JointIndex::kRightElbowPitch,
      booster::robot::b1::JointIndex::kRightElbowYaw};

  auto state_subscriber =
      std::make_shared<booster::robot::ChannelSubscriber<booster_interface::msg::LowState>>(
          kTopicLowState);
  state_subscriber->InitChannel(LowStateCallback);

  std::cout << "Waiting for current low state from " << kTopicLowState << "..." << std::endl;

  const auto start_wait = std::chrono::steady_clock::now();
  while (!g_have_state.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - start_wait).count();
    if (elapsed > 5.0) {
      std::cerr << "ERROR: did not receive low state after 5 seconds.\n"
                << "Check that the simulator is running and that " << kTopicLowState << " exists." << std::endl;
      return 1;
    }
  }

  booster_interface::msg::LowState initial_state;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    initial_state = g_latest_state;
  }

  std::array<float, 8> start_pos{};
  for (int j = 0; j < static_cast<int>(arm_joints.size()); ++j) {
    const int idx = static_cast<int>(arm_joints.at(j));
    start_pos.at(j) = initial_state.motor_state_parallel().at(idx).q();
  }

  booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> publisher;
  publisher.reset(new booster::robot::ChannelPublisher<booster_interface::msg::LowCmd>(kTopicArmSDK));
  publisher->InitChannel();

  booster_interface::msg::LowCmd cmd_msg;
  for (size_t i = 0; i < booster::robot::b1::kJointCnt; i++) {
    booster_interface::msg::MotorCmd motor_cmd;
    cmd_msg.motor_cmd().push_back(motor_cmd);
  }

  std::cout << std::endl;
  std::cout << "Trajectory file: " << opt.hpp_path << std::endl;
  PrintArmPose("Measured start pose:", start_pos);
  std::cout << std::endl;
  std::cout << "Loaded trajectory setpoints: " << trajectory.size() << std::endl;
  std::cout << "kp=" << opt.kp << ", kd=" << opt.kd << ", weight=" << opt.weight << std::endl;
  std::cout << "control_dt=" << opt.control_dt
            << ", initial_ramp_time=" << opt.initial_ramp_time
            << ", segment_time=" << opt.segment_time
            << ", waypoint_hold_time=" << opt.waypoint_hold_time
            << ", final_hold_time=" << opt.final_hold_time
            << ", hold_forever=" << (opt.hold_forever ? "true" : "false") << std::endl;
  std::cout << std::endl;
  PrintArmPose("First setpoint:", trajectory.front());
  std::cout << std::endl;
  PrintArmPose("Last setpoint:", trajectory.back());
  std::cout << std::endl;

  if (opt.wait_for_enter) {
    std::cout << "Press ENTER to start arm trajectory..." << std::endl;
    std::cin.get();
  }

  do {
    std::cout << "Ramping from measured pose to setpoint 0..." << std::endl;
    SendInterpolatedSegment(
        publisher, cmd_msg, arm_joints, start_pos, trajectory.front(), opt,
        opt.initial_ramp_time);

    SendPoseForDuration(
        publisher, cmd_msg, arm_joints, trajectory.front(), opt,
        opt.waypoint_hold_time);

    for (size_t i = 0; i + 1 < trajectory.size(); ++i) {
      std::cout << "Segment " << i << " -> " << (i + 1) << " / "
                << (trajectory.size() - 1) << std::endl;
      SendInterpolatedSegment(
          publisher, cmd_msg, arm_joints,
          trajectory.at(i), trajectory.at(i + 1), opt,
          opt.segment_time);
      SendPoseForDuration(
          publisher, cmd_msg, arm_joints, trajectory.at(i + 1), opt,
          opt.waypoint_hold_time);
    }

    if (opt.loop) {
      std::cout << "Loop enabled: returning to first setpoint..." << std::endl;
      SendInterpolatedSegment(
          publisher, cmd_msg, arm_joints,
          trajectory.back(), trajectory.front(), opt,
          opt.segment_time);
    }
  } while (opt.loop);

  if (opt.hold_forever) {
    std::cout << "Holding final trajectory setpoint. Press Ctrl+C to stop." << std::endl;
    while (true) {
      FillArmCommand(cmd_msg, arm_joints, trajectory.back(), opt);
      publisher->Write(&cmd_msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f)));
    }
  } else {
    std::cout << "Holding final trajectory setpoint for " << opt.final_hold_time << " seconds, then exiting." << std::endl;
    SendPoseForDuration(publisher, cmd_msg, arm_joints, trajectory.back(), opt, opt.final_hold_time);
  }

  return 0;
}
