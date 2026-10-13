#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
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


static const std::string kTopicJointCtrl = "rt/joint_ctrl";
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
  std::string arm_action;
  std::string head_action;
  bool wait_for_enter = false;
  float arm_kp = 80.0f;
  float arm_kd = 2.0f;
  float head_kp = 45.0f;
  float head_kd = 1.2f;
  float weight = 1.0f;
  float control_dt = 0.02f;
  float arm_ramp = 2.0f;
  float head_ramp = 1.2f;
  float hold = 1.0f;
  std::vector<float> arm_q;
  std::vector<float> head_q;
};

static std::vector<float> ParseFloatList(int &i, int argc, const char *argv[], const std::string &name) {
  if (i + 1 >= argc) {
    std::cerr << "ERROR: missing value after " << name << std::endl;
    std::exit(1);
  }
  ++i;
  std::vector<float> values;
  std::stringstream ss(argv[i]);
  std::string item;
  while (std::getline(ss, item, ',')) {
    values.push_back(std::stof(item));
  }
  return values;
}

static const std::map<std::string, std::array<float, 8>> kArmTargets = {
    {"point_left", {
      -1.690000f, -0.720000f, +0.000000f, -0.450000f,
      +0.100000f, +1.500000f, +0.100000f, +0.300000f}},

    {"point_right", {
        +0.000000f, -1.500000f, +0.050000f, -1.350000f,
        -1.690000f, +0.720000f, +0.000000f, +0.450000f}},

    {"point_front_left", {
        -1.30000f, -1.10000f, -0.100000f, -1.100000f,
        +0.100000f, +1.500000f, +0.100000f, +0.300000f}},

    {"point_front_right", {
        +0.000000f, -1.500000f, +0.050000f, -1.350000f,
        -1.30000f, +1.100000f, -0.050000f, +0.600000f}},

    {"neutral", {
        +0.000000f, -1.500000f, +0.050000f, -0.300000f,
        +0.000000f, +1.500000f, +0.050000f, +0.300000f}},

    {"point_right_mid", {
        +0.000000f, -1.500000f, +0.050000f, -0.300000f,
        -1.545876f, +0.967295f, +0.000000f, +0.300000f}},

    {"point_left_mid", {
        -1.545870f, -0.963783f, +0.000000f, -0.300000f,
        +0.000000f, +1.500000f, +0.050000f, +0.300000f}},
};

static const std::map<std::string, std::array<float, 2>> kHeadTargets = {
    {"center",                 {+0.000000f, +0.000000f}},
    {"look_point_left",        {+0.550000f, +0.000000f}},
    {"look_point_right",       {-0.550000f, +0.000000f}},
    {"look_point_front_left",  {+0.280000f, +0.000000f}},
    {"look_point_front_right", {-0.280000f, +0.000000f}},
    {"look_point_left_mid",    {+0.710000f, -0.110000f}},
    {"look_point_right_mid",   {-0.710000f, -0.110000f}},
};

static void PrintAvailableActions() {
  std::cout << "Available arm actions:" << std::endl;
  for (const auto &kv : kArmTargets) std::cout << "  " << kv.first << std::endl;
  std::cout << "Available head actions:" << std::endl;
  for (const auto &kv : kHeadTargets) std::cout << "  " << kv.first << std::endl;
}

static void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " networkInterface armAction headAction [options]\n\n"
            << "Examples:\n"
            << "  " << argv0 << " lo point_front_left look_point_front_left\n"
            << "  " << argv0 << " lo neutral center --arm-ramp 1.0 --head-ramp 1.2 --hold 0.3\n\n"
            << "Options:\n"
            << "  --wait               wait for ENTER before starting\n"
            << "  --list               list available actions and exit\n"
            << "  --arm-ramp VALUE      seconds from current arm pose to target, default 2.0\n"
            << "  --head-ramp VALUE     seconds from current head pose to target, default 1.2\n"
            << "  --hold VALUE          hold both targets then exit, default 1.0\n"
            << "  --control-dt VALUE    command period in seconds, default 0.02\n"
            << "  --weight VALUE        command authority, default 1\n"
            << "  --arm-q \"v0,...,v7\"   explicit arm joint targets (radians), overrides armAction\n"
            << "  --head-q \"v0,v1\"      explicit head joint targets (radians), overrides headAction\n"
            << "  --help, -h           show this help\n";
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

  if (argc == 2 && std::string(argv[1]) == "--list") {
    PrintAvailableActions();
    std::exit(0);
  }

  if (argc < 4) {
    PrintUsage(argv[0]);
    std::exit(1);
  }

  opt.network_interface = argv[1];
  opt.arm_action = argv[2];
  opt.head_action = argv[3];

  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--wait") {
      opt.wait_for_enter = true;
    } else if (arg == "--list") {
      PrintAvailableActions();
      std::exit(0);
    } else if (arg == "--arm-ramp") {
      opt.arm_ramp = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--head-ramp") {
      opt.head_ramp = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--hold") {
      opt.hold = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--control-dt") {
      opt.control_dt = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--weight") {
      opt.weight = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--arm-q") {
      opt.arm_q = ParseFloatList(i, argc, argv, arg);
    } else if (arg == "--head-q") {
      opt.head_q = ParseFloatList(i, argc, argv, arg);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "ERROR: unknown argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      std::exit(1);
    }
  }

  if (opt.arm_q.empty() && kArmTargets.find(opt.arm_action) == kArmTargets.end()) {
    std::cerr << "ERROR: unknown arm action: " << opt.arm_action << std::endl;
    PrintAvailableActions();
    std::exit(1);
  }
  if (!opt.arm_q.empty() && opt.arm_q.size() != 8) {
    std::cerr << "ERROR: --arm-q needs exactly 8 comma-separated values, got "
               << opt.arm_q.size() << std::endl;
    std::exit(1);
  }
  if (opt.head_q.empty() && kHeadTargets.find(opt.head_action) == kHeadTargets.end()) {
    std::cerr << "ERROR: unknown head action: " << opt.head_action << std::endl;
    PrintAvailableActions();
    std::exit(1);
  }
  if (!opt.head_q.empty() && opt.head_q.size() != 2) {
    std::cerr << "ERROR: --head-q needs exactly 2 comma-separated values, got "
               << opt.head_q.size() << std::endl;
    std::exit(1);
  }

  if (opt.control_dt <= 0.0f || opt.arm_ramp < 0.0f || opt.head_ramp < 0.0f || opt.hold < 0.0f) {
    std::cerr << "ERROR: invalid timing parameter." << std::endl;
    std::exit(1);
  }

  return opt;
}

static float SmoothStep(float x) {
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return x * x * (3.0f - 2.0f * x);
}

static void PrintPoses(const std::string &title,
                       const std::array<float, 8> &arm,
                       const std::array<float, 2> &head) {
  std::cout << title << std::endl;
  std::cout << "  arm : ";
  for (float v : arm) std::cout << v << " ";
  std::cout << std::endl;
  std::cout << "  head: " << head[0] << " " << head[1] << std::endl;
}

int main(int argc, char const *argv[]) {
  const Options opt = ParseOptions(argc, argv);
  std::array<float, 8> arm_target;
  if (!opt.arm_q.empty()) {
    std::copy(opt.arm_q.begin(), opt.arm_q.end(), arm_target.begin());
  } else {
    arm_target = kArmTargets.at(opt.arm_action);
  }
  std::array<float, 2> head_target;
  if (!opt.head_q.empty()) {
    std::copy(opt.head_q.begin(), opt.head_q.end(), head_target.begin());
  } else {
    head_target = kHeadTargets.at(opt.head_action);
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

  const std::array<int, 2> head_joints = {0, 1};

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
                << "Check simulator and topic " << kTopicLowState << "." << std::endl;
      return 1;
    }
  }

  booster_interface::msg::LowState initial_state;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    initial_state = g_latest_state;
  }

  std::array<float, 8> arm_start{};
  for (int j = 0; j < 8; ++j) {
    arm_start[j] = initial_state.motor_state_parallel()
                       .at(static_cast<int>(arm_joints[j])).q();
  }
  std::array<float, 2> head_start{};
  for (int j = 0; j < 2; ++j) {
    head_start[j] = initial_state.motor_state_parallel().at(head_joints[j]).q();
  }

  booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> publisher;
  publisher.reset(
      new booster::robot::ChannelPublisher<booster_interface::msg::LowCmd>(kTopicJointCtrl));
  publisher->InitChannel();

  booster_interface::msg::LowCmd cmd_msg;
  for (size_t i = 0; i < booster::robot::b1::kJointCnt; i++) {
    booster_interface::msg::MotorCmd motor_cmd;
    cmd_msg.motor_cmd().push_back(motor_cmd);
  }

  std::cout << std::endl;
  std::cout << "Arm+head goal ramp: " << opt.arm_action << " + " << opt.head_action << std::endl;
  PrintPoses("Measured start pose:", arm_start, head_start);
  PrintPoses("Target pose:", arm_target, head_target);
  std::cout << "arm_ramp=" << opt.arm_ramp << "s, head_ramp=" << opt.head_ramp
            << "s, hold=" << opt.hold << "s, control_dt=" << opt.control_dt << std::endl;
  std::cout << std::endl;

  if (opt.wait_for_enter) {
    std::cout << "Press ENTER to start..." << std::endl;
    std::cin.get();
  }

  const auto sleep_time =
      std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f));

  auto fill_command = [&](const std::array<float, 8> &q_arm,
                          const std::array<float, 2> &q_head) {
    for (int j = 0; j < 8; ++j) {
      const int idx = static_cast<int>(arm_joints[j]);
      cmd_msg.motor_cmd().at(idx).q(q_arm[j]);
      cmd_msg.motor_cmd().at(idx).dq(0.0f);
      cmd_msg.motor_cmd().at(idx).kp(opt.arm_kp);
      cmd_msg.motor_cmd().at(idx).kd(opt.arm_kd);
      cmd_msg.motor_cmd().at(idx).tau(0.0f);
      cmd_msg.motor_cmd().at(idx).weight(opt.weight);
    }
    for (int j = 0; j < 2; ++j) {
      const int idx = head_joints[j];
      cmd_msg.motor_cmd().at(idx).q(q_head[j]);
      cmd_msg.motor_cmd().at(idx).dq(0.0f);
      cmd_msg.motor_cmd().at(idx).kp(opt.head_kp);
      cmd_msg.motor_cmd().at(idx).kd(opt.head_kd);
      cmd_msg.motor_cmd().at(idx).tau(0.0f);
      cmd_msg.motor_cmd().at(idx).weight(opt.weight);
    }
  };

  std::cout << "Ramping arm and head to targets..." << std::endl;
  const float ramp_total = std::max(opt.arm_ramp, opt.head_ramp);
  const int steps = std::max(1, static_cast<int>(ramp_total / opt.control_dt));

  for (int i = 0; i <= steps; ++i) {
    const float t = static_cast<float>(i) * opt.control_dt;
    const float a_arm = SmoothStep(opt.arm_ramp > 0.0f ? t / opt.arm_ramp : 1.0f);
    const float a_head = SmoothStep(opt.head_ramp > 0.0f ? t / opt.head_ramp : 1.0f);

    std::array<float, 8> q_arm{};
    for (int j = 0; j < 8; ++j) {
      q_arm[j] = (1.0f - a_arm) * arm_start[j] + a_arm * arm_target[j];
    }
    std::array<float, 2> q_head{};
    for (int j = 0; j < 2; ++j) {
      q_head[j] = (1.0f - a_head) * head_start[j] + a_head * head_target[j];
    }

    fill_command(q_arm, q_head);
    publisher->Write(&cmd_msg);
    std::this_thread::sleep_for(sleep_time);
  }

  std::cout << "Holding targets for " << opt.hold << " seconds..." << std::endl;
  const int hold_steps = std::max(1, static_cast<int>(opt.hold / opt.control_dt));
  for (int i = 0; i < hold_steps; ++i) {
    fill_command(arm_target, head_target);
    publisher->Write(&cmd_msg);
    std::this_thread::sleep_for(sleep_time);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  booster_interface::msg::LowState final_state;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    final_state = g_latest_state;
  }
  std::array<float, 8> arm_final{};
  for (int j = 0; j < 8; ++j) {
    arm_final[j] = final_state.motor_state_parallel()
                       .at(static_cast<int>(arm_joints[j])).q();
  }
  std::array<float, 2> head_final{};
  for (int j = 0; j < 2; ++j) {
    head_final[j] = final_state.motor_state_parallel().at(head_joints[j]).q();
  }

  std::cout << std::endl;
  PrintPoses("Measured final pose:", arm_final, head_final);

  float max_err = 0.0f;
  for (int j = 0; j < 8; ++j) max_err = std::max(max_err, std::abs(arm_final[j] - arm_target[j]));
  for (int j = 0; j < 2; ++j) max_err = std::max(max_err, std::abs(head_final[j] - head_target[j]));
  std::cout << "Max joint tracking error: " << max_err << " rad" << std::endl;

  return 0;
}
