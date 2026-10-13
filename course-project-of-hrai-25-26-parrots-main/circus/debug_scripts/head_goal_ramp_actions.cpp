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
#include <string>
#include <thread>

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
  std::string action_name;
  bool wait_for_enter = false;
  float kp = 45.0f;
  float kd = 1.2f;
  float weight = 1.0f;
  float control_dt = 0.02f;
  float ramp_time = 1.0f;
  float final_hold_time = 0.6f;
  bool hold_forever = false;
  bool custom_target = false;
  float custom_yaw = 0.0f;
  float custom_pitch = 0.0f;
};

static const std::map<std::string, std::array<float, 2>> kHeadTargets = {
    {"center",          {+0.000000f, +0.000000f}},
    {"look_center",     {+0.000000f, +0.000000f}},

    {"yaw_pos",         {+0.350000f, +0.000000f}},
    {"yaw_neg",         {-0.350000f, +0.000000f}},
    {"pitch_pos",       {+0.000000f, +0.180000f}},
    {"pitch_neg",       {+0.000000f, -0.180000f}},

    {"look_left",       {+0.450000f, +0.000000f}},
    {"look_right",      {-0.450000f, +0.000000f}},
    {"look_front_left", {+0.250000f, +0.000000f}},
    {"look_front_right",{-0.250000f, +0.000000f}},

    {"look_up",         {+0.000000f, -0.180000f}},
    {"look_down",       {+0.000000f, +0.180000f}},

    {"look_point_left",        {+0.550000f, +0.000000f}},
    {"look_point_right",       {-0.550000f, +0.000000f}},
    {"look_point_front_left",  {+0.280000f, +0.000000f}},
    {"look_point_front_right", {-0.280000f, +0.000000f}},
    {"look_point_left_mid",    {+0.710000f, -0.110000f}},
    {"look_point_right_mid",   {-0.710000f, -0.110000f}},
};

static void PrintAvailableActions() {
  std::cout << "Available head actions:" << std::endl;
  for (const auto &kv : kHeadTargets) {
    std::cout << "  " << kv.first << std::endl;
  }
  std::cout << std::endl;
  std::cout << "You can also override any action with: --yaw VALUE --pitch VALUE" << std::endl;
}

static void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " networkInterface actionName [options]\n\n"
            << "Examples:\n"
            << "  " << argv0 << " lo look_left\n"
            << "  " << argv0 << " lo center\n"
            << "  " << argv0 << " lo center --yaw 0.25 --pitch -0.10\n"
            << "  " << argv0 << " lo look_point_right --ramp-time 0.8 --final-hold-time 0.5\n\n"
            << "Options:\n"
            << "  --wait                    wait for ENTER before starting\n"
            << "  --list                    list available actions and exit\n"
            << "  --yaw VALUE                override target yaw, rad\n"
            << "  --pitch VALUE              override target pitch, rad\n"
            << "  --kp VALUE                 proportional gain, default 45\n"
            << "  --kd VALUE                 derivative gain, default 1.2\n"
            << "  --weight VALUE             command authority, default 1\n"
            << "  --control-dt VALUE         command period in seconds, default 0.02\n"
            << "  --ramp-time VALUE          seconds from current pose to target, default 1.0\n"
            << "  --final-hold-time VALUE    hold target then exit, default 0.6\n"
            << "  --hold-forever             keep target pose until Ctrl+C\n"
            << "  --help, -h                 show this help\n";
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

  if (argc < 3) {
    PrintUsage(argv[0]);
    std::exit(1);
  }

  opt.network_interface = argv[1];
  opt.action_name = argv[2];

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--wait") {
      opt.wait_for_enter = true;
    } else if (arg == "--hold-forever") {
      opt.hold_forever = true;
    } else if (arg == "--list") {
      PrintAvailableActions();
      std::exit(0);
    } else if (arg == "--yaw") {
      opt.custom_yaw = ParseFloatArg(i, argc, argv, arg);
      opt.custom_target = true;
    } else if (arg == "--pitch") {
      opt.custom_pitch = ParseFloatArg(i, argc, argv, arg);
      opt.custom_target = true;
    } else if (arg == "--kp") {
      opt.kp = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--kd") {
      opt.kd = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--weight") {
      opt.weight = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--control-dt") {
      opt.control_dt = ParseFloatArg(i, argc, argv, arg);
    } else if (arg == "--ramp-time") {
      opt.ramp_time = ParseFloatArg(i, argc, argv, arg);
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

  if (kHeadTargets.find(opt.action_name) == kHeadTargets.end()) {
    std::cerr << "ERROR: unknown head action: " << opt.action_name << std::endl;
    PrintAvailableActions();
    std::exit(1);
  }

  if (opt.control_dt <= 0.0f || opt.ramp_time < 0.0f || opt.final_hold_time < 0.0f) {
    std::cerr << "ERROR: invalid timing parameter." << std::endl;
    std::exit(1);
  }

  opt.custom_yaw = std::max(-0.80f, std::min(+0.80f, opt.custom_yaw));
  opt.custom_pitch = std::max(-0.40f, std::min(+0.40f, opt.custom_pitch));

  return opt;
}

static float SmoothStep(float x) {
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return x * x * (3.0f - 2.0f * x);
}

static void PrintHeadPose(const std::string &title, const std::array<float, 2> &q) {
  std::cout << title << std::endl;
  std::cout << "  0 AAHead_yaw : " << q[0] << std::endl;
  std::cout << "  1 Head_pitch : " << q[1] << std::endl;
}

static void PrintHeadError(
    const std::string &title,
    const std::array<float, 2> &measured,
    const std::array<float, 2> &target) {
  std::cout << title << std::endl;
  std::cout << "  0 AAHead_yaw : " << measured[0] - target[0] << std::endl;
  std::cout << "  1 Head_pitch : " << measured[1] - target[1] << std::endl;
}

static std::array<float, 2> ExtractHeadPoseFromState(
    const booster_interface::msg::LowState &state,
    const std::array<int, 2> &head_joints) {
  std::array<float, 2> q{};
  for (int j = 0; j < static_cast<int>(head_joints.size()); ++j) {
    const int idx = head_joints.at(j);
    q.at(j) = state.motor_state_parallel().at(idx).q();
  }
  return q;
}

static std::array<float, 2> ReadLatestHeadPose(const std::array<int, 2> &head_joints) {
  booster_interface::msg::LowState state;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    state = g_latest_state;
  }
  return ExtractHeadPoseFromState(state, head_joints);
}

static void FillHeadCommand(
    booster_interface::msg::LowCmd &cmd_msg,
    const std::array<int, 2> &head_joints,
    const std::array<float, 2> &q_head,
    const Options &opt) {
  const float dq = 0.0f;
  const float tau_ff = 0.0f;

  for (int j = 0; j < static_cast<int>(head_joints.size()); ++j) {
    const int idx = head_joints.at(j);
    cmd_msg.motor_cmd().at(idx).q(q_head[j]);
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
    const std::array<int, 2> &head_joints,
    const std::array<float, 2> &q_head,
    const Options &opt,
    float duration_s) {
  if (duration_s <= 0.0f) return;
  const auto sleep_time = std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f));
  int steps = std::max(1, static_cast<int>(duration_s / opt.control_dt));
  for (int i = 0; i < steps; ++i) {
    FillHeadCommand(cmd_msg, head_joints, q_head, opt);
    publisher->Write(&cmd_msg);
    std::this_thread::sleep_for(sleep_time);
  }
}

static void SendInterpolatedSegment(
    booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> &publisher,
    booster_interface::msg::LowCmd &cmd_msg,
    const std::array<int, 2> &head_joints,
    const std::array<float, 2> &q0,
    const std::array<float, 2> &q1,
    const Options &opt,
    float duration_s) {
  const auto sleep_time = std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f));
  int steps = std::max(1, static_cast<int>(duration_s / opt.control_dt));

  for (int i = 0; i <= steps; ++i) {
    const float raw_alpha = static_cast<float>(i) / static_cast<float>(steps);
    const float alpha = SmoothStep(raw_alpha);

    std::array<float, 2> q{};
    for (int j = 0; j < 2; ++j) {
      q[j] = (1.0f - alpha) * q0[j] + alpha * q1[j];
    }

    FillHeadCommand(cmd_msg, head_joints, q, opt);
    publisher->Write(&cmd_msg);
    std::this_thread::sleep_for(sleep_time);
  }
}

int main(int argc, char const *argv[]) {
  const Options opt = ParseOptions(argc, argv);

  std::array<float, 2> target = kHeadTargets.at(opt.action_name);
  if (opt.custom_target) {
    target = {opt.custom_yaw, opt.custom_pitch};
  }

  booster::robot::ChannelFactory::Instance()->Init(0, opt.network_interface);

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

  const std::array<float, 2> start_pos = ReadLatestHeadPose(head_joints);

  booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> publisher;
  publisher.reset(new booster::robot::ChannelPublisher<booster_interface::msg::LowCmd>(kTopicJointCtrl));
  publisher->InitChannel();

  booster_interface::msg::LowCmd cmd_msg;
  for (size_t i = 0; i < booster::robot::b1::kJointCnt; i++) {
    booster_interface::msg::MotorCmd motor_cmd;
    cmd_msg.motor_cmd().push_back(motor_cmd);
  }

  std::cout << std::endl;
  std::cout << "Head goal ramp action: " << opt.action_name << std::endl;
  if (opt.custom_target) {
    std::cout << "Custom override active." << std::endl;
  }
  PrintHeadPose("Measured start pose:", start_pos);
  std::cout << std::endl;
  PrintHeadPose("Target pose:", target);
  std::cout << std::endl;
  std::cout << "kp=" << opt.kp << ", kd=" << opt.kd << ", weight=" << opt.weight << std::endl;
  std::cout << "control_dt=" << opt.control_dt
            << ", ramp_time=" << opt.ramp_time
            << ", final_hold_time=" << opt.final_hold_time
            << ", hold_forever=" << (opt.hold_forever ? "true" : "false") << std::endl;
  std::cout << std::endl;

  if (opt.wait_for_enter) {
    std::cout << "Press ENTER to start head goal ramp..." << std::endl;
    std::cin.get();
  }

  std::cout << "Ramping from measured pose to target..." << std::endl;
  SendInterpolatedSegment(
      publisher, cmd_msg, head_joints, start_pos, target, opt, opt.ramp_time);

  if (opt.hold_forever) {
    std::cout << "Holding target pose. Press Ctrl+C to stop." << std::endl;
    while (true) {
      FillHeadCommand(cmd_msg, head_joints, target, opt);
      publisher->Write(&cmd_msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(opt.control_dt * 1000.0f)));
    }
  } else {
    std::cout << "Holding target pose for " << opt.final_hold_time
              << " seconds, then reading final measured pose." << std::endl;
    SendPoseForDuration(publisher, cmd_msg, head_joints, target, opt, opt.final_hold_time);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const std::array<float, 2> final_pos = ReadLatestHeadPose(head_joints);
    std::cout << std::endl;
    PrintHeadPose("Measured final pose:", final_pos);
    std::cout << std::endl;
    PrintHeadError("Final tracking error, measured - target:", final_pos, target);
  }

  return 0;
}
