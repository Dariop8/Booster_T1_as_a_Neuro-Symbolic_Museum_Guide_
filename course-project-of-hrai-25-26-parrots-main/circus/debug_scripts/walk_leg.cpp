
#include <booster/robot/b1/b1_loco_client.hpp>
#include <booster/robot/channel/channel_factory.hpp>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using namespace booster::robot;
using namespace booster::robot::b1;

static constexpr int kCommandHz = 25;

static B1LocoClient* g_client = nullptr;

static void on_signal(int) {
    if (g_client) g_client->Move(0, 0, 0);
    std::exit(0);
}

int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <network_interface>\n", argv[0]);
        return 1;
    }
    const std::string iface = argv[1];

    ChannelFactory::Instance()->Init(0, iface);

    B1LocoClient client;
    client.Init();
    g_client = &client;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    client.ChangeMode(RobotMode::kWalking);

    const auto period = std::chrono::milliseconds(1000 / kCommandHz);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line.rfind("STOP", 0) == 0) break;

        std::istringstream iss(line);
        float vx, vy, vyaw, duration_s;
        if (!(iss >> vx >> vy >> vyaw >> duration_s)) {
            std::fprintf(stderr, "walk_leg: bad command line: %s\n", line.c_str());
            continue;
        }

        const int n_ticks = static_cast<int>(duration_s * kCommandHz);
        for (int i = 0; i < n_ticks; ++i) {
            client.Move(vx, vy, vyaw);
            std::this_thread::sleep_for(period);
        }
        client.Move(0.0f, 0.0f, 0.0f);
        std::printf("walk_leg: done vx=%.3f vy=%.3f vyaw=%.3f dur=%.2f\n", vx, vy, vyaw, duration_s);
        std::fflush(stdout);
    }

    client.Move(0.0f, 0.0f, 0.0f);
    return 0;
}
