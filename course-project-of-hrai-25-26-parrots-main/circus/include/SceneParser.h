#pragma once

#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>
#include <array>
#include <memory>
#include <pugixml.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "Team.h"
#include "robots/Robot.h"

using namespace pugi;
using namespace std;
using namespace Eigen;

namespace spqr {

struct BallSpec {
        Vector3d position;
};

struct FieldConfig {
        float width = 14.0f;
        float height = 9.0f;
        float center_radius = 1.5f;
        float goal_area_width = 1.0f;
        float goal_area_height = 4.0f;
        float penalty_area_width = 3.0f;
        float penalty_area_height = 6.5f;
        float penalty_mark_distance = 2.1f;
        float goal_width = 2.6f;
        float goal_height = 1.8f;
        float goal_depth = 0.6f;
        float line_width = 0.08f;
        float ball_radius = 0.11f;
};

struct SimulationSettings {
        int max_simulation_time = -1;  // -1 means no limit
};

struct GameConfig {
        bool enable_field = true;
        std::string field = "fieldAdultSize";
        bool game_state_logging = true;
        std::string game_state_logging_path = "game_state.log";
        float game_state_logging_interval = 1.0f;
        int game_duration = 600;
        bool automatic_restart = true;
        int initial_phase_duration = 30;
        int ready_phase_duration = 45;
        int set_phase_duration = 10;
        int kickoff_subphase_duration = 10;
        int other_subphase_duration = 30;
        std::string first_kickoff_team = "red";
        int penalty_duration = 45;
};

struct SimulationConfig {
        SimulationSettings simulation;
        FieldConfig field;
        GameConfig game;
};

/**
 * Scripted, purely kinematic motion along a floor-plane route.
 *
 * Only objects that declare `motion:` in the scene YAML move; everything else
 * is welded exactly where the YAML put it, as before. An object carrying one
 * of these is built as a MuJoCo mocap body (see SceneParser::buildMuJoCoXml)
 * and advanced along the route once per simulation step by PeopleWalker —
 * physics never moves it and nothing can push it off its route, which is what
 * a scripted museum visitor needs to be: an obstacle with its own agenda, not
 * a ragdoll that falls over the first time the robot brushes it.
 */
struct MotionSpec {
        std::vector<std::array<float, 2>> waypoints;  // route in world x/y; z stays at the authored height
        float speed   = 0.3f;    // m/s along the route
        float pause_s = 0.0f;    // dwell time on reaching each waypoint
        bool  loop    = false;   // true = cycle back to waypoint 0; false = ping-pong
        // Yaw, in radians, at which this object's own FRONT points when the
        // object is unrotated — a property of the mesh, not of the route.
        // PeopleWalker turns the body to (direction of travel - forward_yaw),
        // so a model whose front is +x needs 0 and one whose front is -y needs
        // -pi/2. Irrelevant for the rotationally symmetric primitives, which is
        // why nothing needed it until the visitors became meshes.
        float forward_yaw = 0.0f;
};

struct ObjectSpec {
        std::string name;
        std::string type;             // box | sphere | cylinder | mesh
        std::string mesh;             // mesh filename (without extension) — used when type==mesh
        std::string texture;          // PNG path relative to resources/textures/ (optional)
        // Free-form tag read by the navigation stack, not by MuJoCo. "person"
        // marks an object the robot must SENSE rather than look up: those are
        // kept out of the hand-authored static map (nav_map.py's
        // PERCEIVED_ROLES) and published instead as a runtime ground-truth
        // measurement (PeopleGroundTruthWriter.h), standing in for the person
        // detector the T1 does not physically have.
        std::string role;
        std::array<float, 3> pos   = {0, 0, 0};
        std::array<float, 3> size  = {0.05f, 0.05f, 0.05f};
        float scale                = 1.0f;
        std::array<float, 4> rgba  = {0.8f, 0.8f, 0.8f, 1.0f};
        bool has_rgba              = false;  // true = rgba was set explicitly in YAML
        // Body rotation, roll/pitch/yaw in radians, same convention the robots
        // already use. Only meaningful for objects with a distinguishable
        // front: a box or a cylinder does not care, a person does. IGNORED for
        // an object with a `motion:` route — MuJoCo takes a mocap body's pose
        // entirely from mocap_pos/mocap_quat, and PeopleWalker writes the yaw
        // there from the direction of travel instead.
        std::array<float, 3> orientation = {0, 0, 0};
        bool has_orientation       = false;
        bool fixed                 = false;  // true = welded to world (no free joint)
        // Scripted route, present only for objects that declare `motion:`.
        // has_motion (not an empty waypoint list) is the flag, so a route can
        // legitimately be authored as a single waypoint — a person who stands
        // still today and is given somewhere to go tomorrow.
        MotionSpec motion;
        bool has_motion            = false;
};

struct LightSpec {
        std::array<float, 3> pos      = {0, 0, 3};
        std::array<float, 3> dir      = {0, 0, -1};
        std::array<float, 3> diffuse  = {0.5f, 0.5f, 0.5f};
        std::array<float, 3> ambient  = {0, 0, 0};
        std::array<float, 3> specular = {0, 0, 0};
        bool castshadow                = false;
};

struct CellData {
        int row;
        int column;
        string stream;
};

struct GuiConfig {
        int rows = 1;
        int columns = 4;
        std::vector<CellData> cellData;
};

struct SceneSpec {
        SimulationConfig simulationConfig;
        std::vector<std::shared_ptr<Team>> teams;
        GuiConfig guiConfig;
        std::vector<ObjectSpec> objects;
        // Optional top-level "disable_default_floor: true" — skips the
        // debug checker/plane floor entirely (game.enable_field == false
        // scenes only), so the scene can define its own floor as a normal
        // sized+textured object in "objects:" instead of an infinite plane.
        // Default false = unchanged checker behavior for existing scenes.
        bool disable_default_floor = false;
        // Optional top-level "lights:" list — replaces the single default
        // flat ambient light with a custom set (e.g. warm shadow-casting
        // gallery lighting). Empty = unchanged single-light behavior for
        // existing scenes.
        std::vector<LightSpec> lights;
};

class SceneParser {
    public:
        SceneParser(const string& yamlPath);
        string buildMuJoCoXml();
        const SceneSpec& getSceneInfo() const {
            return scene;
        }

    private:
        void buildRobotCommon(const string& robotType, xml_node& mujoco);
        void buildRobotInstance(const shared_ptr<Robot>& robotSpec, xml_node& worldbody, xml_node& actuator, xml_node& sensor);
        void buildObjects(xml_node& mujoco, xml_node& worldbody);
        void prefixSubtree(xml_node& root, const std::string& robotName);

        unordered_set<string> robotTypes;
        YAML::Node sceneRoot;
        SceneSpec scene;
        BallSpec ballSpec;
};

}  // namespace spqr
