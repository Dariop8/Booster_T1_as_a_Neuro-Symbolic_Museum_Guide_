#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCUS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CIRCUS_DIR/.." && pwd)"

CONTAINER="${CONTAINER:-CIRCUS_red_Booster-T1_0_container}"
IFACE="${IFACE:-lo}"
RECORDED_HOST="${RECORDED_HOST:-$REPO_ROOT/exchange/perception_test/recorded_demos}"
RECORDED_CONTAINER="${RECORDED_CONTAINER:-/app/exchange/perception_test/recorded_demos}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ACTION_NAME|PATH_TO_HPP [commander options]"
  echo
  echo "Examples:"
  echo "  $0 point_left --initial-ramp-time 2.0 --segment-time 0.05"
  echo "  $0 point_front_right --initial-ramp-time 2.0 --segment-time 0.05"
  echo "  $0 /absolute/path/to/custom.hpp --hold-forever"
  exit 1
fi

ACTION_OR_HPP="$1"
shift

if [[ "$ACTION_OR_HPP" == *.hpp ]]; then
  HPP_HOST="$ACTION_OR_HPP"
  ACTION_NAME="$(basename "$ACTION_OR_HPP" .hpp)"
  if [[ "$HPP_HOST" == "$RECORDED_HOST"/* ]]; then
    REL="${HPP_HOST#$RECORDED_HOST/}"
    HPP_CONTAINER="$RECORDED_CONTAINER/$REL"
  else
    HPP_CONTAINER="/tmp/${ACTION_NAME}.hpp"
  fi
else
  ACTION_NAME="$ACTION_OR_HPP"
  HPP_HOST="$RECORDED_HOST/${ACTION_NAME}.hpp"
  HPP_CONTAINER="$RECORDED_CONTAINER/${ACTION_NAME}.hpp"
fi

if [[ ! -f "$SCRIPT_DIR/arm_trajectory_from_hpp.cpp" ]]; then
  echo "ERROR: missing $SCRIPT_DIR/arm_trajectory_from_hpp.cpp"
  exit 1
fi

if [[ ! -f "$HPP_HOST" ]]; then
  echo "ERROR: missing trajectory header: $HPP_HOST"
  echo "Expected one of:"
  echo "  $RECORDED_HOST/point_left.hpp"
  echo "  $RECORDED_HOST/point_right.hpp"
  echo "  $RECORDED_HOST/point_front_left.hpp"
  echo "  $RECORDED_HOST/point_front_right.hpp"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container $CONTAINER is not running. Start Circus first."
  exit 1
fi

SDK_DIR="/opt/booster/sdk"
DST_DIR="$SDK_DIR/example/high_level"

DEFAULT_ARGS=(--initial-ramp-time 2.0 --segment-time 0.05 --waypoint-hold-time 0.0 --final-hold-time 1.0)
EXTRA_ARGS=("$@")

echo "Copying runtime HPP trajectory commander into $CONTAINER..."
docker cp "$SCRIPT_DIR/arm_trajectory_from_hpp.cpp" "$CONTAINER:$DST_DIR/arm_trajectory_from_hpp.cpp"

if [[ "$HPP_CONTAINER" == /tmp/* ]]; then
  echo "Copying external HPP into container: $HPP_CONTAINER"
  docker cp "$HPP_HOST" "$CONTAINER:$HPP_CONTAINER"
fi

echo "Compiling arm_trajectory_from_hpp..."
docker exec "$CONTAINER" bash -lc "cd $SDK_DIR && g++ example/high_level/arm_trajectory_from_hpp.cpp -o build/arm_trajectory_from_hpp -std=c++17 -I/opt/booster/sdk/include -L/usr/local/lib -lbooster_robotics_sdk -lfastrtps -lfastcdr -lpthread"

echo "Running arm action trajectory..."
echo "Action:    $ACTION_NAME"
echo "Interface: $IFACE"
echo "HPP host:  $HPP_HOST"
echo "HPP robot: $HPP_CONTAINER"
echo

docker exec -ti "$CONTAINER" /opt/booster/sdk/build/arm_trajectory_from_hpp "$IFACE" "$HPP_CONTAINER" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}"
