#!/usr/bin/env bash
set -euo pipefail


BUILD_ONLY=0

if [ "${1:-}" = "--build-only" ]; then
  BUILD_ONLY=1
  ARM_ACTION="neutral"
  HEAD_ACTION="center"
  shift || true
else
  ARM_ACTION="${1:-}"
  HEAD_ACTION="${2:-}"
  if [ -z "$ARM_ACTION" ] || [ -z "$HEAD_ACTION" ]; then
    echo "Usage: $0 <point_left|point_right|point_front_left|point_front_right|neutral> <look_point_...|center> [extra args]"
    echo "       $0 --build-only"
    echo
    echo "Examples:"
    echo "  $0 point_front_left look_point_front_left"
    echo "  $0 neutral center --arm-ramp 1.0 --head-ramp 1.2 --hold 0.3"
    exit 1
  fi
  shift 2 || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER="${CONTAINER_NAME:-}"
if [ -z "$CONTAINER" ]; then
  CONTAINER="$(docker ps --format '{{.Names}}' | grep '^CIRCUS_.*Booster-T1.*container$' | head -n 1 || true)"
fi

if [ -z "$CONTAINER" ]; then
  echo "ERROR: Booster container not found. Start Circus first."
  exit 1
fi

IFACE="${BOOSTER_IFACE:-lo}"

SRC_HOST="$SCRIPT_DIR/arm_head_goal_ramp_actions.cpp"
SRC_CONTAINER="/opt/booster/sdk/example/high_level/arm_head_goal_ramp_actions.cpp"
BIN_CONTAINER="/opt/booster/sdk/build/arm_head_goal_ramp_actions"

if [ ! -f "$SRC_HOST" ]; then
  echo "ERROR: source file not found: $SRC_HOST"
  exit 1
fi

HOST_HASH="$(md5sum "$SRC_HOST" | awk '{print $1}')"
NEED_REBUILD=0

if [ "${FORCE_REBUILD:-0}" = "1" ]; then
  NEED_REBUILD=1
elif ! docker exec "$CONTAINER" bash -lc "test -x '$BIN_CONTAINER'" >/dev/null 2>&1; then
  NEED_REBUILD=1
elif ! docker exec "$CONTAINER" bash -lc "test -f '$SRC_CONTAINER'" >/dev/null 2>&1; then
  NEED_REBUILD=1
else
  CONTAINER_HASH="$(
    docker exec "$CONTAINER" bash -lc "md5sum '$SRC_CONTAINER' 2>/dev/null | cut -d' ' -f1" \
      | tr -d '\r'
  )"

  if [ "$HOST_HASH" != "$CONTAINER_HASH" ]; then
    NEED_REBUILD=1
  fi
fi

if [ "$NEED_REBUILD" = "1" ]; then
  echo "[ArmHead Runner] Building arm_head_goal_ramp_actions..."
  docker exec "$CONTAINER" bash -lc "mkdir -p /opt/booster/sdk/example/high_level /opt/booster/sdk/build"
  docker cp "$SRC_HOST" "$CONTAINER:$SRC_CONTAINER"

  docker exec "$CONTAINER" bash -lc "
    set -e
    cd /opt/booster/sdk
    g++ example/high_level/arm_head_goal_ramp_actions.cpp \
      -o build/arm_head_goal_ramp_actions \
      -std=c++17 \
      -I/opt/booster/sdk/include \
      -L/usr/local/lib \
      -lbooster_robotics_sdk \
      -lfastrtps \
      -lfastcdr \
      -lpthread
  "
else
  echo "[ArmHead Runner] Using cached arm_head_goal_ramp_actions binary."
fi

if [ "$BUILD_ONLY" = "1" ]; then
  echo "[ArmHead Runner] Build-only completed."
  exit 0
fi

echo "Running arm+head goal action..."
echo "Arm action:  $ARM_ACTION"
echo "Head action: $HEAD_ACTION"
echo "Interface:   $IFACE"

docker exec "$CONTAINER" "$BIN_CONTAINER" "$IFACE" "$ARM_ACTION" "$HEAD_ACTION" "$@"
