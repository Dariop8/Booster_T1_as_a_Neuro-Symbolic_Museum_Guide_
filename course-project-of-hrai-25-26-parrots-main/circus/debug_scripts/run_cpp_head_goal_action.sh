#!/usr/bin/env bash
set -euo pipefail

BUILD_ONLY=0

if [ "${1:-}" = "--build-only" ]; then
  BUILD_ONLY=1
  ACTION="center"
  shift || true
else
  ACTION="${1:-}"
  if [ -z "$ACTION" ]; then
    echo "Usage: $0 <center|look_left|look_right|look_front_left|look_front_right|look_up|look_down|...> [extra args]"
    echo "       $0 --build-only"
    echo
    echo "Examples:"
    echo "  $0 look_left"
    echo "  $0 center --yaw 0.25 --pitch -0.10"
    echo "  FORCE_REBUILD=1 $0 look_right"
    exit 1
  fi
  shift || true
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

SRC_HOST="$SCRIPT_DIR/head_goal_ramp_actions.cpp"
SRC_CONTAINER="/opt/booster/sdk/example/high_level/head_goal_ramp_actions.cpp"
BIN_CONTAINER="/opt/booster/sdk/build/head_goal_ramp_actions"

DEFAULT_ARGS=(
  --ramp-time 1.0
  --final-hold-time 0.6
)

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
  echo "[Head Runner] Building head_goal_ramp_actions..."
  docker exec "$CONTAINER" bash -lc "mkdir -p /opt/booster/sdk/example/high_level /opt/booster/sdk/build"
  docker cp "$SRC_HOST" "$CONTAINER:$SRC_CONTAINER"

  docker exec "$CONTAINER" bash -lc "
    set -e
    cd /opt/booster/sdk
    g++ example/high_level/head_goal_ramp_actions.cpp \
      -o build/head_goal_ramp_actions \
      -std=c++17 \
      -I/opt/booster/sdk/include \
      -L/usr/local/lib \
      -lbooster_robotics_sdk \
      -lfastrtps \
      -lfastcdr \
      -lpthread
  "
else
  echo "[Head Runner] Using cached head_goal_ramp_actions binary."
fi

if [ "$BUILD_ONLY" = "1" ]; then
  echo "[Head Runner] Build-only completed."
  exit 0
fi

echo "Running head goal action..."
echo "Action:    $ACTION"
echo "Interface: $IFACE"

docker exec "$CONTAINER" "$BIN_CONTAINER" "$IFACE" "$ACTION" "${DEFAULT_ARGS[@]}" "$@"
