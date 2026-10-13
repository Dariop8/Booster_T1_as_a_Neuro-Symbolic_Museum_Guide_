#!/usr/bin/env bash
set -euo pipefail

BUILD_ONLY=0
if [ "${1:-}" = "--build-only" ]; then
  BUILD_ONLY=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER="${CONTAINER_NAME:-}"
if [ -z "$CONTAINER" ]; then
  CONTAINER="$(docker ps --format '{{.Names}}' | grep '^CIRCUS_.*Booster-T1.*container$' | head -n 1 || true)"
fi

if [ -z "$CONTAINER" ]; then
  echo "ERROR: Booster container not found. Start Circus first." >&2
  exit 1
fi

IFACE="${BOOSTER_IFACE:-lo}"

SRC_HOST="$SCRIPT_DIR/walk_leg.cpp"
SRC_CONTAINER="/opt/booster/sdk/example/high_level/walk_leg.cpp"
BIN_CONTAINER="/opt/booster/sdk/build/walk_leg"

if [ ! -f "$SRC_HOST" ]; then
  echo "ERROR: source file not found: $SRC_HOST" >&2
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
  echo "[Walk Runner] Building walk_leg..." >&2
  docker exec "$CONTAINER" bash -lc "mkdir -p /opt/booster/sdk/example/high_level /opt/booster/sdk/build"
  docker cp "$SRC_HOST" "$CONTAINER:$SRC_CONTAINER"

  docker exec "$CONTAINER" bash -lc "
    set -e
    cd /opt/booster/sdk
    g++ example/high_level/walk_leg.cpp \
      -o build/walk_leg \
      -std=c++17 \
      -I/opt/booster/sdk/include \
      -L/usr/local/lib \
      -lbooster_robotics_sdk \
      -lfastrtps \
      -lfastcdr \
      -lpthread
  "
else
  echo "[Walk Runner] Using cached walk_leg binary." >&2
fi

if [ "$BUILD_ONLY" = "1" ]; then
  echo "[Walk Runner] Build-only completed." >&2
  exit 0
fi

exec docker exec -i "$CONTAINER" "$BIN_CONTAINER" "$IFACE"
