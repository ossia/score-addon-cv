#!/usr/bin/env bash
# Standalone test runner for score-addon-cv.
#
# Builds and runs the Catch2 unit tests WITHOUT a full score build: the CV objects are
# plain structs, so they only need halp/avnd headers, Eigen and magic_enum. This gives a
# fast (~30s) verification loop.
#
# Usage:
#   tests/run_standalone.sh                 # build + run everything
#   tests/run_standalone.sh "[canny]"       # build + run only tests with that Catch2 tag
#
# Requires: the Catch2 already fetched by a configured score build (see CATCH_DIR below).
set -euo pipefail

ADDON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORE=/home/jcelerier/ossia/score
CATCH_BUILD=/home/jcelerier/builds/build-asan-ubsan/_deps
OUT="${TMPDIR:-/tmp}/score_addon_cv_tests"

if [[ ! -d "$CATCH_BUILD/catch2cv-src" ]]; then
  echo "error: Catch2 not found at $CATCH_BUILD/catch2cv-src" >&2
  echo "       configure a score build with -DSCORE_ADDON_CV_TESTS=ON first." >&2
  exit 1
fi

INCLUDES=(
  -I"$ADDON"
  -I"$SCORE/3rdparty/avendish/include"
  -I"$SCORE/3rdparty/eigen"
  -I"$SCORE/3rdparty/libossia/3rdparty/magic_enum/include/magic_enum"
  -I"$CATCH_BUILD/catch2cv-src/src"
  -I"$CATCH_BUILD/catch2cv-build/generated-includes"
)

# Test translation units + the real object sources under test.
mapfile -t TESTS < <(find "$ADDON/tests" -maxdepth 1 -name 'test_*.cpp' | sort)
mapfile -t SRCS  < <(find "$ADDON/CV" -name '*.cpp' | sort)

echo "== compiling ${#TESTS[@]} test TUs + ${#SRCS[@]} object TUs =="
# -fno-sanitize=shift-base: Catch2's own RNG/hasher trip these
# internally; they are not our code and would otherwise drown the real output.
g++ -std=c++23 -g -O1 -fsanitize=address,undefined \
  -fno-sanitize=shift-base -fno-omit-frame-pointer \
  "${INCLUDES[@]}" \
  "${TESTS[@]}" "${SRCS[@]}" \
  "$CATCH_BUILD/catch2cv-build/src/libCatch2Maind.a" \
  "$CATCH_BUILD/catch2cv-build/src/libCatch2d.a" \
  -o "$OUT"

echo "== running =="
exec "$OUT" "$@"
