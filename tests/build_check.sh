#!/usr/bin/env bash
# Isolated build+run of ONE test translation unit plus an explicit list of object sources.
#
# Use this while developing (especially when several agents work in the tree at once):
# unlike run_standalone.sh it does NOT glob, so an unrelated half-written file elsewhere
# in CV/ cannot break your build.
#
# Usage:
#   tests/build_check.sh tests/test_edges.cpp CV/Cpu/Canny.cpp CV/Cpu/GaussianBlur.cpp
#   tests/build_check.sh tests/test_edges.cpp CV/Cpu/Canny.cpp -- "[canny]"
#
# Everything after a literal `--` is passed to the Catch2 binary.
set -euo pipefail

ADDON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORE=/home/jcelerier/ossia/score
CATCH_BUILD=/home/jcelerier/builds/build-asan-ubsan/_deps

SRCS=()
CATCH_ARGS=()
seen_sep=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then seen_sep=1; continue; fi
  if [[ $seen_sep -eq 1 ]]; then CATCH_ARGS+=("$a"); else SRCS+=("$ADDON/$a"); fi
done

if [[ ${#SRCS[@]} -eq 0 ]]; then
  echo "usage: tests/build_check.sh <test.cpp> [object.cpp ...] [-- <catch args>]" >&2
  exit 2
fi

OUT="${TMPDIR:-/tmp}/cv_check_$(basename "${SRCS[0]}" .cpp)_$$"

g++ -std=c++23 -g -O1 -fsanitize=address,undefined \
  -fno-sanitize=shift-base -fno-omit-frame-pointer \
  -I"$ADDON" \
  -I"$SCORE/3rdparty/avendish/include" \
  -I"$SCORE/3rdparty/eigen" \
  -I"$SCORE/3rdparty/libossia/3rdparty/magic_enum/include/magic_enum" \
  -I"$CATCH_BUILD/catch2cv-src/src" \
  -I"$CATCH_BUILD/catch2cv-build/generated-includes" \
  "${SRCS[@]}" \
  "$CATCH_BUILD/catch2cv-build/src/libCatch2Maind.a" \
  "$CATCH_BUILD/catch2cv-build/src/libCatch2d.a" \
  -o "$OUT"

"$OUT" "${CATCH_ARGS[@]+"${CATCH_ARGS[@]}"}"
rc=$?
rm -f "$OUT"
exit $rc
