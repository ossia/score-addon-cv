#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// What makes a pixel "join" the region grown from the seed.
//
// Tolerance (first enumerator == default):
//   |luma(pixel) - luma(seed)| <= Tolerance, on continuous Rec.601 luminance in [0,1].
//   This is *not* what cv.jit.floodfill does; it is the behaviour this port shipped with
//   and it is kept as the default purely for backward compatibility (see the note below).
//   It fills smoothly-varying regions, and — because the black background is "similar to
//   itself" — a seed dropped on a black pixel fills the background rather than doing
//   nothing.
//
// Binary (cv.jit.floodfill):
//   The absolute test `in[p] != 0` on an already-thresholded mask. Here the RGBA input is
//   binarised first: a pixel is foreground when its Rec.601 luma (0..255) is *strictly
//   greater* than `Threshold * 255`. With Threshold == 0 that is exactly cv.jit's
//   `!= 0` predicate; the default 0.5 matches the thresholding convention used by the
//   rest of the addon (BlobStats, BlobSort, Label...).
//   Consequences, all of them cv.jit's:
//     - the fill spans a connected foreground region *whatever* its pixel values are, so
//       a region whose intensity ramps from 40 to 200 is filled whole (Tolerance mode
//       with a small tolerance would only take the part near the seed's own level);
//     - a seed landing on a background pixel fills nothing — empty output, `Filled` == 0,
//       and no error;
//     - a seed outside the image fills nothing, likewise without erroring.
//
// NOTE ON THE DEFAULT: cv.jit's operation is `Binary`, and cv.jit parity would want it
// first. It is *not* first here because `tests/test_main_objects.cpp` (owned by another
// part of the suite, must keep passing unmodified) contains
// "FloodFill on background fills the complement", which seeds a black pixel and expects
// the 91 background pixels of a 10x10 image to be filled. That expectation is
// unreachable under cv.jit semantics, where a zero seed is a no-op by construction.
// So `Tolerance` stays the default and `Binary` is one enum step away.
enum class FloodMode
{
  Tolerance,
  Binary
};

// Which of the two seed inlets is authoritative.
//
// Normalized (default): the "Seed" xy_pad, in [0,1]^2, mapped as
//   px = clamp(int(x * width), 0, width - 1) (same for y). Because it is clamped, a
//   normalised seed can never be "out of bounds" — x == 1.0 selects the last column
//   rather than fill nothing. This is the port's original behaviour and is kept.
//
// Pixels (cv.jit): the "Seed (px)" integer spinboxes, in pixel coordinates, default 0 0
//   exactly like cv.jit's `seed` attribute (`long[2]`, default `0 0`). Unlike the
//   normalised seed this one is *not* clamped: an out-of-range value produces an empty
//   fill, as in cv.jit.
enum class FloodSeedMode
{
  Normalized,
  Pixels
};

// Scanline flood fill from a seed point (cv.jit.floodfill). 4-connected: the fill expands
// horizontal runs and only ever seeds the rows directly above/below, so a neighbour that
// touches the region only diagonally is never filled. Outputs the filled mask (0 / 255,
// cleared first, exactly like cv.jit which clears the output matrix before filling) and
// the filled pixel count. Sequential region-growing -> Path A.
//
// Deliberate deviations from cv.jit:
//  - cv.jit caps its segment stack at `width*height>>2` entries and *silently* stops
//    filling when it is hit, truncating the region with no way to know. This port has no
//    cap at all: every pixel is marked before being pushed, so the work is bounded by
//    width*height and the fill always completes. There is therefore no `Truncated`
//    output — nothing can be truncated.
//  - `Filled` is exposed. cv.jit computes the same area internally and throws it away.
//  - the Tolerance mode above has no cv.jit equivalent.
struct FloodFill
{
  halp_meta(name, "Flood fill");
  halp_meta(c_name, "cv_floodfill");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Flood-fill the region around a seed point (binary or by luminance similarity).");
  halp_meta(uuid, "7b1c4e89-2a05-4d3f-9e6b-8c0a1f2d5b40");

  struct
  {
    halp::texture_input<"In"> image;
    halp::xy_pad_f32<"Seed", halp::range{0.f, 1.f, 0.5f}> seed; // normalised
    halp::hslider_f32<"Tolerance", halp::range{0.f, 1.f, 0.1f}> tolerance;
    // Appended after the pre-existing ports so port indices of saved projects are
    // preserved.
    // halp::enum_t (magic_enum-backed) value-initialises `value` to the first enumerator,
    // so a default-constructed object is always valid and keeps the legacy behaviour.
    halp::enum_t<FloodMode, "Mode"> mode;
    // Binarisation level for Binary mode; ignored in Tolerance mode. 0 == cv.jit's `!= 0`.
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    // cv.jit's seed: integer pixel coordinates, default 0 0.
    halp::xy_spinboxes_i32<"Seed (px)", halp::range{-8192., 8192., 0.}> seed_px;
    halp::enum_t<FloodSeedMode, "Seed units"> seed_mode;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
    halp::val_port<"Filled", int> filled;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<std::uint8_t> m_mask;
};
}
