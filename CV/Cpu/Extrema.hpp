#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <vector>

namespace cv
{
// One local maximum of a Hough accumulator, in raw accumulator bin indices.
//
// The accumulator produced by CV/Shaders/Analysis/Hough.cs is laid out as
// accum[theta * RHO_BINS + rho], i.e. when read back as an image the COLUMN is the rho
// bin and the ROW is the theta bin. cv.jit.extrema outputs (j, i) = (column, row) in that
// order, so `x` is the rho bin and `y` is the theta bin. cv::HoughLines consumes exactly
// this.
struct hough_peak
{
  int x{};       // column index = rho bin
  int y{};       // row index    = theta bin
  float value{}; // accumulator count at that bin (texture luminance, 0..255)

  halp_field_names(x, y, value);
};

// Neighbourhood used by the local-maximum test.
//
// NOTE ON cv.jit COMPATIBILITY: cv.jit.extrema's `mode` attribute is documented in the
// source as "0 = 4 connectivity, 1 = 8 connectivity" but the switch does the opposite --
// case 0 runs the eight-neighbour test and case 1 the four-neighbour one. We follow the
// CODE, not the comment: Neighbours8 is first so it maps to mode 0 (the cv.jit default).
enum class ExtremaNeighbourhood
{
  Neighbours8, // mode 0: strictly greater than all 8 surrounding cells
  Neighbours4  // mode 1: strictly greater than E / W / N / S only
};

// Local-maximum picker over a Hough accumulator (cv.jit.extrema).
//
// Takes the accumulator as a texture (read as luminance) and emits the bins that are
// strictly greater than `Threshold` AND strictly greater than each tested neighbour.
//
// FAITHFUL TO cv.jit:
//  - the border row/column is never tested (scan is rows [1, h-2] x cols [1, w-2]);
//  - the comparison is a strict `>`, so a plateau of equal cells yields NO peak;
//  - results come out in raster order and the scan ABORTS the moment `Max points` is
//    reached, so they are first-found, not strongest-first. There is no sorting and no
//    sub-pixel refinement.
//
// DELIBERATE DEVIATION (bug fix): cv.jit.extrema computes its vertical neighbour offsets
// from `in_minfo.dim[1]` (the matrix HEIGHT) instead of the row stride. That only
// coincides with the intended offsets when width == height, and otherwise samples wrong
// cells and can read past the end of the matrix. We use the real row stride, which also
// makes mode 0 a true 8-neighbour test and mode 1 a true E/W/N/S test (with the buggy
// offsets, mode 1 actually compared against E/W/NW/SE).
struct Extrema
{
  halp_meta(name, "Extrema");
  halp_meta(c_name, "cv_extrema");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Local maxima of a Hough accumulator: emits (rho bin, theta bin, count) peaks. "
      "Feed the output to Hough lines.");
  halp_meta(uuid, "c1a70000-0005-4a00-9000-000000000005");

  struct
  {
    halp::texture_input<"In"> image;
    // cv.jit ranges: threshold clamped to [0, 4096], maxpoints to [1, 4096].
    //
    // `Threshold` is [0, 255], NOT cv.jit's [0, 4096]: the accumulator reaches this object
    // as an RGBA8 texture whatever produced it, so the decoded count can never exceed 255
    // and every setting above that is dead travel on the slider. (cv.jit.extrema reads a
    // char matrix, so its own 4096 was already unreachable.) cv::HoughTransform reports the
    // true unclamped peak on its `Max count` outlet if you need to know that the
    // accumulator saturated.
    halp::hslider_i32<"Threshold", halp::range{0, 255, 20}> threshold;
    halp::hslider_i32<"Max points", halp::range{1, 4096, 64}> maxpoints;
    halp::enum_t<ExtremaNeighbourhood, "Mode"> mode;

    // ---- Pass-through geometry (not used by the peak search) -------------------------
    // cv::HoughLines needs the accumulator's bin steps and the source frame size to turn a
    // bin index back into a real (rho, theta). cv::HoughTransform publishes all four; this
    // object sits between the two, so it forwards them unchanged and a patch can wire
    // HoughTransform -> Extrema -> HoughLines as one straight run instead of fanning the
    // scalars around the peak-picker. 0 means "unset": the corresponding output stays 0
    // and cv::HoughLines falls back to deriving the value from the accumulator size.
    halp::hslider_t<double, "Theta step", halp::range{0., 3.15, 0.}> theta_step;
    halp::hslider_t<double, "Rho step", halp::range{0., 65536., 0.}> rho_step;
    halp::hslider_i32<"Src width", halp::range{0, 8192, 0}> src_width;
    halp::hslider_i32<"Src height", halp::range{0, 8192, 0}> src_height;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Peaks")
      std::vector<hough_peak> value;
    } peaks;
    // Accumulator dimensions, so a downstream Hough lines object can be wired up
    // without the user re-typing them.
    halp::val_port<"Acc width", int> acc_width;
    halp::val_port<"Acc height", int> acc_height;
    // Forwarded verbatim from the inputs of the same name (see above).
    halp::val_port<"Theta step", double> theta_step;
    halp::val_port<"Rho step", double> rho_step;
    halp::val_port<"Src width", int> src_width;
    halp::val_port<"Src height", int> src_height;
  } outputs;

  void operator()() noexcept;

private:
  // Scratch: the accumulator decoded to integer counts, row-major, stride == width.
  std::vector<int> m_acc;
};
}
