#pragma once

#include <CV/Cpu/Canny.hpp>
#include <CV/Cpu/HoughLines.hpp> // cv::line_segment

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace cv
{
// cv.jit.lines: Canny + PROBABILISTIC Hough transform, emitting line SEGMENTS.
//
// This is a single object on purpose -- cv.jit.lines is one object: it runs cv::Canny on the
// input and feeds the edge map straight to cv::HoughLinesP. The Canny stage is not
// duplicated here, it literally runs cv::Canny (CV/Cpu/Canny.hpp) as a sub-object, so the
// two objects can never drift apart.
//
// WHY PROBABILISTIC AND NOT "HoughTransform -> Extrema -> HoughLines": the standard
// transform only knows the (rho, theta) of an infinite line; it has no idea where the line
// starts and stops. HoughLinesP walks the edge map along each detected line, collects the
// connected run of edge pixels (tolerating up to `gap` missing ones), and reports the two
// endpoints -- so the output is a real segment. It also *removes* the pixels it consumed
// from the accumulator, so a single strong line is reported once instead of as a cluster of
// near-duplicate peaks.
//
// ATTRIBUTE MAPPING (cv.jit.lines defaults in brackets):
//   threshold  [150] float64, clipped [1,255] -> Canny band low = threshold - 10,
//                                                high = threshold + 10 (THRESHOLD_RANGE),
//                                                each clipped to [0,255].
//   resolution [1]   long, clipped [1,10]     -> BOTH rho = resolution pixels
//                                                AND theta = pi / (180 / resolution),
//                                                i.e. `resolution` degrees.
//   sensitivity[50]  long, clipped [1,255]    -> accumulator vote threshold.
//   gap        [2]   float64, floored at 0    -> maxLineGap.
//   length     [10]  float64, floored at 0    -> minLineLength.
//
// cv::HoughLinesP itself rounds `gap` and `length` to integers before use, and its
// "long enough" test is Chebyshev, not Euclidean:
//     good = |x2 - x1| >= length || |y2 - y1| >= length
// so a diagonal segment passes as soon as either projection is long enough. Reproduced.
//
// DELIBERATE DEVIATIONS from cv.jit / OpenCV:
//  - RNG: OpenCV shuffles the edge points with cv::RNG((uint64)-1), a fixed seed. We use a
//    std::mt19937 reseeded with a fixed constant at the start of every frame -- same
//    property (identical input -> identical output, run to run) via a portable generator.
//  - The trig table is double, not float: with a float table the walk direction of a
//    near-axis-aligned line can quantise differently. Endpoints are unaffected in practice.
//  - Accumulator writes are bounds-checked. OpenCV's are not; the margin is normally safe
//    but depends on the rounding of numrho.
//  - `Max lines`: cv::HoughLinesP is called by cv.jit with no cap, so a noisy frame can emit
//    an unbounded list. We cap it (default 4096, cv.jit.hough's own MAX_LINES).
//
// OUTPUT: `Lines` is std::vector<cv::line_segment> -- the SAME element type as
// CV/Cpu/HoughLines.hpp, so Lines and HoughLines are drop-in interchangeable for anything
// downstream. Coordinates are source-image pixels, exactly as cv.jit's 4-plane long matrix.
struct Lines
{
  halp_meta(name, "Line segments");
  halp_meta(c_name, "cv_lines");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Canny + probabilistic Hough transform: detects straight line segments and reports "
      "their endpoints in pixel coordinates.");
  halp_meta(uuid, "c1a70000-0045-4a00-9000-000000000002");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_t<double, "Threshold", halp::range{1., 255., 150.}> threshold;
    halp::hslider_i32<"Resolution", halp::range{1, 10, 1}> resolution;
    halp::hslider_i32<"Sensitivity", halp::range{1, 255, 50}> sensitivity;
    halp::hslider_t<double, "Gap", halp::range{0., 255., 2.}> gap;
    halp::hslider_t<double, "Length", halp::range{0., 4096., 10.}> length;
    halp::hslider_i32<"Max lines", halp::range{1, 4096, 4096}> maxlines;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Lines")
      std::vector<line_segment> value;
    } lines;
  } outputs;

  void operator()() noexcept;

private:
  // The Canny stage, run as-is -- not a copy of it.
  Canny m_canny;

  std::vector<int> m_accum;          // numangle * numrho
  std::vector<std::uint8_t> m_mask;  // W*H, 1 = edge pixel not yet consumed
  std::vector<std::int32_t> m_nz;    // packed y*W + x of the remaining edge pixels
  std::vector<double> m_trig;        // 2*numangle: [cos/rho, sin/rho] per angle
  std::mt19937 m_rng;
};
}
