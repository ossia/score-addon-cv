#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <vector>

namespace cv
{
// Camera calibration via Zhang's method (cv.jit.calibration), OpenCV-free, Eigen.
//
// STATEFUL & SELF-CONTAINED: it runs the shared chessboard detector internally on
// each frame. A `capture` toggle grabs the currently-detected ordered corner set
// into an internal list of views. A `solve` toggle then runs Zhang's method over
// all captured views:
//   per-view homography (Hartley-normalised DLT, Eigen SVD) -> build the linear
//   system for the image of the absolute conic (b vector) -> recover intrinsics K
//   (fx,fy,cx,cy AND a solved-for skew K(0,1)) -> per-view extrinsics (R,t) ->
//   radial distortion (k1,k2) by linear least squares -> RMS reprojection error.
//
// SCOPE / LIMITATIONS (documented):
//   * LINEAR Zhang ONLY — there is NO nonlinear (Levenberg-Marquardt) bundle
//     refinement of the intrinsics/extrinsics; results are the closed-form estimate.
//   * Distortion model is RADIAL k1,k2 ONLY — no tangential (p1,p2) terms.
//   * Skew is solved for (not forced to 0); it lands near zero on good planar data.
//   * Requires a MINIMUM of 3 captured views (from differing board orientations);
//     `solve` is a no-op with fewer.
//
// Object-space model: planar board, inner corners on a unit grid (col, row, 0),
// i.e. square size = 1 unit. K is recovered in pixel units relative to that grid;
// cx,cy,fx,fy are reported in pixels (see outputs).
struct Calibration
{
  halp_meta(name, "Camera calibration");
  halp_meta(c_name, "cv_calibration");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Zhang camera calibration from chessboard views. Toggle Capture to grab the "
      "current detected board, Solve to compute intrinsics K (fx,fy,cx,cy), radial "
      "distortion (k1,k2) and RMS reprojection error. Self-contained: detects the "
      "board internally. Needs >= 3 captured views from differing orientations.");
  halp_meta(uuid, "d0e3b1ad-0542-4564-8314-b5c02d5adb40");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_i32<"Cols", halp::range{2, 32, 7}> cols;
    halp::hslider_i32<"Rows", halp::range{2, 32, 7}> rows;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::toggle<"Capture"> capture;
    halp::toggle<"Solve"> solve;
    halp::toggle<"Reset"> reset;
  } inputs;

  struct
  {
    // K, row-major 3x3, in pixel units (fx,0,cx, 0,fy,cy, 0,0,1).
    halp::val_port<"K", std::array<float, 9>> K;
    halp::val_port<"Focal", halp::xy_type<float>> focal;  // fx, fy (pixels)
    halp::val_port<"Center", halp::xy_type<float>> center; // cx, cy (pixels)
    halp::val_port<"Distortion", halp::xy_type<float>> distortion; // k1, k2
    halp::val_port<"RMS", float> rms;
    halp::val_port<"Views", int> views;
    halp::val_port<"Solved", bool> solved;
  } outputs;

  void operator()() noexcept;

private:
  // Each captured view: image corners (pixels) in row-major grid order.
  struct View
  {
    std::vector<std::array<float, 2>> img; // detected corners, pixels
  };
  std::vector<View> m_views;
  int m_cols = 7, m_rows = 7;
  int m_imgW = 0, m_imgH = 0;
};
}
