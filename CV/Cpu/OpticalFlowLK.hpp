#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// UNITS OF THE THREE FLOW/TRACKING OBJECTS -- they do NOT agree, on purpose:
//   * cv::OpticalFlowLK (here): per-frame DISPLACEMENT, NORMALISED (u/W, v/H), on a grid
//     regenerated every frame. Nothing accumulates across frames.
//   * cv::PointTracker: normalised POSITIONS of an explicit point set, ACCUMULATED across
//     frames. A per-frame displacement is the caller's difference of two positions.
//   * cv::HornSchunck: dense per-pixel flow in PIXELS PER FRAME (not normalised), emitted as
//     a float32 texture pair.
// A normalised velocity is aspect-dependent: on a non-square frame (u/W, v/H) is not a
// rotation of the pixel displacement, and `magnitude` is not a pixel speed. That is
// deliberate -- it makes "Min magnitude" resolution-independent -- but it means this
// magnitude cannot be compared with HornSchunck's without multiplying back by W and H.
struct flow_vector
{
  halp::xy_type<float> position; // normalised, tracked point location
  halp::xy_type<float> velocity; // normalised displacement this frame
  float magnitude;               // = |velocity|, in the same normalised units
  // status: 1 = the 2x2 LK system was WELL-CONDITIONED AT FULL RESOLUTION (pyramid level 0),
  //             i.e. this grid point produced a real measurement;
  //         0 = lost: singular / ill-conditioned at level 0. This is OpenCV's rule and the
  //             one cv::PointTracker uses -- a coarse level that cannot be solved only
  //             forfeits its refinement, it does not lose the point.
  int status;
  // gated: 1 = the measurement was valid but its magnitude fell below "Min magnitude" and is
  //            reported as no motion (velocity and magnitude zeroed). status stays 1.
  //        0 = not gated (or `status == 0`, in which case nothing was measured to gate).
  // `status` and `gated` are SEPARATE because they mean different things: "there is nothing
  // to measure here" against "this point is standing still". They used to be conflated into
  // status = 0, which made every static well-conditioned point read as untrackable.
  int gated;
  // The grid is emitted in full: every interior grid point produces one flow_vector so the
  // caller knows the complete grid layout; lost and gated points carry zero velocity and
  // zero magnitude.

  halp_field_names(position, velocity, magnitude, status, gated);
};

// Sparse Lucas-Kanade optical flow on a regular grid of points (cv.jit.track / LKflow).
// Keeps the previous grayscale frame; for each grid point solves the 2x2 LK normal equations
// over a window to estimate displacement. Uses a coarse-to-fine image pyramid so that
// displacements larger than the window can be tracked: the flow is solved at the coarsest
// level and propagated to each finer level as a starting offset, then refined.
//
// The pyramid is built by halving each dimension INDEPENDENTLY (max(1, n/2)), so on an odd
// dimension the x and y scale ratios diverge; every mapping between levels therefore uses the
// actual per-axis ratio of the two levels involved, never a shared factor or a hardcoded x2.
// It is truncated as soon as a level would no longer be strictly larger than the (2*Window+1)
// search window, the same rule as OpenCV's buildOpticalFlowPyramid and cv::PointTracker.
//
// Outputs per-point velocity vectors with a per-point status flag. Stateful -> A.
struct OpticalFlowLK
{
  halp_meta(name, "Optical flow (LK)");
  halp_meta(c_name, "cv_optical_flow_lk");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Sparse Lucas-Kanade optical flow on a grid of points.");
  halp_meta(uuid, "4a7e9c12-3b58-4d0f-8e6a-1c2b9d0f7a85");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_i32<"Grid", halp::range{2, 64, 16}> grid;
    halp::hslider_i32<"Window", halp::range{2, 16, 5}> window;
    halp::hslider_f32<"Min magnitude", halp::range{0.f, 0.5f, 0.001f}> min_mag;
    halp::hslider_i32<"Pyramid levels", halp::range{1, 4, 3}> pyramid_levels;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Flow");
      std::vector<flow_vector> value;
    } flow;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_prev; // previous frame grayscale [0,1], size W*H
  int m_pw = 0, m_ph = 0;
};
}
