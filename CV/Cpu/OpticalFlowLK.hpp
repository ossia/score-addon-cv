#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
struct flow_vector
{
  halp::xy_type<float> position; // normalised, tracked point location
  halp::xy_type<float> velocity; // normalised displacement this frame
  float magnitude;
  // status: 1 = tracked (the 2x2 LK system was well-conditioned and motion >= min_mag),
  //         0 = lost/ill-conditioned (singular system, or gated below min_mag).
  // The grid is emitted in full: every interior grid point produces one flow_vector so the
  // caller knows the complete grid layout; lost points carry zero velocity/magnitude.
  int status;

  halp_field_names(position, velocity, magnitude, status);
};

// Sparse Lucas-Kanade optical flow on a regular grid of points (cv.jit.track / LKflow).
// Keeps the previous grayscale frame; for each grid point solves the 2x2 LK normal equations
// over a window to estimate displacement. Uses a coarse-to-fine image pyramid so that
// displacements larger than the window can be tracked: the flow is solved at the coarsest
// level and propagated (x2) to each finer level as a starting offset, then refined.
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
