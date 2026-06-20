#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>

namespace cv
{
// Tracking mode (cv.jit.shift mode 0/1). CamShift first so it is the default enumerator.
enum class TrackMode
{
  CamShift,
  MeanShift
};

// Colour-window tracker (cv.jit.shift — MeanShift / CamShift). OpenCV-free.
//
// On a rising edge of `Set`, grabs a square ROI around the seed point and builds a HUE
// histogram (gated by saturation/value) of that region from the current frame: this is the
// target colour model. Each subsequent frame the image is histogram-backprojected (each pixel
// scored by its hue-bin probability) and mean-shift iterations recentre the search window onto
// the centroid of the backprojection weights.
//
// Two modes (cv.jit.shift mode 0/1):
//   - MeanShift: fixed-size window, only recentre. Window size never changes and the
//     reported angle is forced to 0 (no orientation estimate).
//   - CamShift: after recentring, adapt the window size from the zeroth moment and the
//     orientation from the second moments (current adaptive behaviour).
//
// The `mass` output is the zeroth moment (total backprojection weight) inside the converged
// window, normalised by the window area to [0,1]: high when the tracked colour fills the
// window (confident track), ~0 when the target is absent. Use it to gate on confidence.
//
// Stateful -> the model histogram and the current window live in members.
struct CamShift
{
  halp_meta(name, "CamShift");
  halp_meta(c_name, "cv_camshift");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Track a coloured object by hue histogram backprojection (CamShift).");
  halp_meta(uuid, "73433fda-d319-4a5d-924d-884b97debab5");

  static constexpr int Bins = 32;

  struct
  {
    halp::texture_input<"In"> image;
    halp::xy_pad_f32<"Seed", halp::range{0.f, 1.f, 0.5f}> seed;
    halp::hslider_f32<"Init size", halp::range{0.01f, 0.5f, 0.1f}> initSize;
    halp::hslider_i32<"Iterations", halp::range{1, 32, 10}> iterations;
    halp::hslider_f32<"Min saturation", halp::range{0.f, 1.f, 0.2f}> minSat;
    halp::hslider_f32<"Min value", halp::range{0.f, 1.f, 0.2f}> minVal;
    // Tracking mode (cv.jit.shift): MeanShift = fixed window, recentre only (angle 0);
    // CamShift = adapt window size + orientation from image moments.
    // halp::enum_t (magic_enum-backed) value-initializes its `value` to the first enumerator,
    // so a default-constructed object is always valid (no UBSAN invalid-enum load). CamShift
    // is listed first so it remains the default.
    halp::enum_t<TrackMode, "Mode"> mode;
    halp::toggle<"Set"> set;
  } inputs;

  struct
  {
    halp::val_port<"Center", halp::xy_type<float>> center;
    halp::val_port<"Size", halp::xy_type<float>> size;
    halp::val_port<"Angle", float> angle;
    // Zeroth moment / window area, in [0,1]: tracking-confidence (component "mass").
    halp::val_port<"Mass", float> mass;
    halp::val_port<"Tracking", bool> tracking;
  } outputs;

  void operator()() noexcept;

private:
  std::array<float, Bins> m_hist{}; // normalised hue histogram of the target model
  bool m_haveModel = false;

  // Current search window in pixel coordinates (top-left x,y + size).
  float m_wx = 0.f, m_wy = 0.f, m_ww = 0.f, m_wh = 0.f;

  bool m_prevSet = false;
};
}
