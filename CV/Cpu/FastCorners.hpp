#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
struct corner_point
{
  halp::xy_type<float> position; // normalised
  float score;

  halp_field_names(position, score);
};

// FAST corner detector (FAST-9, cv.jit.features). Detects corners via the segment test on a
// Bresenham circle of radius 3, with non-maximum suppression. OpenCV-free. The output drives
// interaction (corner clouds, tracking seeds). For descriptors + matching see FeatureMatch.
struct FastCorners
{
  halp_meta(name, "FAST corners");
  halp_meta(c_name, "cv_fast_corners");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "FAST-9 corner detector with non-maximum suppression.");
  halp_meta(uuid, "2c8b0e47-5d39-4a16-9f7c-3e1a0b6d2f98");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 0.5f, 0.08f}> threshold;
    halp::toggle<"Suppress"> suppress;
    halp::hslider_i32<"Max corners", halp::range{16, 8192, 1024}> max_corners;
    // Minimum spacing between accepted corners, normalised to image size (0..0.2).
    // 0 disables spacing: behaviour is then just "strongest N". Like goodFeaturesToTrack's
    // minDistance, it greedily de-clusters corners after sorting by strength.
    halp::hslider_f32<"Min distance", halp::range{0.f, 0.2f, 0.f}> min_distance;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Corners");
      std::vector<corner_point> value;
    } corners;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_gray;
  std::vector<float> m_scoremap;
};
}
