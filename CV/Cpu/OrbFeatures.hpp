#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace cv
{
struct keypoint
{
  halp::xy_type<float> position; // normalised [0,1]
  float angle;                   // radians (intensity-centroid orientation)

  // 256-bit rotated-BRIEF descriptor, packed as 8 x 32-bit words.
  std::array<std::uint32_t, 8> desc;

  halp_field_names(position, angle, desc);
};

// Oriented FAST + rotated BRIEF (ORB) keypoint detector/descriptor. Port of
// cv.jit.keypoints, OpenCV-free. Detects FAST-9 corners, computes an intensity-centroid
// orientation per corner, then a 256-bit steered-BRIEF descriptor. Stateless per frame.
struct OrbFeatures
{
  halp_meta(name, "ORB features");
  halp_meta(c_name, "cv_orb_features");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Oriented FAST + rotated BRIEF keypoint detector and 256-bit descriptors.");
  halp_meta(uuid, "e963f6b1-808f-469d-8ffe-ad1c3c035e4c");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 0.5f, 0.08f}> threshold;
    halp::hslider_i32<"Max features", halp::range{16, 4096, 512}> max_features;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Keypoints");
      std::vector<keypoint> value;
    } keypoints;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_gray;
};
}
