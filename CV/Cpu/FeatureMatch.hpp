#pragma once

#include <CV/Support/Brief.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <vector>

namespace cv
{
struct feature_match
{
  halp::xy_type<float> prev; // normalised position in the previous frame
  halp::xy_type<float> cur;  // normalised position in the current frame
  float distance;            // Hamming distance of the matched descriptors

  halp_field_names(prev, cur, distance);
};

// Temporal ORB feature matcher. Port of cv.jit.keypoints.match, OpenCV-free and
// self-contained: each frame it detects oriented-FAST + rotated-BRIEF keypoints, then
// brute-force matches them against the previous frame's keypoints by Hamming distance with
// Lowe's ratio test. STATEFUL: keeps the previous frame internally. Outputs matched pairs.
struct FeatureMatch
{
  halp_meta(name, "Feature match");
  halp_meta(c_name, "cv_feature_match");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Temporal ORB keypoint matcher (brute-force Hamming + Lowe's ratio test).");
  halp_meta(uuid, "873bfa0e-42f3-49c0-a90e-fd8050bab089");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 0.5f, 0.08f}> threshold;
    halp::hslider_i32<"Max features", halp::range{16, 4096, 512}> max_features;
    halp::hslider_f32<"Ratio", halp::range{0.1f, 1.f, 0.8f}> ratio;
    // Rising edge: snapshot the CURRENT frame's keypoints+descriptors as a fixed reference.
    // Once set, every subsequent frame matches against this reference (template/object
    // recognition, the primary cv.jit.keypoints.match use case) instead of the previous frame.
    halp::toggle<"Set reference"> set_reference;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Matches");
      std::vector<feature_match> value;
    } matches;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_gray;
  std::vector<cv_support::kp> m_prev; // previous frame's keypoints (pixel space)
  int m_prevW{};
  int m_prevH{};

  // Stored reference set (template) captured on a rising edge of set_reference. When
  // non-empty, frames match against this instead of the previous frame.
  std::vector<cv_support::kp> m_ref;
  int m_refW{};
  int m_refH{};
  bool m_hasRef{};
  bool m_prevSetRef{}; // rising-edge tracking for set_reference
};
}
