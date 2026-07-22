#pragma once

#include <CV/Support/Brief.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <vector>

namespace cv
{
// One accepted correspondence. `prev` refers to the FIRST set, `cur` to the SECOND:
//  - two-set mode: prev = a keypoint of "Keypoints A", cur = the matched one of "Keypoints B";
//  - texture mode: prev = the previous frame (or the stored reference), cur = this frame.
//
// cv.jit.keypoints.match emits the complete 6-plane keypoint row for BOTH sides (two output
// matrices), not just the coordinates, so the full payload of each side is carried through
// here as flat fields rather than only the positions.
struct feature_match
{
  halp::xy_type<float> prev; // normalised position, first set
  halp::xy_type<float> cur;  // normalised position, second set
  float distance;            // Hamming distance of the matched descriptors

  // Remaining keypoint planes of each side (cv.jit: size, angle, response, octave).
  float prev_size{};
  float prev_angle{}; // radians
  float prev_response{};
  int prev_octave{};
  float cur_size{};
  float cur_angle{}; // radians
  float cur_response{};
  int cur_octave{};

  halp_field_names(
      prev, cur, distance, prev_size, prev_angle, prev_response, prev_octave, cur_size,
      cur_angle, cur_response, cur_octave);
};

// ORB feature matcher. Port of cv.jit.keypoints.match, OpenCV-free. Two modes:
//
//  * TWO-SET mode (cv.jit's actual signature: 4 matrix inlets = two independently produced
//    keypoint+descriptor sets, i.e. two different images — the shipped help patch matches a
//    template image against a live camera). Active as soon as EITHER keypoint list input is
//    connected/non-empty. No detection happens in this mode: the texture inlet is ignored
//    entirely, and the object is a pure matcher fed by e.g. two OrbFeatures instances.
//
//  * TEXTURE mode (the historical behaviour of this port, kept so existing patches work).
//    Used only when BOTH list inputs are empty: each frame it detects keypoints itself and
//    matches them against the previous frame, or against a snapshot taken on a rising edge of
//    "Set reference". STATEFUL. Single-scale — the Octaves control belongs to OrbFeatures;
//    for a multi-scale pipeline use two OrbFeatures into the list inputs.
//
// cv.jit tolerances reproduced (they occur constantly while patching, and cv.jit deliberately
// stays silent rather than erroring): an empty set on either side, or keypoint/descriptor
// counts that disagree, emit an EMPTY result with no error. Here keypoints and descriptors
// travel together in one `cv::keypoint`, so a count mismatch is unrepresentable by
// construction; the empty-set case is handled explicitly.
//
// cv.jit bug NOT reproduced: it dereferences knnMatch's `match[1]` without checking the
// result size, so a train set with fewer than 2 descriptors reads out of bounds. See
// cv_support::matchRatioBy, which returns empty instead.
//
// Matching is brute-force Hamming rather than cv.jit's FLANN: FLANN is an approximate
// nearest-neighbour index designed for float descriptors, while these are binary, so
// exhaustive popcount is both exact and fast at these feature counts.
struct FeatureMatch
{
  halp_meta(name, "Feature match");
  halp_meta(c_name, "cv_feature_match");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "ORB keypoint matcher (brute-force Hamming + Lowe's ratio test): matches two "
      "externally supplied keypoint sets, or falls back to temporal/reference matching on "
      "its texture input.");
  halp_meta(uuid, "873bfa0e-42f3-49c0-a90e-fd8050bab089");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 0.5f, 0.08f}> threshold;
    halp::hslider_i32<"Max features", halp::range{16, 4096, 512}> max_features;
    // Lowe's ratio. cv.jit hard-codes 0.7 (LOWES_RATIO in cv.jit.keypoints.match.cpp); this
    // port defaults to the same value for parity, but exposes it since the useful setting is
    // scene-dependent. Higher = more permissive (more matches, more outliers).
    halp::hslider_f32<"Ratio", halp::range{0.1f, 1.f, 0.7f}> ratio;
    // Rising edge: snapshot the CURRENT frame's keypoints+descriptors as a fixed reference.
    // Once set, every subsequent frame matches against this reference instead of the previous
    // frame. TEXTURE MODE ONLY — irrelevant when the list inputs are used.
    halp::toggle<"Set reference"> set_reference;

    // The two independently produced sets. Wire two OrbFeatures here (template vs live) for
    // the cv.jit.keypoints.match use case.
    struct
    {
      halp_meta(name, "Keypoints A");
      std::vector<keypoint> value;
    } set_a;
    struct
    {
      halp_meta(name, "Keypoints B");
      std::vector<keypoint> value;
    } set_b;
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
