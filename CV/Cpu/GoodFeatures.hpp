#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// A detected feature. This is the SHARED element type between cv::GoodFeatures (producer)
// and cv::PointTracker (consumer, via its "Seeds" list input), so
//   GoodFeatures.Features  ->  PointTracker.Seeds
// chains directly with no adapter object. That chain is the cv.jit.features2track
// abstraction.
//
// COORDINATES: normalised to [0,1] (x / width, y / height), which is the port convention of
// this addon. cv.jit.features emits raw *pixel* coordinates in a 2-plane float32 matrix.
struct feature_point
{
  float x{};
  float y{};
  // Shi-Tomasi response (minimum eigenvalue of the windowed structure tensor). Relative
  // units: only ratios between features of the same frame are meaningful, which is exactly
  // what `Quality` thresholds on.
  float score{};

  halp_field_names(x, y, score);
};

// Shi-Tomasi "good features to track" corner detector (cv.jit.features).
//
// This is `cv::goodFeaturesToTrack(..., useHarrisDetector=false, blockSize=3)`, i.e. the
// MINIMUM EIGENVALUE of the 3x3-windowed structure tensor
//
//     M = [ sum Ix^2   sum IxIy ]      response = lambda_min(M)
//         [ sum IxIy   sum Iy^2 ]                = ((a+c) - sqrt((a-c)^2 + 4b^2)) / 2
//
// with Ix, Iy from a 3x3 Sobel. It is NOT the Harris response (no `det - k*trace^2`, the
// `k = 0.04` OpenCV parameter is unused when useHarrisDetector is false) and it is NOT the
// FAST segment test used by cv::FastCorners -- FastCorners thresholds an ABSOLUTE
// brightness difference, this object thresholds a RELATIVE quality level. Both objects
// exist on purpose; they answer different questions.
//
// Parameters follow cv.jit.features:
//  - `Quality`  = OpenCV's qualityLevel (cv.jit's `threshold`, float64, default 0.1,
//                 clipped to [0.001, 1]). A corner survives when
//                     response >= Quality * (strongest response in the searched region).
//                 With a ROI the "strongest response" is the strongest INSIDE the ROI,
//                 matching OpenCV's `minMaxLoc(eig, ..., mask)`.
//  - `Distance` = OpenCV's minDistance, in PIXELS (cv.jit default 5, forced >= 1).
//                 Greedy strongest-first spacing: a candidate is dropped when it is closer
//                 than `Distance` to an already-accepted (stronger) feature.
//  - `Max features` = cv.jit's hard-coded MAX_FEATURES cap, exposed (default 2048).
//  - `Use ROI` + `ROI x1/y1/x2/y2` = cv.jit's `useroi` / `roi` mask, in pixels. Detection is
//                 restricted to the rectangle; the corners are the same, only the searched
//                 region (and hence the reference maximum) changes.
//  - `Precision` = cv.jit's `precision`, i.e. `cv::cornerSubPix` refinement with
//                 winSize (3,3) -- OpenCV counts winSize as a HALF size, so the actual
//                 neighbourhood is 7x7 -- zeroZone (-1,-1), TermCriteria(MAX_ITER, 10, 0.1).
//
// Output is sorted STRONGEST FIRST.
//
// DELIBERATE DEVIATIONS from cv.jit / OpenCV, all tested:
//  - The quality test is `>=` (as specified for this port), where OpenCV uses
//    `cv::threshold(..., THRESH_TOZERO)` which is a strict `>`. OpenCV's strict form makes
//    `qualityLevel = 1.0` return NOTHING AT ALL (the maximum is not greater than itself);
//    here it returns the strongest corner (and its ties), which is what the control implies.
//  - cv.jit clips `roi[0]`/`roi[2]` to [0, width-1] but `roi[1]`/`roi[3]` to [0, height] --
//    an off-by-one asymmetry. We clip both axes to the image extent.
//  - No visualisation texture output (cv.jit.features has none either); use the feature list.
struct GoodFeatures
{
  halp_meta(name, "Good features (Shi-Tomasi)");
  halp_meta(c_name, "cv_good_features");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Shi-Tomasi minimum-eigenvalue corner detector (goodFeaturesToTrack): relative "
      "quality level, minimum pixel distance, optional ROI and sub-pixel refinement.");
  halp_meta(uuid, "c1a70000-0023-4a00-9000-000000000023");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Quality", halp::range{0.001f, 1.f, 0.1f}> quality;
    halp::hslider_f32<"Distance", halp::range{1.f, 256.f, 5.f}> distance;
    halp::hslider_i32<"Max features", halp::range{1, 8192, 2048}> max_features;
    halp::toggle<"Precision"> precision;
    halp::toggle<"Use ROI"> useroi;
    halp::hslider_i32<"ROI x1", halp::range{0, 8192, 0}> roi_x1;
    halp::hslider_i32<"ROI y1", halp::range{0, 8192, 0}> roi_y1;
    halp::hslider_i32<"ROI x2", halp::range{0, 8192, 0}> roi_x2;
    halp::hslider_i32<"ROI y2", halp::range{0, 8192, 0}> roi_y2;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Features")
      std::vector<feature_point> value;
    } features;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_gray; // W*H, [0,1]
  std::vector<float> m_gx;   // W*H, Sobel dI/dx
  std::vector<float> m_gy;   // W*H, Sobel dI/dy
  std::vector<float> m_resp; // W*H, lambda_min of the 3x3-summed structure tensor
};
}
