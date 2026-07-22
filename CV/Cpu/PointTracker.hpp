#pragma once

#include <CV/Cpu/GoodFeatures.hpp> // cv::feature_point -- the shared seed element type

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// UNITS OF THE THREE FLOW/TRACKING OBJECTS -- they do NOT agree, on purpose:
//   * cv::PointTracker (here): POSITIONS, normalised ([0,1] = px/W, py/H), ACCUMULATED
//     across frames. There is no velocity output; a per-frame displacement is the caller's
//     difference of two consecutive positions.
//   * cv::OpticalFlowLK: per-frame DISPLACEMENT, normalised (u/W, v/H), on a grid that is
//     regenerated every frame. Nothing accumulates.
//   * cv::HornSchunck: dense per-pixel flow in PIXELS PER FRAME (not normalised), as a
//     float32 texture pair.
// A normalised velocity is therefore aspect-dependent: (u/W, v/H) is not a rotation of a
// pixel displacement on a non-square frame, and a magnitude computed from it is not a pixel
// speed. This is deliberate -- it makes thresholds resolution-independent -- but it means
// OpticalFlowLK's magnitude must not be compared against HornSchunck's without scaling.
//
// One track. `x`, `y` are NORMALISED ([0,1] = x/width, y/height), the port convention of
// this addon; cv.jit.track emits raw PIXEL coordinates in a 3-plane float32 matrix
// (x, y, status) of dim npoints.
struct tracked_point
{
  float x{};
  float y{};
  // 1 = tracked this frame, 0 = lost -- the window ran out of the frame, or the structure
  // tensor AT THE FINEST PYRAMID LEVEL was singular / below "Min eigenvalue" (OpenCV's rule:
  // a coarse level that cannot be refined only forfeits its refinement, it does not lose the
  // point). A lost point KEEPS its last coordinates.
  int status{};
  // Mean absolute photometric residual between the previous and current windows after
  // convergence, in [0,1] grey units. This is OpenCV's `err` output, which cv.jit.track
  // requests and then throws away. 0 for a lost point (as in OpenCV).
  float error{};

  halp_field_names(x, y, status, error);
};

// Pyramidal Lucas-Kanade tracking of an explicit, user-supplied set of points
// (cv.jit.track = cv::calcOpticalFlowPyrLK over a persistent point set).
//
// WHAT MAKES THIS A TRACKER, and not cv::OpticalFlowLK:
//   * the point set is EXPLICIT (seeded by the "Seeds" list input or the `set` message),
//     not a regular grid regenerated every frame;
//   * IDENTITY IS POSITIONAL: index i is always the same track, for as long as the object
//     lives;
//   * the tracked positions BECOME the next frame's starting positions (cv.jit does
//     `std::swap(previous_points, points)` at the end of every matrix_calc), so positions
//     ACCUMULATE across frames instead of being a per-frame displacement.
//
// Seeding:
//   * "Seeds" (list input, `std::vector<cv::feature_point>`): applied whenever the list
//     CHANGES. While it stays the same the tracks keep running -- re-applying it every
//     frame would destroy the persistence that is the whole point of the object. Plugging
//     cv::GoodFeatures straight into it gives the cv.jit.features2track chain.
//     When the seed list is non-empty its size determines the number of tracks; otherwise
//     the "Points" control does (cv.jit's `npoints` attribute, default 1).
//   * `set <index> <x> <y>` message: cv.jit.track's `set`, re-seeding ONE track without
//     disturbing the others. Coordinates are normalised here (pixels in cv.jit).
//   * `reset` message: drop all tracks and the reference frame.
//
// LOST POINTS ARE RETRIED. A point whose window leaves the frame is reported with
// status = 0 and its last coordinates are left untouched; because those coordinates are
// also the next frame's starting guess, the point is retried from there and can be
// recovered if the motion brings the pattern back. This is documented cv.jit behaviour and
// is reproduced deliberately -- a textbook tracker would delete the point.
//
// Defaults are the OpenCV defaults cv.jit inherits: window (2*radius+1)^2 with radius 7
// (15x15), maxLevel 3 (4 pyramid levels), TermCriteria(COUNT+EPS, 30, 0.01),
// minEigThreshold 1e-4.
//
// DELIBERATE DEVIATIONS from cv.jit, all tested:
//  - cv.jit.track has a FIRST-FRAME BUG: with no previous image it skips the flow call,
//    outputs the (uninitialised, all-zero) `points` vector, and then swaps it into
//    `previous_points` -- silently wiping any point seeded with `set` before the first
//    frame arrived. Here the first frame emits the seeded positions unchanged (no motion)
//    and does not clobber them.
//  - `set` with an index beyond the current track count GROWS the track set instead of
//    erroring out, so seeding does not depend on setting "Points" first.
//  - the LK maths runs on [0,1] grey with central differences, where OpenCV uses 8-bit grey
//    with Scharr derivatives. The reported minimum eigenvalue is converted back into
//    OpenCV's units ((32*255)^2 / 2^20 = 63.5, the Scharr gain against OpenCV's own
//    FLT_SCALE rescaling -- see PointTracker.cpp) so that the documented default
//    minEigThreshold of 1e-4 keeps its OpenCV meaning -- namely "reject only a window with
//    essentially no texture at all". Without that conversion 1e-4 would be ~4 orders of
//    magnitude too strict in eigenvalue units and would drop perfectly trackable points.
//  - cv.jit's `maxiter` attribute (default 20) is unused there -- it is never passed to
//    OpenCV, which keeps its own 30 -- so "Max iterations" here defaults to the effective
//    OpenCV value of 30, not to 20.
struct PointTracker
{
  halp_meta(name, "Point tracker (LK)");
  halp_meta(c_name, "cv_point_tracker");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Pyramidal Lucas-Kanade tracking of an explicit set of points, carried across "
      "frames with stable per-index identity.");
  halp_meta(uuid, "c1a70000-0022-4a00-9000-000000000022");

  struct
  {
    halp::texture_input<"In"> image;
    struct
    {
      halp_meta(name, "Seeds")
      std::vector<feature_point> value;
    } seeds;
    halp::hslider_i32<"Points", halp::range{1, 2048, 1}> npoints;
    halp::hslider_i32<"Radius", halp::range{1, 32, 7}> radius;
    halp::hslider_i32<"Max level", halp::range{0, 5, 3}> max_level;
    halp::hslider_i32<"Max iterations", halp::range{1, 100, 30}> max_iter;
    halp::hslider_f32<"Epsilon", halp::range{0.f, 1.f, 0.01f}> epsilon;
    halp::hslider_f32<"Min eigenvalue", halp::range{0.f, 0.01f, 0.0001f}> min_eig;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Points")
      std::vector<tracked_point> value;
    } points;
  } outputs;

  // cv.jit.track's `set <index> <x> <y>` (normalised coordinates here).
  void set(int index, float x, float y);
  // Drop every track and the reference frame.
  void reset();

  halp_start_messages(PointTracker)
    halp_mem_fun(set)
    halp_mem_fun(reset)
  halp_end_messages

  void operator()() noexcept;

private:
  std::vector<tracked_point> m_pts;        // persistent tracks (cv.jit's previous_points)
  std::vector<feature_point> m_last_seeds; // last applied seed list
  std::vector<float> m_prev;               // previous frame, grey [0,1], size W*H
  int m_pw = 0, m_ph = 0;
  int m_min_n = 0;        // highest index touched by `set`, + 1
  bool m_have_prev = false;
};
}
