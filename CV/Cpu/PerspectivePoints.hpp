#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace cv
{
// Plain 2D point, the element type of every point-list port in the "point ops" family
// (PerspectivePoints, FaceParts, FaceRigidPoints). Objects that share it chain with no
// adapter: FaceParts' "Eye L" outlet plugs straight into PerspectivePoints' "Points"
// inlet, for instance.
struct point2
{
  float x, y;

  halp_field_names(x, y);
};

// Row-major 3x3 identity, the default value of PerspectivePoints' matrix inlet.
inline constexpr std::array<float, 9> identity3{
    1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};

// Perspective transform of a point list (cv.jit.perspective).
//
// cv.jit.perspective is a *point* operator, not an image operator: its left inlet takes a
// 2-plane float32 `n x 1` Jitter matrix (n 2D points), its right inlet a 1-plane float32
// 3x3 matrix, and it calls cvPerspectiveTransform() on them. It never touches pixels.
// (CV/Shaders/Filters/Perspective.fs is a different operation entirely -- it warps an
// image by *inverse* mapping the destination raster, so it is not the counterpart of this
// object and must not be confused with it.)
//
// The mapping is the FORWARD one:
//     [x' y' w']^T = H * [x y 1]^T,   out = (x'/w', y'/w')
// which is exactly what a matrix produced by cv.jit.getperspective / cv::getPerspective
// (and by this addon's `Homography` object) expects. Feeding such a matrix to the
// inverse-mapping shader would warp the other way; here it does not.
//
// The matrix inlet takes the same `std::array<float, 9>` row-major layout that
// `cv::Homography` emits on its "Matrix" port, so `Homography -> PerspectivePoints` is a
// direct cable. It defaults to the identity, so a fresh object is a pass-through until it
// is fed -- max.cv.jit.perspective.cpp pre-fills its right inlet with the identity for the
// same reason.
struct PerspectivePoints
{
  halp_meta(name, "Perspective transform (points)");
  halp_meta(c_name, "cv_perspective_points");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Applies a 3x3 homography to a list of 2D points (cv.jit.perspective / "
      "cvPerspectiveTransform). Forward mapping [x' y' w'] = H.[x y 1], output "
      "(x'/w', y'/w'). The matrix is row-major and defaults to the identity, and matches "
      "the Homography object's Matrix output so the two chain directly.");
  halp_meta(uuid, "c1a70000-0040-4a00-9000-000000000001");

  struct
  {
    // Variable-length point list. The output always has exactly as many points.
    struct
    {
      halp_meta(name, "Points");
      std::vector<point2> value;
    } points;

    // Row-major 3x3: element [r * 3 + c] is H(r, c). Identity by default.
    halp::val_port<"Matrix", std::array<float, 9>> matrix{.value = identity3};
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Points");
      std::vector<point2> value;
    } points;
  } outputs;

  // Points on (or past) the horizon line of H -- those whose w' vanishes -- have no image
  // in the affine plane. OpenCV's perspectiveTransform emits (0, 0) for them rather than
  // an infinity, and that is reproduced here so that no inf/NaN can ever leave this
  // object. The threshold is 1e-8 rather than OpenCV's FLT_EPSILON (~1.19e-7): slightly
  // tighter, so a handful of extremely-near-horizon points that OpenCV zeroes are still
  // mapped here. Anything that could actually overflow to inf is caught either way.
  static constexpr double w_epsilon = 1e-8;

  void operator()() noexcept
  {
    const auto& in = inputs.points.value;
    auto& out = outputs.points.value;

    // Point count follows the input count, unconditionally.
    out.resize(in.size());

    const auto& h = inputs.matrix.value;
    // Accumulate in double: the coefficients of a homography routinely span several orders
    // of magnitude (the third row is ~1e-2 while the translation column is ~1e2), and the
    // division by w' amplifies any cancellation in the numerators.
    const double h0 = h[0], h1 = h[1], h2 = h[2];
    const double h3 = h[3], h4 = h[4], h5 = h[5];
    const double h6 = h[6], h7 = h[7], h8 = h[8];

    for(std::size_t i = 0; i < in.size(); ++i)
    {
      const double x = in[i].x;
      const double y = in[i].y;

      const double w = h6 * x + h7 * y + h8;
      if(!(std::abs(w) > w_epsilon) || !std::isfinite(w))
      {
        out[i] = {0.f, 0.f};
        continue;
      }

      const double u = (h0 * x + h1 * y + h2) / w;
      const double v = (h3 * x + h4 * y + h5) / w;
      if(!std::isfinite(u) || !std::isfinite(v))
      {
        // Reachable with a non-finite input point or a non-finite matrix coefficient:
        // still no inf/NaN on the wire.
        out[i] = {0.f, 0.f};
        continue;
      }

      out[i] = {static_cast<float>(u), static_cast<float>(v)};
    }
  }
};
}
