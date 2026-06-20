#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <vector>

namespace cv
{
struct chessboard_corner
{
  halp::xy_type<float> position; // normalised [0,1], row-major grid order

  halp_field_names(position);
};

// Chessboard inner-corner detector (cv.jit.findchessboardcorners). OpenCV-free,
// Eigen/std-based. Pipeline: grayscale -> smoothing -> saddle-point detection
// (Hessian with one strong + and one strong - eigenvalue = an X-junction) ->
// non-maximum suppression -> best-effort ordering into the requested
// (cols x rows) inner-corner grid.
//
// LIMITATION: the grid ordering is a best-effort PCA-axis banded sort, not a full
// topology/graph ordering. It recovers correct row-major order for roughly
// fronto-parallel or mildly-perspective boards; under strong perspective or when
// spurious saddle points survive it may mis-order, and `found` will be false if
// the detected count does not match cols*rows exactly. The saddle detection
// itself is robust and fires only at true checkerboard junctions.
//
// The `Threshold` input gates saddle detection: it scales the response fraction a
// junction must reach to survive non-maximum suppression. Lower -> permissive
// (more, weaker corners kept); higher -> strict (only the strongest junctions).
struct ChessboardCorners
{
  halp_meta(name, "Chessboard corners");
  halp_meta(c_name, "cv_chessboard_corners");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Detect a chessboard's inner corners via saddle-point detection and order "
      "them into a cols x rows grid. Ordering is best-effort (PCA-banded sort), "
      "not full topology ordering; see header notes.");
  halp_meta(uuid, "a04652ff-73cd-4afb-bedd-4dc715743c09");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_i32<"Cols", halp::range{2, 32, 7}> cols;     // inner corners/row
    halp::hslider_i32<"Rows", halp::range{2, 32, 7}> rows;     // inner corners/col
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
  } inputs;

  struct
  {
    halp::val_port<"Found", bool> found;
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Corners");
      std::vector<chessboard_corner> value;
    } corners;
  } outputs;

  void operator()() noexcept;
};
}
