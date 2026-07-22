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
//
// PARITY WITH cv.jit.findchessboardcorners:
//  - `Subpixel` / `Window size` / `Zero zone` map onto cv.jit's
//    cvFindCornerSubPix(gray, corners, count, window_size, zero_zone,
//    TermCriteria(EPS|ITER, 30, 0.1)) call. `Window size` and `Zero zone` are HALF
//    sizes, exactly as in cv.jit (defaults 11 11 and -1 -1). `Subpixel` has no
//    cv.jit equivalent — cv.jit always refines — so it defaults to ON: integer
//    corner positions are useless for calibration, and the whole point of the
//    parameters above is to parametrise the refinement. Switch it off only to look
//    at the raw saddle detections.
//  - The `Out` texture reproduces cv.jit's FIRST outlet: the input image with
//    cvDrawChessboardCorners painted on it — a colour zig-zag polyline through the
//    ordered corners with a cross + circle marker at each when the board was found,
//    plain red circles when it was not. The shipped help patch is built around this
//    visual feedback. Drawing is integer Bresenham lines/circles, no dependencies.
//  - Corners are emitted even when `Found` is false (cv.jit sends whatever
//    cvFindChessboardCorners returned on its corner outlet regardless of the flag),
//    and those partial detections are sub-pixel refined too.
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
    // Sub-pixel refinement, ON by default (see the parity note above).
    halp::toggle<"Subpixel", halp::toggle_setup{.init = true}> subpixel;
    // cv.jit `window_size`, default 11 11 — HALF of the refinement window.
    halp::xy_spinboxes_i32<"Window size", halp::range{1, 64, 11}> window_size;
    // cv.jit `zero_zone`, default -1 -1 — HALF of the central dead zone, -1 = none.
    halp::xy_spinboxes_i32<"Zero zone", halp::range{-1, 32, -1}> zero_zone;
  } inputs;

  struct
  {
    // cv.jit's first outlet: the input image with the corners drawn onto it.
    halp::texture_output<"Out", halp::rgba_texture> image;
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
