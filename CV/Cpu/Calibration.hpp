#pragma once

// `chessboard_corner` is the shared element type of the corner list ports: the
// ChessboardCorners object emits it and Calibration accepts it, so the two chain
// directly (see CONTRIBUTING_AGENTS.md, "List ports work in BOTH directions").
#include <CV/Cpu/ChessboardCorners.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <vector>

namespace cv
{
// One captured board view: cols*rows detected inner corners in PIXEL units, in
// row-major grid order (row 0 left-to-right, then row 1, ...).
using CalibrationView = std::vector<std::array<double, 2>>;

// Result of the linear Zhang solve. Everything is in PIXEL units except k1/k2
// (dimensionless, applied to normalised camera coordinates) and the diagnostics.
struct CalibrationSolution
{
  bool ok = false;

  double fx = 0, fy = 0; // focal lengths, pixels
  double cx = 0, cy = 0; // principal point, pixels, TOP-LEFT origin
  double skew = 0;       // K(0,1); solved for, not forced to zero
  double k1 = 0, k2 = 0; // radial distortion
  double rms = 0;        // RMS reprojection error, pixels

  // --- diagnostics, exposed so tests (and callers) can see *why* a solve failed --
  // Zhang's V b = 0 needs V to have rank exactly 5, i.e. a ONE-dimensional null
  // space. `rank_ratio` is sigma_4 / sigma_0 of V: ~1e-4 on a healthy set of
  // differently-oriented views, ~1e-20 when the views are duplicates / parallel /
  // differ only by a rotation about the board normal. Compared against
  // `rank_tolerance` below.
  double rank_ratio = 0;
  // |B11*B22 - B12^2| relative to |B11*B22| + B12^2. Zhang's closed form divides by
  // that determinant; because b is a unit vector, its ABSOLUTE magnitude is ~1/f^4
  // (1e-12 for f ~ 800), so only a RELATIVE guard is meaningful.
  double denom_ratio = 0;
  int views = 0; // number of views the solve consumed

  // A view set below this rank ratio is rejected outright: no K is produced. The
  // separation between healthy (~1e-4) and rank-deficient (~1e-20) sets is 16
  // orders of magnitude, so the exact value is not critical.
  static constexpr double rank_tolerance = 1e-8;
  // Relative floor on the closed form's denominator.
  static constexpr double denom_tolerance = 1e-10;
};

/* Zhang's method, linear (closed-form) only.
 *
 * `views` holds >= 3 sets of cols*rows image corners, in PIXELS, row-major. The
 * object-space model is a planar unit grid: inner corner (c, r) sits at (c, r, 0),
 * i.e. square size = 1 unit, so fx/fy/cx/cy come out in pixels.
 *
 * Steps: per-view Hartley-normalised DLT homography -> the linear system V b = 0 for
 * the image of the absolute conic -> closed-form K (including a solved-for skew) ->
 * per-view extrinsics (R, t) -> radial k1/k2 by linear least squares -> RMS.
 *
 * REJECTION (this is the part that matters). Zhang's system only determines b when V
 * has rank 5. Three identical views, three views that differ only by a translation in
 * the board plane, and three views that differ only by a rotation about the board
 * normal all leave V with rank 2, i.e. a 4-dimensional null space. Taking
 * `matrixV().col(5)` regardless then returns an arbitrary vector out of that null
 * space, and roughly 40% of such vectors happen to survive the fx^2 > 0 / fy^2 > 0
 * sign test and yield a completely wrong K (measured: fx = 1297 for a truth of 820).
 * The RMS reprojection error CANNOT catch this -- any consistent K/R/t factorisation
 * reproduces the (identical) homographies exactly, so RMS reads 0.000000 on the worst
 * possible answer. The rank test below is therefore the ONLY usable quality signal,
 * and `ok = false` leaves every intrinsic at 0.
 */
CalibrationSolution calibrate_zhang(
    int cols, int rows, const std::vector<CalibrationView>& views) noexcept;

// Camera calibration via Zhang's method (cv.jit.calibration), OpenCV-free, Eigen.
//
// STATEFUL & SELF-CONTAINED: it runs the shared chessboard detector internally on
// each frame. A `Capture` toggle grabs the currently-detected ordered corner set
// into an internal list of views; a `Solve` toggle then runs `calibrate_zhang` over
// all captured views. BOTH ARE EDGE-TRIGGERED (see below).
//
// EDGE TRIGGERING: Capture, Solve and Reset all fire on the RISING EDGE of their
// toggle, exactly like Learn's Capture/Train/Save/Load/Reset. Holding Capture down
// used to store one view per rendered frame -- ~60 identical views per second of a
// single board pose, which is precisely the rank-deficient input that makes Zhang's
// system unsolvable (and used to be reported as a perfect solve, RMS 0.000000).
// Holding Solve used to re-run the whole per-view DLT + SVD on every frame.
//
// CORNER SOURCE: when the "Corners" list input is non-empty it REPLACES the internal
// detector for that capture (same convention as Homography / SolvePnP: a non-empty
// list wins over the built-in source). The list carries normalised [0,1] positions,
// so a texture is still required -- its dimensions convert them back to pixels, and
// they are also what the normalised outputs below are relative to. This is how an
// external / better corner detector is plugged in, and it is what the numerical
// tests use.
//
// SCOPE / LIMITATIONS (documented):
//   * LINEAR Zhang ONLY -- there is NO nonlinear (Levenberg-Marquardt) bundle
//     refinement of the intrinsics/extrinsics; results are the closed-form estimate.
//   * Distortion model is RADIAL k1,k2 ONLY -- no tangential (p1,p2) terms.
//   * Skew is solved for (not forced to 0); it lands near zero on good planar data.
//   * Requires a MINIMUM of 3 captured views FROM DIFFERING BOARD ORIENTATIONS;
//     `Solve` is a no-op with fewer, and reports Solved = false with more views
//     that are geometrically degenerate (see `calibrate_zhang`).
//   * At most `max_views` views are kept; capturing past that drops the oldest,
//     as Learn does with `max_samples`.
//
// Object-space model: planar board, inner corners on a unit grid (col, row, 0),
// i.e. square size = 1 unit. K is recovered in pixel units relative to that grid;
// cx,cy,fx,fy are reported in pixels (see outputs).
struct Calibration
{
  halp_meta(name, "Camera calibration");
  halp_meta(c_name, "cv_calibration");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Zhang camera calibration from chessboard views. Capture (rising edge) grabs the "
      "current detected board, Solve (rising edge) computes intrinsics K (fx,fy,cx,cy), "
      "radial distortion (k1,k2) and RMS reprojection error. Self-contained: detects the "
      "board internally, or takes corners from the Corners list input. Needs >= 3 "
      "captured views from DIFFERING orientations; identical or parallel views are "
      "rejected (Solved = false) instead of producing a bogus K.");
  halp_meta(uuid, "d0e3b1ad-0542-4564-8314-b5c02d5adb40");

  // Bound on the stored view count (Learn's `max_samples` equivalent). Beyond this
  // the oldest view is dropped.
  static constexpr std::size_t max_views = 64;

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_i32<"Cols", halp::range{2, 32, 7}> cols;
    halp::hslider_i32<"Rows", halp::range{2, 32, 7}> rows;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::toggle<"Capture"> capture;
    halp::toggle<"Solve"> solve;
    halp::toggle<"Reset"> reset;

    // Externally-detected inner corners, normalised [0,1], row-major grid order.
    // When non-empty this replaces the internal detector for the next Capture.
    struct
    {
      halp_meta(name, "Corners");
      std::vector<chessboard_corner> value;
    } corners;
  } inputs;

  struct
  {
    // K, row-major 3x3, in pixel units (fx,skew,cx, 0,fy,cy, 0,0,1).
    halp::val_port<"K", std::array<float, 9>> K;
    halp::val_port<"Focal", halp::xy_type<float>> focal;   // fx, fy (pixels)
    halp::val_port<"Center", halp::xy_type<float>> center; // cx, cy (pixels, top-left)

    /* The same intrinsics in the NORMALISED convention that CV/Shaders/Filters/
     * Undistort.fs (and any other ISF filter working on isf_FragNormCoord) expects.
     * The pixel-unit ports above are 2-3 orders of magnitude outside those shaders'
     * slider ranges (fx in [0.1, 4], cx in [0, 1]), so nothing could be wired up
     * before these existed.
     *
     * Mapping: x scales by the image WIDTH, y by the image HEIGHT, because
     * isf_FragNormCoord is [0,1]^2 over the frame rather than a square:
     *     fx_n = fx / W      fy_n = fy / H
     *     cx_n = cx / W      cy_n = 1 - cy / H
     * With that substitution the shader's
     *     xn = ((uv - c_n) / f_n)
     * equals the pixel-space ((u - c) / f) exactly, so r^2 is identical and k1/k2
     * transfer UNCHANGED from the Distortion port.
     *
     * THE 1 - ON cy IS DELIBERATE. isf_FragNormCoord is derived in the ISF vertex
     * preamble as ((gl_Position.xy + 1) / 2), i.e. a BOTTOM-LEFT origin (the ISF /
     * OpenGL convention); score's ISF back-end then flips it back when sampling
     * (ISF_FIXUP_TEXCOORD(coord) = vec2(coord.x, 1 - coord.y)) because its textures
     * are stored top-left. cy here comes from the CPU corner detector, whose rows
     * count from the TOP. The two frames are opposite, hence the flip. Everything
     * else in the shader is invariant under it: the y term only ever enters as
     * (uv.y - cy_n), whose sign flip is squared away in r^2 and undone again when
     * the distorted point is re-projected.
     */
    halp::val_port<"Focal (normalised)", halp::xy_type<float>> focal_n;
    halp::val_port<"Center (normalised)", halp::xy_type<float>> center_n;

    halp::val_port<"Distortion", halp::xy_type<float>> distortion; // k1, k2
    halp::val_port<"RMS", float> rms;
    halp::val_port<"Views", int> views;
    halp::val_port<"Solved", bool> solved;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<CalibrationView> m_views;
  int m_cols = 7, m_rows = 7;
  int m_imgW = 0, m_imgH = 0;

  // Rising-edge latches: `Capture`/`Solve`/`Reset` fire once per off->on transition.
  bool m_prevCapture = false;
  bool m_prevSolve = false;
  bool m_prevReset = false;
};
}
