// Tests for tracking / calibration objects: CamShift, ChessboardCorners, Calibration.
// Chessboard detection robustness is best-effort (documented in the objects), so those tests
// assert the contract (no crash, sane outputs, flags behave) rather than exact grid recovery.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/CamShift.hpp>
#include <CV/Cpu/ChessboardCorners.hpp>
#include <CV/Cpu/Calibration.hpp>

#include <cmath>

using Catch::Approx;
using namespace cvtest;

namespace
{
// Paint a saturated red disc of radius r at (cx,cy) on a neutral-grey background.
Image makeRedBlob(int W, int H, int cx, int cy, int r)
{
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
      img.set(x, y, 110, 110, 110); // neutral grey (low saturation)
  for(int y = -r; y <= r; ++y)
    for(int x = -r; x <= r; ++x)
      if(x * x + y * y <= r * r)
      {
        int px = cx + x, py = cy + y;
        if(px >= 0 && py >= 0 && px < W && py < H)
          img.set(px, py, 230, 20, 20); // saturated red
      }
  return img;
}

// Synthetic checkerboard: (cols+1)x(rows+1) squares -> cols x rows inner corners.
Image makeCheckerboard(int W, int H, int cols, int rows)
{
  Image img(W, H, 255);
  int sq = W / (cols + 1);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      int cxi = x / sq, cyi = y / sq;
      std::uint8_t v = ((cxi + cyi) & 1) ? 0 : 255;
      img.setGray(x, y, v);
    }
  return img;
}
}

// ----------------------------------------------------------------------------- CamShift
TEST_CASE("CamShift tracks a coloured blob across frames", "[camshift]")
{
  cv::CamShift obj;
  obj.inputs.initSize.value = 0.2f;
  obj.inputs.iterations.value = 10;
  obj.inputs.minSat.value = 0.2f;
  obj.inputs.minVal.value = 0.2f;

  // Frame 0: blob at centre; arm the tracker with Set on a rising edge.
  Image f0 = makeRedBlob(64, 64, 32, 32, 8);
  feed(obj.inputs.image, f0);
  obj.inputs.seed.value = {0.5f, 0.5f};
  obj.inputs.set.value = false;
  obj();
  obj.inputs.set.value = true; // rising edge -> grab model
  feed(obj.inputs.image, f0);
  obj();
  REQUIRE(obj.outputs.tracking.value);

  // Frame 1: blob moved right+down; window should follow.
  obj.inputs.set.value = false;
  Image f1 = makeRedBlob(64, 64, 42, 40, 8);
  feed(obj.inputs.image, f1);
  obj();

  CHECK(obj.outputs.tracking.value);
  CHECK(obj.outputs.center.value.x == Approx(42.f / 64.f).margin(0.1));
  CHECK(obj.outputs.center.value.y == Approx(40.f / 64.f).margin(0.1));
}

TEST_CASE("CamShift is not tracking before Set", "[camshift]")
{
  cv::CamShift obj;
  Image f0 = makeRedBlob(64, 64, 32, 32, 8);
  feed(obj.inputs.image, f0);
  obj.inputs.set.value = false;
  obj();
  CHECK_FALSE(obj.outputs.tracking.value);
}

// Helper: arm the tracker on a centred blob and return the object ready to track.
namespace
{
void armCamShift(cv::CamShift& obj, Image& f0)
{
  obj.inputs.initSize.value = 0.2f;
  obj.inputs.iterations.value = 10;
  obj.inputs.minSat.value = 0.2f;
  obj.inputs.minVal.value = 0.2f;
  feed(obj.inputs.image, f0);
  obj.inputs.seed.value = {0.5f, 0.5f};
  obj.inputs.set.value = false;
  obj();
  obj.inputs.set.value = true; // rising edge -> grab model
  feed(obj.inputs.image, f0);
  obj();
  obj.inputs.set.value = false;
}
}

TEST_CASE("CamShift MeanShift mode keeps window fixed and angle 0", "[camshift]")
{
  using Mode = cv::TrackMode;

  cv::CamShift obj;
  obj.inputs.mode.value = Mode::MeanShift;
  Image f0 = makeRedBlob(64, 64, 32, 32, 8);
  armCamShift(obj, f0);
  REQUIRE(obj.outputs.tracking.value);

  // After arming, the window size on this first tracked frame.
  const float w0 = obj.outputs.size.value.x;
  const float h0 = obj.outputs.size.value.y;
  CHECK(obj.outputs.angle.value == Approx(0.f));

  // Frame 1: blob moves; in MeanShift the centre follows but size stays fixed, angle 0.
  Image f1 = makeRedBlob(64, 64, 42, 40, 8);
  feed(obj.inputs.image, f1);
  obj();
  CHECK(obj.outputs.tracking.value);
  CHECK(obj.outputs.angle.value == Approx(0.f));
  CHECK(obj.outputs.size.value.x == Approx(w0));
  CHECK(obj.outputs.size.value.y == Approx(h0));
  // Still tracks the moved centre.
  CHECK(obj.outputs.center.value.x == Approx(42.f / 64.f).margin(0.1));
  CHECK(obj.outputs.center.value.y == Approx(40.f / 64.f).margin(0.1));
}

TEST_CASE("CamShift mode adapts window size to the blob", "[camshift]")
{
  using Mode = cv::TrackMode;

  // Two trackers armed identically on a small blob; one in CamShift mode adapts the
  // window to the (larger) blob in the next frame, the MeanShift one does not.
  Image f0 = makeRedBlob(64, 64, 32, 32, 6);

  cv::CamShift cam;
  cam.inputs.mode.value = Mode::CamShift;
  armCamShift(cam, f0);
  const float camW0 = cam.outputs.size.value.x;

  cv::CamShift mean;
  mean.inputs.mode.value = Mode::MeanShift;
  armCamShift(mean, f0);
  const float meanW0 = mean.outputs.size.value.x;

  // Next frame: a much larger blob at the same centre.
  Image f1 = makeRedBlob(64, 64, 32, 32, 16);

  feed(cam.inputs.image, f1);
  cam();
  feed(mean.inputs.image, f1);
  mean();

  REQUIRE(cam.outputs.tracking.value);
  REQUIRE(mean.outputs.tracking.value);

  // CamShift grew its reported size; MeanShift kept it fixed.
  CHECK(cam.outputs.size.value.x > camW0);
  CHECK(mean.outputs.size.value.x == Approx(meanW0));
}

TEST_CASE("CamShift mass is positive on a blob and ~0 when it disappears", "[camshift]")
{
  cv::CamShift obj;
  Image f0 = makeRedBlob(64, 64, 32, 32, 10);
  armCamShift(obj, f0);
  REQUIRE(obj.outputs.tracking.value);
  CHECK(obj.outputs.mass.value > 0.f);
  CHECK(obj.outputs.mass.value <= 1.f);

  // Frame with the blob removed: a flat neutral-grey image (no matching hue/sat).
  Image blank(64, 64, 0);
  for(int y = 0; y < 64; ++y)
    for(int x = 0; x < 64; ++x)
      blank.set(x, y, 110, 110, 110); // low-saturation grey -> backprojection 0
  feed(obj.inputs.image, blank);
  obj();
  // No backprojection weight -> not tracking and mass collapses to 0.
  CHECK_FALSE(obj.outputs.tracking.value);
  CHECK(obj.outputs.mass.value == Approx(0.f));
}

// -------------------------------------------------------------------- ChessboardCorners
TEST_CASE("ChessboardCorners detects the full 3x3 grid and bounds its output", "[chessboard]")
{
  cv::ChessboardCorners obj;
  obj.inputs.cols.value = 3;
  obj.inputs.rows.value = 3;
  obj.inputs.threshold.value = 0.5f;

  // makeCheckerboard(64, 64, 3, 3) uses sq = 64/4 = 16, i.e. a 4x4 grid of 16 px squares,
  // whose interior junctions are exactly the 3x3 = 9 inner corners at (16,16)..(48,48).
  // Assert that UNCONDITIONALLY: the previous form guarded the count check behind
  // `if(found)`, so an implementation that detected nothing satisfied the whole case.
  Image board = makeCheckerboard(64, 64, 3, 3);
  feed(obj.inputs.image, board);
  obj();

  CHECK(obj.outputs.found.value);
  CHECK(obj.outputs.count.value == 3 * 3);
  CHECK(obj.outputs.count.value == static_cast<int>(obj.outputs.corners.value.size()));
  REQUIRE(obj.outputs.corners.value.size() == 9u);
  for(auto& c : obj.outputs.corners.value)
  {
    CHECK(c.position.x >= 0.f);
    CHECK(c.position.x <= 1.f);
    CHECK(c.position.y >= 0.f);
    CHECK(c.position.y <= 1.f);
  }
}

// Build a checkerboard with a strong-contrast top-left region and a weak-contrast
// remainder. The single strong region sets a high max saddle response, so the
// weak junctions sit just below/above the (threshold-scaled) response gate: a low
// threshold keeps the whole grid, a high threshold rejects the weak corners.
namespace
{
Image makeMixedContrastBoard(int W, int H, int cols, int rows, float weakAmp)
{
  Image img(W, H, 255);
  int sq = W / (cols + 1);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      int cxi = x / sq, cyi = y / sq;
      bool black = ((cxi + cyi) & 1);
      const bool strong = (cxi < 2 && cyi < 2); // top-left 2x2 squares: full contrast
      const float amp = strong ? 1.0f : weakAmp;
      std::uint8_t lo = (std::uint8_t)std::lround(128.f - 127.f * amp);
      std::uint8_t hi = (std::uint8_t)std::lround(128.f + 127.f * amp);
      img.setGray(x, y, black ? lo : hi);
    }
  return img;
}
}

TEST_CASE("ChessboardCorners threshold gates detection (no longer a no-op)", "[chessboard]")
{
  // Same image, two threshold extremes. A higher threshold raises the saddle-response
  // gate, so it must keep STRICTLY FEWER (or equal) corners than a low threshold. The
  // weak junctions are borderline w.r.t. the gate, so the two thresholds disagree ->
  // proves the control is wired into detection rather than a silent no-op.
  auto run = [](float thr) {
    cv::ChessboardCorners obj;
    obj.inputs.cols.value = 5;
    obj.inputs.rows.value = 5;
    obj.inputs.threshold.value = thr;
    Image board = makeMixedContrastBoard(96, 96, 5, 5, 0.20f);
    feed(obj.inputs.image, board);
    obj();
    return obj.outputs.count.value;
  };

  const int permissive = run(0.0f); // lowest gate -> most saddles kept (full grid)
  const int strict = run(1.0f);     // highest gate -> weak corners rejected

  INFO("permissive(thr=0)=" << permissive << " strict(thr=1)=" << strict);
  // Monotonic effect: stricter threshold never keeps MORE corners...
  CHECK(strict <= permissive);
  // ...and on this borderline image it actually removes some -> proves it's wired.
  CHECK(strict < permissive);
}

TEST_CASE("ChessboardCorners finds no board on a flat image", "[chessboard]")
{
  cv::ChessboardCorners obj;
  obj.inputs.cols.value = 5;
  obj.inputs.rows.value = 4;
  Image flat(64, 64, 128);
  feed(obj.inputs.image, flat);
  obj();
  CHECK_FALSE(obj.outputs.found.value);
}

// ---------------------------------------------------------------------------- Calibration
//
// =========================================================================================
// Derivations
// =========================================================================================
// Calibration used to have ZERO numerical coverage: nothing read K, Focal, Distortion or
// RMS, and disabling its whole capture branch (`if(false && ...)`) left the entire suite
// green. Everything below is derived from a single synthetic pinhole camera.
//
// GROUND TRUTH. A 7 x 6 inner-corner board sits on the world plane Z = 0 with the corner
// (c, r) at (c, r, 0) -- square size = 1 unit, exactly the object model calibrate_zhang
// assumes. A view is generated by placing the board at (R, t) and projecting with
//     u = fx * (R X + t).x / (R X + t).z + cx      v = fy * (R X + t).y / ... + cy
// using fx = 820, fy = 790, cx = 331, cy = 245 on a 640 x 480 frame. fx != fy and
// (cx, cy) != (W/2, H/2) on purpose: a solver that silently assumed square pixels or a
// centred principal point would pass with symmetric numbers. All five poses were checked
// to keep the whole board inside the frame.
//
// Zhang's method is EXACT on noise-free data, so these are not "close enough" tolerances:
// 3 views recover fx = 820.000000000, fy = 790.000000000, cx = 331.000000000,
// cy = 245.000000000 with skew ~ 1e-14, k1/k2 ~ 1e-14 and RMS ~ 1e-13 px. The margins
// below are 1e-6 px, i.e. eight orders tighter than any plausible wrong answer.
//
// RANK DEFICIENCY (the bug this suite exists to pin). Zhang's V b = 0 only determines b
// when V has rank 5. `calPoses[4]` repeated three times leaves V with rank 2, i.e. a
// 4-dimensional null space, and taking matrixV().col(5) regardless returns an arbitrary
// vector out of it. Measured with the rank test removed, on exactly the data below:
//     ok = 1,  fx = 1297.450,  fy = 3064.757,  cx = 0.0023,  cy = -0.0032,
//     skew = -1350,  RMS = 1.46e-09
// -- a completely fabricated K that the RMS output calls PERFECT, because any consistent
// K/R/t factorisation reproduces three identical homographies exactly. sigma_4/sigma_0 of
// V is 2.1e-21 there versus 1.2e-04 for three genuinely different orientations, which is
// why the rank ratio (and nothing else) is the usable signal.
//
// NORMALISED PORTS. Focal (normalised) / Center (normalised) are what an ISF filter such
// as CV/Shaders/Filters/Undistort.fs can actually accept (its sliders are fx in [0.1, 4],
// cx in [0, 1], against a raw fx of 820). x scales by W, y by H, and cy is FLIPPED
// because isf_FragNormCoord counts from the bottom while the corner detector counts image
// rows from the top:
//     fx/W = 820/640 = 1.28125           fy/H = 790/480 = 1.6458333
//     cx/W = 331/640 = 0.5171875         1 - cy/H = 1 - 245/480 = 0.4895833
// Asserting 0.4895833 (and explicitly NOT 0.5104167) is what pins the Y convention.
// =========================================================================================
namespace
{
struct CalCam
{
  double fx, fy, cx, cy;
  int W, H;
};
constexpr CalCam calCam{820.0, 790.0, 331.0, 245.0, 640, 480};
constexpr int calCols = 7;
constexpr int calRows = 6;

using CalM9 = std::array<double, 9>; // row-major 3x3

CalM9 calMul(const CalM9& a, const CalM9& b)
{
  CalM9 r{};
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
    {
      double s = 0;
      for(int k = 0; k < 3; ++k)
        s += a[static_cast<std::size_t>(i * 3 + k)] * b[static_cast<std::size_t>(k * 3 + j)];
      r[static_cast<std::size_t>(i * 3 + j)] = s;
    }
  return r;
}
CalM9 calRotX(double a)
{
  return {1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a)};
}
CalM9 calRotY(double a)
{
  return {std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)};
}
CalM9 calRotZ(double a)
{
  return {std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1};
}
CalM9 boardRot(double ax, double ay, double az)
{
  return calMul(calMul(calRotZ(az), calRotY(ay)), calRotX(ax));
}

struct BoardPose
{
  double ax, ay, az;
  std::array<double, 3> t;
};

// Project the calCols x calRows inner corners through the calCam pinhole, row-major.
cv::CalibrationView projectBoard(const BoardPose& p)
{
  const CalM9 R = boardRot(p.ax, p.ay, p.az);
  cv::CalibrationView out;
  out.reserve(static_cast<std::size_t>(calCols * calRows));
  for(int r = 0; r < calRows; ++r)
    for(int c = 0; c < calCols; ++c)
    {
      const double X = c, Y = r; // Z = 0: the board IS the plane
      const double Xc = R[0] * X + R[1] * Y + p.t[0];
      const double Yc = R[3] * X + R[4] * Y + p.t[1];
      const double Zc = R[6] * X + R[7] * Y + p.t[2];
      out.push_back({calCam.fx * Xc / Zc + calCam.cx, calCam.fy * Yc / Zc + calCam.cy});
    }
  return out;
}

// The same view as the object's "Corners" list input expects it: normalised [0,1].
std::vector<cv::chessboard_corner> asCornerList(const cv::CalibrationView& v)
{
  std::vector<cv::chessboard_corner> out;
  out.reserve(v.size());
  for(const auto& q : v)
    out.push_back(
        {{static_cast<float>(q[0] / calCam.W), static_cast<float>(q[1] / calCam.H)}});
  return out;
}

// Five well-separated board orientations; every projected corner lands inside the frame.
const std::array<BoardPose, 5> calPoses{
    {{0.00, 0.00, 0.00, {-3.0, -2.5, 14.0}},
     {0.40, -0.30, 0.15, {-3.2, -2.2, 13.0}},
     {-0.35, 0.45, -0.20, {-2.9, -2.2, 15.0}},
     {0.25, 0.50, 0.60, {-2.6, -2.4, 16.0}},
     {-0.60, -0.30, -0.40, {-3.0, -2.5, 14.0}}}};

std::vector<cv::CalibrationView> calViews(int n)
{
  std::vector<cv::CalibrationView> out;
  for(int i = 0; i < n; ++i)
    out.push_back(projectBoard(calPoses[static_cast<std::size_t>(i)]));
  return out;
}

// Deterministic +/- amp perturbation of every corner (a fixed LCG, so the numbers below
// are reproducible byte for byte).
std::vector<cv::CalibrationView>
withNoise(std::vector<cv::CalibrationView> vs, double amp)
{
  unsigned s = 12345u;
  auto next = [&] {
    s = s * 1664525u + 1013904223u;
    return ((s >> 8) & 0xffffu) / 65535.0 * 2.0 - 1.0;
  };
  for(auto& v : vs)
    for(auto& q : v)
    {
      q[0] += amp * next();
      q[1] += amp * next();
    }
  return vs;
}

// Drive one rising-edge capture of `corners` into `obj` (two evaluations: low, then high).
void captureOnce(
    cv::Calibration& obj, Image& frame, const std::vector<cv::chessboard_corner>& corners)
{
  obj.inputs.corners.value = corners;
  obj.inputs.capture.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.capture.value = true;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.capture.value = false;
}

// Drive one rising-edge Solve.
void solveOnce(cv::Calibration& obj, Image& frame)
{
  obj.inputs.solve.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.solve.value = true;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.solve.value = false;
}
}

TEST_CASE("Calibration recovers known intrinsics from synthetic views", "[calibration]")
{
  // Zhang is exact on noise-free data: these are equalities, not approximations.
  for(int n : {3, 4, 5})
  {
    const auto S = cv::calibrate_zhang(calCols, calRows, calViews(n));
    INFO("views = " << n);
    REQUIRE(S.ok);
    CHECK(S.views == n);
    CHECK(S.fx == Approx(calCam.fx).margin(1e-6));
    CHECK(S.fy == Approx(calCam.fy).margin(1e-6));
    CHECK(S.cx == Approx(calCam.cx).margin(1e-6));
    CHECK(S.cy == Approx(calCam.cy).margin(1e-6));
    // Skew is solved for, not forced: on a clean planar target it lands at the noise
    // floor. k1/k2 likewise, since the generator applies no distortion at all.
    CHECK(S.skew == Approx(0.0).margin(1e-6));
    CHECK(S.k1 == Approx(0.0).margin(1e-9));
    CHECK(S.k2 == Approx(0.0).margin(1e-9));
    CHECK(S.rms < 1e-9);
    // A healthy view set is nowhere near the rank floor.
    CHECK(S.rank_ratio > 1e-5);
    CHECK(S.denom_ratio > 0.1);
  }
}

TEST_CASE("Calibration degrades gracefully with corner noise", "[calibration]")
{
  // 0.3 px of corner noise: the linear (unrefined) estimate must stay sub-percent, and
  // the RMS must REPORT that noise rather than the 1e-13 of the clean case -- an RMS
  // that stayed at zero here would mean it is not measuring the data at all.
  const auto S3 = cv::calibrate_zhang(calCols, calRows, withNoise(calViews(3), 0.3));
  REQUIRE(S3.ok);
  INFO("3 noisy views: fx=" << S3.fx << " fy=" << S3.fy << " cx=" << S3.cx << " cy=" << S3.cy);
  CHECK(std::abs(S3.fx - calCam.fx) / calCam.fx < 0.01);
  CHECK(std::abs(S3.fy - calCam.fy) / calCam.fy < 0.01);
  CHECK(std::abs(S3.cx - calCam.cx) < 3.0);
  CHECK(std::abs(S3.cy - calCam.cy) < 3.0);
  CHECK(S3.rms > 0.05);
  CHECK(S3.rms < 1.0);

  const auto S5 = cv::calibrate_zhang(calCols, calRows, withNoise(calViews(5), 0.3));
  REQUIRE(S5.ok);
  CHECK(std::abs(S5.fx - calCam.fx) / calCam.fx < 0.01);
  CHECK(std::abs(S5.cx - calCam.cx) < 3.0);
}

TEST_CASE("Calibration rejects rank-deficient view sets", "[calibration]")
{
  // Positive control first: the SAME view count, only the orientations differ. Without it
  // a "return false always" implementation would satisfy every section below.
  {
    const auto S = cv::calibrate_zhang(calCols, calRows, calViews(3));
    REQUIRE(S.ok);
    REQUIRE(S.rank_ratio > 1e-5);
  }

  SECTION("three identical views")
  {
    // THE regression case. With the rank test removed this exact input returns ok = 1,
    // fx = 1297.450 (truth 820), fy = 3064.757 (truth 790), cx = 0.0023 (truth 331),
    // cy = -0.0032 (truth 245) -- and RMS = 1.46e-09, i.e. "perfect".
    const auto v = projectBoard(calPoses[4]);
    const auto S = cv::calibrate_zhang(calCols, calRows, {v, v, v});
    INFO("rank ratio = " << S.rank_ratio);
    CHECK_FALSE(S.ok);
    CHECK(S.rank_ratio < cv::CalibrationSolution::rank_tolerance);
    // Rejection leaves every intrinsic at zero; nothing fabricated escapes.
    CHECK(S.fx == 0.0);
    CHECK(S.fy == 0.0);
    CHECK(S.rms == 0.0);
  }

  SECTION("three views differing only by an in-plane translation")
  {
    // Parallel board orientations: h1/h2 are the same up to scale in every view, so the
    // two constraint rows a view contributes are the same two rows every time.
    const std::vector<cv::CalibrationView> vs{
        projectBoard({0, 0, 0, {-3.0, -2.5, 14.0}}),
        projectBoard({0, 0, 0, {-3.4, -2.1, 14.5}}),
        projectBoard({0, 0, 0, {-2.6, -2.9, 13.5}})};
    const auto S = cv::calibrate_zhang(calCols, calRows, vs);
    INFO("rank ratio = " << S.rank_ratio);
    CHECK_FALSE(S.ok);
    CHECK(S.rank_ratio < cv::CalibrationSolution::rank_tolerance);
  }

  SECTION("three views differing only by a rotation about the board normal")
  {
    // Spinning the board in its own plane adds no new constraint either.
    const std::vector<cv::CalibrationView> vs{
        projectBoard({0, 0, 0.0, {-3.0, -2.5, 14.0}}),
        projectBoard({0, 0, 0.7, {-3.0, -2.5, 14.0}}),
        projectBoard({0, 0, 1.4, {-3.0, -2.5, 14.0}})};
    const auto S = cv::calibrate_zhang(calCols, calRows, vs);
    INFO("rank ratio = " << S.rank_ratio);
    CHECK_FALSE(S.ok);
    CHECK(S.rank_ratio < cv::CalibrationSolution::rank_tolerance);
  }

  SECTION("fewer than three views is never solvable")
  {
    CHECK_FALSE(cv::calibrate_zhang(calCols, calRows, calViews(2)).ok);
    CHECK_FALSE(cv::calibrate_zhang(calCols, calRows, {}).ok);
  }

  SECTION("a view whose corner count does not match cols*rows is refused")
  {
    auto vs = calViews(3);
    vs[1].pop_back();
    CHECK_FALSE(cv::calibrate_zhang(calCols, calRows, vs).ok);
  }
}

TEST_CASE("Calibration publishes the recovered intrinsics on its ports", "[calibration]")
{
  // End-to-end through the object, feeding exact corners on the "Corners" list input
  // (the internal detector cannot reach this accuracy; see its own header note). This is
  // the only case that reads K / Focal / Center / Distortion / RMS at all.
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);

  for(int i = 0; i < 5; ++i)
    captureOnce(obj, frame, asCornerList(projectBoard(calPoses[static_cast<std::size_t>(i)])));
  REQUIRE(obj.outputs.views.value == 5);

  solveOnce(obj, frame);
  REQUIRE(obj.outputs.solved.value);

  // float ports + the [0,1] round trip through the list input cost ~1e-4 px.
  CHECK(obj.outputs.focal.value.x == Approx(820.f).margin(1e-2));
  CHECK(obj.outputs.focal.value.y == Approx(790.f).margin(1e-2));
  CHECK(obj.outputs.center.value.x == Approx(331.f).margin(1e-2));
  CHECK(obj.outputs.center.value.y == Approx(245.f).margin(1e-2));
  CHECK(obj.outputs.rms.value < 1e-3f);

  // K is row-major (fx, skew, cx, 0, fy, cy, 0, 0, 1) and agrees with Focal/Center.
  const auto& K = obj.outputs.K.value;
  CHECK(K[0] == Approx(obj.outputs.focal.value.x));
  CHECK(K[1] == Approx(0.f).margin(1e-3)); // skew
  CHECK(K[2] == Approx(obj.outputs.center.value.x));
  CHECK(K[3] == 0.f);
  CHECK(K[4] == Approx(obj.outputs.focal.value.y));
  CHECK(K[5] == Approx(obj.outputs.center.value.y));
  CHECK(K[6] == 0.f);
  CHECK(K[7] == 0.f);
  CHECK(K[8] == 1.f);

  // No distortion was applied by the generator, so k1/k2 must come back at zero.
  CHECK(obj.outputs.distortion.value.x == Approx(0.f).margin(1e-4));
  CHECK(obj.outputs.distortion.value.y == Approx(0.f).margin(1e-4));

  // --- the ISF-normalised ports ----------------------------------------------------
  // x by WIDTH, y by HEIGHT (isf_FragNormCoord is [0,1]^2 over a non-square frame).
  CHECK(obj.outputs.focal_n.value.x == Approx(820.f / 640.f).margin(1e-5)); // 1.28125
  CHECK(obj.outputs.focal_n.value.y == Approx(790.f / 480.f).margin(1e-5)); // 1.6458333
  CHECK(obj.outputs.center_n.value.x == Approx(331.f / 640.f).margin(1e-5)); // 0.5171875
  // Y ORIGIN: isf_FragNormCoord counts from the BOTTOM of the frame, the corner detector
  // counts image rows from the TOP, so the principal point's y is flipped.
  CHECK(obj.outputs.center_n.value.y == Approx(1.f - 245.f / 480.f).margin(1e-5)); // 0.4895833
  // ... and it is emphatically NOT the unflipped value.
  CHECK(obj.outputs.center_n.value.y != Approx(245.f / 480.f).margin(1e-3));
  // Both focal components are inside Undistort.fs' [0.1, 4] slider range, and both
  // centre components inside its [0, 1] -- the whole point of these ports.
  CHECK(obj.outputs.focal_n.value.x > 0.1f);
  CHECK(obj.outputs.focal_n.value.x < 4.f);
  CHECK(obj.outputs.focal_n.value.y > 0.1f);
  CHECK(obj.outputs.focal_n.value.y < 4.f);
  CHECK(obj.outputs.center_n.value.x >= 0.f);
  CHECK(obj.outputs.center_n.value.x <= 1.f);
  CHECK(obj.outputs.center_n.value.y >= 0.f);
  CHECK(obj.outputs.center_n.value.y <= 1.f);
}

TEST_CASE("Calibration reports a degenerate capture set as unsolved and keeps K", "[calibration]")
{
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);

  // 1. a good calibration, so there is a real K on the port to protect.
  for(int i = 0; i < 3; ++i)
    captureOnce(obj, frame, asCornerList(projectBoard(calPoses[static_cast<std::size_t>(i)])));
  solveOnce(obj, frame);
  REQUIRE(obj.outputs.solved.value);
  const auto goodK = obj.outputs.K.value;
  const float goodRms = obj.outputs.rms.value;
  REQUIRE(goodK[0] == Approx(820.f).margin(1e-2));

  // 2. start over and capture the SAME pose three times, which is what holding Capture
  //    down used to produce. V drops to rank 2; pre-fix this published fx = 1297.45 with
  //    RMS = 1.5e-09, i.e. a fabricated K that claimed to be perfect.
  obj.inputs.reset.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.reset.value = true;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.reset.value = false;
  REQUIRE(obj.outputs.views.value == 0);

  const auto dup = asCornerList(projectBoard(calPoses[4]));
  for(int i = 0; i < 3; ++i)
    captureOnce(obj, frame, dup);
  REQUIRE(obj.outputs.views.value == 3);

  solveOnce(obj, frame);
  CHECK_FALSE(obj.outputs.solved.value);
  // K, Focal, Center and RMS are left exactly as the last GOOD solve left them: a stale
  // real calibration, never an invented one.
  CHECK(obj.outputs.K.value == goodK);
  CHECK(obj.outputs.focal.value.x == Approx(goodK[0]));
  CHECK(obj.outputs.rms.value == Approx(goodRms));
  CHECK(obj.outputs.K.value[0] != Approx(1297.45f).margin(1.f)); // the pre-fix answer
}

TEST_CASE("Calibration Capture is edge-triggered", "[calibration]")
{
  // Holding Capture down for N frames on ONE board pose stored N identical views -- at
  // 60 fps that is ~60 duplicates per second, and duplicates are exactly the input that
  // makes Zhang's system rank-deficient. One rising edge == one view.
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);
  obj.inputs.corners.value = asCornerList(projectBoard(calPoses[0]));

  obj.inputs.capture.value = false;
  feed(obj.inputs.image, frame);
  obj();
  REQUIRE(obj.outputs.views.value == 0);

  obj.inputs.capture.value = true; // rising edge
  for(int frameIdx = 0; frameIdx < 8; ++frameIdx)
  {
    feed(obj.inputs.image, frame);
    obj();
    INFO("frame " << frameIdx << " with Capture held high");
    CHECK(obj.outputs.views.value == 1); // ... and it stays 1
  }

  // Dropping and re-raising the toggle is what stores a second view.
  obj.inputs.capture.value = false;
  feed(obj.inputs.image, frame);
  obj();
  CHECK(obj.outputs.views.value == 1);
  obj.inputs.capture.value = true;
  feed(obj.inputs.image, frame);
  obj();
  CHECK(obj.outputs.views.value == 2);
}

TEST_CASE("Calibration Solve is edge-triggered", "[calibration]")
{
  // Solve used to re-run the full per-view DLT + SVD on every single frame it was held
  // down. Observable consequence: with Solve held high, views captured AFTERWARDS were
  // silently folded into a new solve. One rising edge == one solve.
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);

  // Two views only: a Solve now cannot succeed.
  for(int i = 0; i < 2; ++i)
    captureOnce(obj, frame, asCornerList(projectBoard(calPoses[static_cast<std::size_t>(i)])));
  REQUIRE(obj.outputs.views.value == 2);

  obj.inputs.solve.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.solve.value = true; // rising edge, 2 views -> refused
  feed(obj.inputs.image, frame);
  obj();
  REQUIRE_FALSE(obj.outputs.solved.value);

  // Now add the third view WITHOUT ever lowering Solve. A level-triggered Solve would
  // pick it up on the very next frame and flip Solved to true.
  obj.inputs.corners.value = asCornerList(projectBoard(calPoses[2]));
  obj.inputs.capture.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.capture.value = true;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.capture.value = false;
  REQUIRE(obj.outputs.views.value == 3);

  for(int frameIdx = 0; frameIdx < 4; ++frameIdx)
  {
    feed(obj.inputs.image, frame);
    obj();
    INFO("frame " << frameIdx << " with Solve still held high");
    CHECK_FALSE(obj.outputs.solved.value);
  }

  // A fresh rising edge does solve, and gets the right answer.
  obj.inputs.solve.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.solve.value = true;
  feed(obj.inputs.image, frame);
  obj();
  CHECK(obj.outputs.solved.value);
  CHECK(obj.outputs.focal.value.x == Approx(820.f).margin(1e-2));
}

TEST_CASE("Calibration Reset is edge-triggered and clears the captured views", "[calibration]")
{
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);

  for(int i = 0; i < 3; ++i)
    captureOnce(obj, frame, asCornerList(projectBoard(calPoses[static_cast<std::size_t>(i)])));
  solveOnce(obj, frame);
  REQUIRE(obj.outputs.views.value == 3);
  REQUIRE(obj.outputs.solved.value);

  // Rising edge: the views really are dropped and Solved goes back to false.
  obj.inputs.reset.value = false;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.reset.value = true;
  feed(obj.inputs.image, frame);
  obj();
  CHECK(obj.outputs.views.value == 0);
  CHECK_FALSE(obj.outputs.solved.value);
  CHECK(obj.outputs.rms.value == 0.f);

  // Reset is held HIGH: a capture must still accumulate. A level-triggered Reset would
  // wipe the list again at the top of every frame, so views would never leave 0.
  for(int i = 0; i < 3; ++i)
    captureOnce(obj, frame, asCornerList(projectBoard(calPoses[static_cast<std::size_t>(i)])));
  CHECK(obj.outputs.views.value == 3);
  obj.inputs.reset.value = false;

  // A solve with too few views reports unsolved rather than leaving the flag stale.
  obj.inputs.reset.value = true;
  feed(obj.inputs.image, frame);
  obj();
  obj.inputs.reset.value = false;
  captureOnce(obj, frame, asCornerList(projectBoard(calPoses[0])));
  REQUIRE(obj.outputs.views.value == 1);
  solveOnce(obj, frame);
  CHECK_FALSE(obj.outputs.solved.value);
}

TEST_CASE("Calibration caps the number of stored views", "[calibration]")
{
  // Unbounded growth: `Capture` used to push_back forever with no equivalent of Learn's
  // `max_samples`. The oldest view is dropped once the cap is reached.
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);

  const int cap = static_cast<int>(cv::Calibration::max_views);
  for(int i = 0; i < cap + 5; ++i)
    captureOnce(
        obj, frame, asCornerList(projectBoard(calPoses[static_cast<std::size_t>(i % 5)])));
  CHECK(obj.outputs.views.value == cap);

  // Still a working calibration afterwards: the retained views are the LAST ones, which
  // still cover all five orientations.
  solveOnce(obj, frame);
  REQUIRE(obj.outputs.solved.value);
  CHECK(obj.outputs.focal.value.x == Approx(820.f).margin(1e-2));
}

TEST_CASE("Calibration accumulates exactly one view per capture", "[calibration]")
{
  cv::Calibration obj;
  obj.inputs.cols.value = 3;
  obj.inputs.rows.value = 3;

  // The internal-detector path (the Corners list is left empty). The same board the
  // detector resolves fully (9 of 9 corners, see the ChessboardCorners case above), so
  // every capture MUST store a view: after the k-th capture, views == k+1. `views >=
  // prevViews` was satisfied by a capture that did nothing at all (views stuck at 0),
  // which is what this case is supposed to be testing -- and it is what fails the moment
  // the capture branch is neutered.
  Image board = makeCheckerboard(64, 64, 3, 3);
  REQUIRE(obj.inputs.corners.value.empty());

  REQUIRE(obj.outputs.views.value == 0);
  for(int i = 0; i < 3; ++i)
  {
    obj.inputs.capture.value = false;
    feed(obj.inputs.image, board);
    obj();
    CHECK(obj.outputs.views.value == i); // no rising edge -> no new view
    obj.inputs.capture.value = true;     // rising edge -> capture
    feed(obj.inputs.image, board);
    obj();
    CHECK(obj.outputs.views.value == i + 1);
  }
}

TEST_CASE("Calibration ignores a Corners list of the wrong size", "[calibration]")
{
  // A mis-sized list must not be silently padded, truncated, or blended with a detector
  // run on the (blank) frame: nothing is captured at all.
  cv::Calibration obj;
  obj.inputs.cols.value = calCols;
  obj.inputs.rows.value = calRows;
  Image frame(calCam.W, calCam.H, 128);

  auto shortList = asCornerList(projectBoard(calPoses[0]));
  shortList.pop_back();
  captureOnce(obj, frame, shortList);
  CHECK(obj.outputs.views.value == 0);

  // The correctly-sized list is accepted.
  captureOnce(obj, frame, asCornerList(projectBoard(calPoses[0])));
  CHECK(obj.outputs.views.value == 1);
}
