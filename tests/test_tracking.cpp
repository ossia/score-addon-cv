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
TEST_CASE("ChessboardCorners runs without crashing and bounds its output", "[chessboard]")
{
  cv::ChessboardCorners obj;
  obj.inputs.cols.value = 3;
  obj.inputs.rows.value = 3;
  obj.inputs.threshold.value = 0.5f;

  Image board = makeCheckerboard(64, 64, 3, 3);
  feed(obj.inputs.image, board);
  obj();

  // Contract: count matches the list size; all corners normalised; found implies exact count.
  CHECK(obj.outputs.count.value == static_cast<int>(obj.outputs.corners.value.size()));
  for(auto& c : obj.outputs.corners.value)
  {
    CHECK(c.position.x >= 0.f);
    CHECK(c.position.x <= 1.f);
    CHECK(c.position.y >= 0.f);
    CHECK(c.position.y <= 1.f);
  }
  if(obj.outputs.found.value)
    CHECK(obj.outputs.count.value == 3 * 3);
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
TEST_CASE("Calibration reset clears state; solve needs views", "[calibration]")
{
  cv::Calibration obj;
  obj.inputs.cols.value = 3;
  obj.inputs.rows.value = 3;
  obj.inputs.threshold.value = 0.5f;

  Image board = makeCheckerboard(64, 64, 3, 3);
  feed(obj.inputs.image, board);

  // Reset (rising edge).
  obj.inputs.reset.value = false;
  obj();
  obj.inputs.reset.value = true;
  obj();
  CHECK(obj.outputs.views.value == 0);
  CHECK_FALSE(obj.outputs.solved.value);
  obj.inputs.reset.value = false;

  // Solve with no/insufficient views must not crash and must report unsolved.
  obj.inputs.solve.value = false;
  obj();
  obj.inputs.solve.value = true;
  feed(obj.inputs.image, board);
  obj();
  CHECK_FALSE(obj.outputs.solved.value); // < 3 views
}

TEST_CASE("Calibration capture does not crash and counts views monotonically", "[calibration]")
{
  cv::Calibration obj;
  obj.inputs.cols.value = 3;
  obj.inputs.rows.value = 3;

  Image board = makeCheckerboard(64, 64, 3, 3);

  int prevViews = obj.outputs.views.value;
  for(int i = 0; i < 3; ++i)
  {
    obj.inputs.capture.value = false;
    feed(obj.inputs.image, board);
    obj();
    obj.inputs.capture.value = true; // rising edge -> attempt capture
    feed(obj.inputs.image, board);
    obj();
    CHECK(obj.outputs.views.value >= prevViews); // never decreases
    prevViews = obj.outputs.views.value;
  }
}
