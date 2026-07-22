// Tests for the basic texture-in objects: Luminance, FloodFill, OpticalFlowLK.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/Luminance.hpp>
#include <CV/Cpu/FloodFill.hpp>
#include <CV/Cpu/OpticalFlowLK.hpp>

#include <cmath>

using Catch::Approx;
using namespace cvtest;

// --------------------------------------------------------------------------- Luminance
TEST_CASE("Luminance converts RGBA to Rec.601 grey", "[luminance]")
{
  cv::Luminance obj;
  Image img(4, 4, 0);
  img.set(0, 0, 255, 0, 0);   // red   -> ~76
  img.set(1, 0, 0, 255, 0);   // green -> ~150
  img.set(2, 0, 0, 0, 255);   // blue  -> ~29
  img.set(3, 0, 255, 255, 255); // white -> 255
  feed(obj.inputs.image, img);

  obj();

  auto& out = obj.outputs.image.texture;
  REQUIRE(out.bytes != nullptr);
  REQUIRE(out.width == 4);
  REQUIRE(out.height == 4);
  CHECK(out.bytes[0] == Approx(76).margin(2));
  CHECK(out.bytes[1] == Approx(150).margin(2));
  CHECK(out.bytes[2] == Approx(29).margin(2));
  CHECK(out.bytes[3] == 255);
}

TEST_CASE("Luminance ignores an unchanged frame", "[luminance]")
{
  cv::Luminance obj;
  Image img(2, 2, 128);
  feed(obj.inputs.image, img);
  obj.inputs.image.texture.changed = false; // not changed
  obj();
  // No output should have been produced (texture stays null/zero-size).
  CHECK(obj.outputs.image.texture.width == 0);
}

// ---------------------------------------------------------------------------- FloodFill
TEST_CASE("FloodFill fills a connected bright region only", "[floodfill]")
{
  cv::FloodFill obj;
  Image img(16, 16, 0);
  img.fillRect(4, 4, 5, 5, 255); // a 5x5 white block at (4,4)..(8,8)

  feed(obj.inputs.image, img);
  obj.inputs.seed.value = {6.f / 16.f, 6.f / 16.f}; // seed inside the block
  obj.inputs.tolerance.value = 0.1f;
  obj();

  CHECK(obj.outputs.filled.value == 25); // exactly the 5x5 block
  // and the output mask is white inside, black outside.
  auto& out = obj.outputs.image.texture;
  REQUIRE(out.bytes);
  CHECK(out.bytes[6 * 16 + 6] == 255);
  CHECK(out.bytes[0] == 0);
}

TEST_CASE("FloodFill on background fills the complement", "[floodfill]")
{
  cv::FloodFill obj;
  Image img(10, 10, 0);
  img.fillRect(3, 3, 3, 3, 255); // 9 white px

  feed(obj.inputs.image, img);
  obj.inputs.seed.value = {0.f, 0.f}; // seed in the black background
  obj.inputs.tolerance.value = 0.1f;
  obj();

  CHECK(obj.outputs.filled.value == 100 - 9);
}

TEST_CASE("FloodFill does not cross to a disconnected region", "[floodfill]")
{
  cv::FloodFill obj;
  Image img(16, 16, 0);
  img.fillRect(2, 2, 3, 3, 255);   // block A (9 px)
  img.fillRect(11, 11, 3, 3, 255); // block B (9 px), disconnected

  feed(obj.inputs.image, img);
  obj.inputs.seed.value = {3.f / 16.f, 3.f / 16.f}; // seed in A
  obj.inputs.tolerance.value = 0.1f;
  obj();
  CHECK(obj.outputs.filled.value == 9); // only A
}

// ------------------------------------------------------------------------ OpticalFlowLK
namespace
{
// A smooth sinusoidal texture shifted by `sh` pixels in x.
Image makeShifted(int W, int H, float sh)
{
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      float v = 0.5f + 0.4f * std::sin((x - sh) * 0.3f) * std::cos(y * 0.3f);
      std::uint8_t g = static_cast<std::uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f);
      img.setGray(x, y, g);
    }
  return img;
}
}

TEST_CASE("OpticalFlowLK first frame produces no flow (no previous)", "[flow]")
{
  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 5;
  obj.inputs.min_mag.value = 0.f;

  Image f0 = makeShifted(64, 64, 0.f);
  feed(obj.inputs.image, f0);
  obj();
  CHECK(obj.outputs.count.value == 0); // needs a previous frame
}

namespace
{
// Average velocity over the points marked tracked (status == 1). Returns the count of tracked
// points via `nTracked`.
void avgTrackedVelocity(
    const cv::OpticalFlowLK& obj, float& avx, float& avy, int& nTracked)
{
  float sumvx = 0.f, sumvy = 0.f;
  int n = 0;
  for(auto& fv : obj.outputs.flow.value)
  {
    if(fv.status != 1)
      continue;
    sumvx += fv.velocity.x;
    sumvy += fv.velocity.y;
    ++n;
  }
  nTracked = n;
  avx = n ? sumvx / n : 0.f;
  avy = n ? sumvy / n : 0.f;
}
}

TEST_CASE("OpticalFlowLK detects a rightward shift", "[flow]")
{
  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 6;
  obj.inputs.min_mag.value = 0.f;
  obj.inputs.pyramid_levels.value = 3;

  Image f0 = makeShifted(64, 64, 0.f);
  feed(obj.inputs.image, f0);
  obj(); // primes previous frame

  Image f1 = makeShifted(64, 64, 2.f); // shifted +2px in x
  feed(obj.inputs.image, f1);
  obj();

  REQUIRE(obj.outputs.count.value > 0);
  // Average velocity.x of tracked points should be positive (rightward), velocity.y near 0.
  float avx = 0.f, avy = 0.f;
  int nTracked = 0;
  avgTrackedVelocity(obj, avx, avy, nTracked);
  REQUIRE(nTracked > 0);
  CHECK(avx > 0.f);
  CHECK(std::abs(avy) < std::abs(avx));
}

namespace
{
// A medium-wavelength texture (period ~31px) chosen so that a large pixel displacement is
// still unambiguous, yet a single 5px LK window cannot span it in one solve. At this frequency
// a single-scale solve badly undershoots a 12px shift, while the coarse-to-fine pyramid (>= 2
// levels) recovers it almost exactly. Empirically validated.
Image makeShiftedSmooth(int W, int H, float sh)
{
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      float v = 0.5f + 0.4f * std::sin((x - sh) * 0.20f) * std::cos(y * 0.16f);
      std::uint8_t g = static_cast<std::uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f);
      img.setGray(x, y, g);
    }
  return img;
}
}

TEST_CASE("OpticalFlowLK pyramid tracks a large shift bigger than the window", "[flow]")
{
  // A 12px shift is far larger than a 5px window: a single-scale LK solve cannot recover it in
  // one window, but the coarse-to-fine pyramid can.
  const int shift = 12;
  const int W = 96, H = 96;
  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 5;
  obj.inputs.min_mag.value = 0.f;
  obj.inputs.pyramid_levels.value = 3;

  Image f0 = makeShiftedSmooth(W, H, 0.f);
  feed(obj.inputs.image, f0);
  obj();

  Image f1 = makeShiftedSmooth(W, H, static_cast<float>(shift));
  feed(obj.inputs.image, f1);
  obj();

  float avx = 0.f, avy = 0.f;
  int nTracked = 0;
  avgTrackedVelocity(obj, avx, avy, nTracked);
  REQUIRE(nTracked > 0);
  // Expected normalised x velocity ~ shift / W. The image content shifts +shift px, so the
  // flow (where current matches a +shift-displaced previous) is rightward and large.
  const float expected = static_cast<float>(shift) / W;
  CHECK(avx > 0.f);
  // Clearly larger than what a single 5px window could see, and near the true displacement.
  CHECK(avx > (5.f / W));
  CHECK(avx == Approx(expected).margin(expected * 0.4f));
  CHECK(std::abs(avy) < std::abs(avx));
}

TEST_CASE("OpticalFlowLK pyramid beats single-scale on the large shift", "[flow]")
{
  // Sanity check that the pyramid is doing real work: it should estimate the 12px shift more
  // accurately than a single-scale (pyramid_levels == 1) solve over the same 5px window.
  const int shift = 12;
  const int W = 96, H = 96;
  const float expected = static_cast<float>(shift) / W;

  auto run = [&](int levels) {
    cv::OpticalFlowLK obj;
    obj.inputs.grid.value = 8;
    obj.inputs.window.value = 5;
    obj.inputs.min_mag.value = 0.f;
    obj.inputs.pyramid_levels.value = levels;
    Image f0 = makeShiftedSmooth(W, H, 0.f);
    feed(obj.inputs.image, f0);
    obj();
    Image f1 = makeShiftedSmooth(W, H, static_cast<float>(shift));
    feed(obj.inputs.image, f1);
    obj();
    float avx = 0.f, avy = 0.f;
    int n = 0;
    avgTrackedVelocity(obj, avx, avy, n);
    return avx;
  };

  const float single = run(1);
  const float pyr = run(3);
  // The multi-level pyramid is closer to the true displacement than the single-scale estimate.
  CHECK(std::abs(pyr - expected) < std::abs(single - expected));
  // And the pyramid lands near the true displacement.
  CHECK(pyr == Approx(expected).margin(expected * 0.4f));
}

TEST_CASE("OpticalFlowLK emits a complete grid with per-point status", "[flow]")
{
  // A flat (constant) region is ill-conditioned: gradients vanish, the 2x2 system is singular,
  // so those points must be marked lost (status 0). Textured points moving should be tracked.
  const int W = 64, H = 64;
  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 5;
  obj.inputs.min_mag.value = 0.f;
  obj.inputs.pyramid_levels.value = 3;

  // Left half: flat grey (no texture). Right half: sinusoidal texture.
  auto build = [&](float sh) {
    Image img(W, H, 0);
    for(int y = 0; y < H; ++y)
      for(int x = 0; x < W; ++x)
      {
        std::uint8_t g;
        if(x < W / 2)
          g = 128; // flat
        else
        {
          float v = 0.5f + 0.4f * std::sin((x - sh) * 0.3f) * std::cos(y * 0.3f);
          g = static_cast<std::uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f);
        }
        img.setGray(x, y, g);
      }
    return img;
  };

  Image f0 = build(0.f);
  feed(obj.inputs.image, f0);
  obj();
  Image f1 = build(3.f);
  feed(obj.inputs.image, f1);
  obj();

  // The complete interior grid is emitted: (grid-1)^2 points.
  const int g = 8;
  REQUIRE(obj.outputs.count.value == (g - 1) * (g - 1));
  REQUIRE(obj.outputs.flow.value.size() == static_cast<std::size_t>((g - 1) * (g - 1)));

  int lostInFlat = 0, trackedInTextured = 0;
  for(auto& fv : obj.outputs.flow.value)
  {
    const bool inFlat = fv.position.x < 0.5f;
    if(inFlat && fv.status == 0)
      ++lostInFlat;
    if(!inFlat && fv.status == 1)
      ++trackedInTextured;
    // Lost points must carry zero velocity...
    if(fv.status == 0)
    {
      CHECK(fv.velocity.x == 0.f);
      CHECK(fv.velocity.y == 0.f);
      // ...and must NOT be reported as gated: there was no measurement to gate. `gated` is
      // only ever set on a point that produced a valid measurement.
      CHECK(fv.gated == 0);
    }
    // min_mag is 0 here, so nothing can be gated.
    CHECK(fv.gated == 0);
  }
  CHECK(lostInFlat > 0);     // flat region marked lost
  CHECK(trackedInTextured > 0); // textured region tracked
}

TEST_CASE("OpticalFlowLK: a static well-conditioned point is GATED, not lost", "[flow]")
{
  // The bug this pins: `min_mag` used to be folded into `status`, so with the default
  // threshold (0.001) every stationary but perfectly trackable point reported status = 0 --
  // indistinguishable from "there is nothing here to track". A consumer counting lost points
  // to decide the tracker had failed saw 100% failure on a still image.
  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 5;
  obj.inputs.pyramid_levels.value = 3;
  // Leave min_mag at its port default.
  REQUIRE(obj.inputs.min_mag.value == 0.001f);

  Image f0 = makeShifted(64, 64, 0.f);
  feed(obj.inputs.image, f0);
  obj();
  Image f1 = makeShifted(64, 64, 0.f); // byte-identical: the scene did not move
  feed(obj.inputs.image, f1);
  obj();

  REQUIRE(obj.outputs.flow.value.size() == 49);
  int gated = 0;
  for(auto& fv : obj.outputs.flow.value)
  {
    // Every point is well-conditioned (the texture is everywhere non-degenerate)...
    CHECK(fv.status == 1);
    // ...and every point measured zero motion, so every point is gated.
    CHECK(fv.gated == 1);
    CHECK(fv.velocity.x == 0.f);
    CHECK(fv.velocity.y == 0.f);
    CHECK(fv.magnitude == 0.f);
    gated += fv.gated;
  }
  CHECK(gated == 49);

  // With the gate open the very same frames report status 1 / gated 0 and (numerically) no
  // motion: the only thing min_mag changed is the flag, never the conditioning.
  obj.inputs.min_mag.value = 0.f;
  Image f2 = makeShifted(64, 64, 0.f);
  feed(obj.inputs.image, f2);
  obj();
  for(auto& fv : obj.outputs.flow.value)
  {
    CHECK(fv.status == 1);
    CHECK(fv.gated == 0);
    CHECK(std::abs(fv.magnitude) < 1e-5f);
  }
}

TEST_CASE("OpticalFlowLK: solvable only at coarse levels means LOST, not tracked", "[flow]")
{
  // OpenCV's rule, which cv::PointTracker already implements: a pyramid level that cannot be
  // solved forfeits only ITS refinement; the point is lost when the FINEST level fails.
  //
  // The scene: a 64x64 translating texture with a 17x17 FLAT square pinned at its centre
  // (32,32), i.e. spanning [24,40]^2. With Window = 5 the search window is 11x11:
  //   * level 0 at (32,32) spans [27,37]^2 -- entirely inside the flat square, so the
  //     structure tensor is singular and there is no measurement at full resolution;
  //   * level 1's 11x11 window covers +-10 FULL-RESOLUTION pixels, [22,42]^2, which still
  //     barely reaches out of the square, and level 2's covers +-20, [12,52]^2, which is
  //     mostly texture -- so the coarse levels DO solve, and produce a displacement that has
  //     nothing to do with the (featureless) centre.
  // The old flag was set by any level that solved and never cleared, so this point reported
  // status = 1 carrying that unrefined coarse estimate: measured (+0.0266, +0.0152)
  // normalised, i.e. (1.70, 0.97) px of pure fiction. It must be reported lost.
  const int W = 64, H = 64;
  auto build = [&](float tx, float ty) {
    Image img(W, H, 0);
    for(int y = 0; y < H; ++y)
      for(int x = 0; x < W; ++x)
      {
        float v;
        if(std::abs(x - 32) <= 8 && std::abs(y - 32) <= 8)
          v = 128.f; // the flat island, pinned
        else
          v = 128.f + 85.f * (0.5f * std::sin((x - tx) * 0.31f) * std::cos((y - ty) * 0.23f)
                              + 0.5f * std::sin((x - tx) * 0.11f + (y - ty) * 0.17f));
        img.setGray(x, y, static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.f, 255.f))));
      }
    return img;
  };

  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 5;
  obj.inputs.min_mag.value = 0.f;
  obj.inputs.pyramid_levels.value = 3;

  Image f0 = build(0.f, 0.f);
  feed(obj.inputs.image, f0);
  obj();
  Image f1 = build(3.f, 2.f);
  feed(obj.inputs.image, f1);
  obj();

  REQUIRE(obj.outputs.flow.value.size() == 49);
  // Grid point (gx, gy) = (4, 4) sits at (4*64/8, 4*64/8) = (32, 32), the centre of the
  // island; emission order is gy outer, gx inner over 1..7.
  const std::size_t centre = (4 - 1) * 7 + (4 - 1);
  CHECK(obj.outputs.flow.value[centre].position.x == Approx(0.5f));
  CHECK(obj.outputs.flow.value[centre].position.y == Approx(0.5f));
  CHECK(obj.outputs.flow.value[centre].status == 0);
  CHECK(obj.outputs.flow.value[centre].gated == 0);
  CHECK(obj.outputs.flow.value[centre].velocity.x == 0.f);
  CHECK(obj.outputs.flow.value[centre].velocity.y == 0.f);

  // It is the ONLY lost point: its 8-px-away neighbours have windows that reach out of the
  // 17x17 island at full resolution, so they are solvable and must stay tracked. (This is
  // what makes the assertion above about the level-0 rule and not about "flat regions are
  // lost" in general.)
  int lost = 0;
  for(auto& fv : obj.outputs.flow.value)
    lost += (fv.status == 0);
  CHECK(lost == 1);
}

// ------------------------------------------------------- transposition symmetry (BUG 2/3)
//
// WHAT THIS PAIR OF FRAMES IS, AND WHY IT IS THE RIGHT TEST
//
// The estimator must commute with TRANSPOSING the image: if `tall(x, y) == wide(y, x)` for
// every pixel, then every quantity it computes must swap its axes too. That is a property of
// the code, not of the scene, so it can be asserted to float precision -- unlike an accuracy
// comparison, which is limited by how well LK happens to do on the scene.
//
// It is exactly the property BUG 2 broke: `OpticalFlowLK` mapped a grid point into a pyramid
// level with `py = cy * (p.w / W)` -- the WIDTH ratio applied to the y coordinate -- and
// propagated between levels with a hardcoded x2. `downsample()` halves each dimension
// independently with max(1, n/2), so on an ODD dimension the two ratios differ (127 -> 63
// gives 0.49606 while 71 -> 35 gives 0.49296) and the two orientations then sample and
// propagate differently. Measured worst asymmetry over the grid below:
//     per-axis ratios (correct): 6.7e-08   <- float rounding from summation order only
//     shared width ratio + x2  : 1.2e-05   <- 185x larger
// so the 2e-6 tolerance asserted here sits an order of magnitude above the noise floor and an
// order of magnitude below the bug.
//
// 127 x 71 is deliberately odd in BOTH dimensions and non-square. With Window = 5 (an 11x11
// window) the pyramid is 127x71 -> 63x35 -> 31x17, all strictly larger than 11.
//
// THE MOTION is a linear stretch, not a translation: pixel (x, y) of the moved frame carries
// what was at (x - 6x/127, y - 3y/71), so the apparent displacement grows from ~0 at the
// origin to (6, 3) px at the far corner. A uniform translation would give every grid point
// the same magnitude and the min_mag gate would be all-or-nothing; here the normalised
// magnitude spans 0.008 .. 0.059 across the grid, so a threshold of 0.025 gates SOME points
// and not others -- and that non-trivial PATTERN is what must survive transposition.
namespace
{
float stretchTex(float u, float v)
{
  float s = 0.f;
  s += 0.45f * std::sin(0.16f * u + 0.09f * v + 0.7f);
  s += 0.30f * std::sin(0.07f * u - 0.19f * v + 2.1f);
  s += 0.25f * std::sin(0.21f * u + 0.05f * v + 4.2f);
  s += 0.20f * std::sin(0.06f * u + 0.17f * v + 1.3f);
  return s;
}

constexpr int kLong = 127;  // long axis extent
constexpr int kShort = 71;  // short axis extent
constexpr float kAmpLong = 6.f;   // px of stretch accumulated over the long axis
constexpr float kAmpShort = 3.f;  // ... and over the short axis

// `a` is the long-axis coordinate, `b` the short-axis one; `moved` selects reference/moved.
float stretchVal(float a, float b, bool moved)
{
  const float da = moved ? kAmpLong * a / kLong : 0.f;
  const float db = moved ? kAmpShort * b / kShort : 0.f;
  return 128.f + 85.f * stretchTex(a - da, b - db);
}

Image wideFrame(bool moved)
{
  Image img(kLong, kShort, 0);
  for(int y = 0; y < kShort; ++y)
    for(int x = 0; x < kLong; ++x)
      img.setGray(x, y, static_cast<std::uint8_t>(std::lround(
                            std::clamp(stretchVal(x, y, moved), 0.f, 255.f))));
  return img;
}

// The exact transpose: tallFrame(m).pixel(x, y) == wideFrame(m).pixel(y, x), bit for bit
// (both sides evaluate the same float expression and round it the same way).
Image tallFrame(bool moved)
{
  Image img(kShort, kLong, 0);
  for(int y = 0; y < kLong; ++y)
    for(int x = 0; x < kShort; ++x)
      img.setGray(x, y, static_cast<std::uint8_t>(std::lround(
                            std::clamp(stretchVal(y, x, moved), 0.f, 255.f))));
  return img;
}

std::vector<cv::flow_vector> runStretch(bool transposed, float minmag)
{
  cv::OpticalFlowLK obj;
  obj.inputs.grid.value = 8;
  obj.inputs.window.value = 5;
  obj.inputs.min_mag.value = minmag;
  obj.inputs.pyramid_levels.value = 3;
  Image f0 = transposed ? tallFrame(false) : wideFrame(false);
  feed(obj.inputs.image, f0);
  obj();
  Image f1 = transposed ? tallFrame(true) : wideFrame(true);
  feed(obj.inputs.image, f1);
  obj();
  return obj.outputs.flow.value;
}
}

TEST_CASE("OpticalFlowLK commutes with transposing an odd non-square frame", "[flow]")
{
  const int g = 8;
  const auto wide = runStretch(false, 0.f);
  const auto tall = runStretch(true, 0.f);
  REQUIRE(wide.size() == static_cast<std::size_t>((g - 1) * (g - 1)));
  REQUIRE(tall.size() == wide.size());

  // Grid point (gx, gy) of the wide frame is at (gx*127/8, gy*71/8); the transposed frame's
  // point (gy, gx) is at (gy*71/8, gx*127/8) -- the same pixel, transposed. Emission order is
  // gy outer / gx inner, so the partner index is simply the transposed index.
  float worst = 0.f;
  int compared = 0;
  for(int gy = 1; gy < g; ++gy)
    for(int gx = 1; gx < g; ++gx)
    {
      const auto& a = wide[static_cast<std::size_t>((gy - 1) * (g - 1) + (gx - 1))];
      const auto& b = tall[static_cast<std::size_t>((gx - 1) * (g - 1) + (gy - 1))];
      INFO("grid (" << gx << ", " << gy << ")");
      // Positions transpose exactly (both are k/8 for integer k... up to the integer division
      // in cx = gx*W/grid, which is why this is a margin and not an equality).
      CHECK(a.position.x == Approx(b.position.y).margin(1.f / kLong));
      CHECK(a.position.y == Approx(b.position.x).margin(1.f / kLong));
      // The load-bearing assertion: the velocity components swap, to float precision.
      CHECK(a.velocity.x == Approx(b.velocity.y).margin(2e-6));
      CHECK(a.velocity.y == Approx(b.velocity.x).margin(2e-6));
      CHECK(a.magnitude == Approx(b.magnitude).margin(2e-6));
      CHECK(a.status == b.status);
      worst = std::max(
          {worst, std::abs(a.velocity.x - b.velocity.y),
           std::abs(a.velocity.y - b.velocity.x)});
      ++compared;
    }
  CHECK(compared == 49);
  // Pinned so that a regression is visible as a number, not just a failed CHECK: the correct
  // per-axis code measures ~7e-8 here, the shared-width-ratio + x2 code measured 1.2e-5.
  INFO("worst asymmetry " << worst);
  CHECK(worst < 2e-6f);

  // The scene really is a stretch, and the estimator really did see it: the displacement must
  // grow along the long axis. Analytic displacement at (gx*127/8, gy*71/8) is
  // (6*gx/8, 3*gy/8) px, i.e. (0.75, 0.375) px at gx = gy = 1 and (5.25, 2.625) at 7.
  const auto& near = wide[0];                       // (gx, gy) = (1, 1)
  const auto& far = wide[(7 - 1) * (g - 1) + (7 - 1)]; // (7, 7)
  CHECK(near.velocity.x * kLong == Approx(0.75f).margin(0.4f));
  CHECK(near.velocity.y * kShort == Approx(0.375f).margin(0.4f));
  CHECK(far.velocity.x * kLong == Approx(5.25f).margin(0.5f));
  CHECK(far.velocity.y * kShort == Approx(2.625f).margin(0.5f));
  CHECK(far.magnitude > 5.f * near.magnitude);
}

TEST_CASE("OpticalFlowLK: the min_mag gate is a separate flag, and it too transposes",
          "[flow]")
{
  const int g = 8;

  SECTION("an open gate marks nothing")
  {
    for(const auto& fv : runStretch(false, 0.f))
    {
      CHECK(fv.status == 1);
      CHECK(fv.gated == 0);
    }
  }

  SECTION("a threshold inside the magnitude spread splits the grid, identically in both "
          "orientations")
  {
    // The measured magnitudes span 0.0083 (gx=gy=1) .. 0.0593 (gx=gy=7); 0.025 sits inside
    // that range, so the gate is genuinely selective rather than all-or-nothing.
    const auto wide = runStretch(false, 0.025f);
    const auto tall = runStretch(true, 0.025f);

    int gatedCount = 0, mismatches = 0;
    for(int gy = 1; gy < g; ++gy)
      for(int gx = 1; gx < g; ++gx)
      {
        const auto& a = wide[static_cast<std::size_t>((gy - 1) * (g - 1) + (gx - 1))];
        const auto& b = tall[static_cast<std::size_t>((gx - 1) * (g - 1) + (gy - 1))];
        INFO("grid (" << gx << ", " << gy << ")");
        // A gated point is NOT a lost point: the conditioning verdict is unchanged by the
        // threshold. (Before the fix these two facts shared one integer.)
        CHECK(a.status == 1);
        CHECK(b.status == 1);
        CHECK(a.gated == b.gated);
        if(a.gated)
        {
          CHECK(a.velocity.x == 0.f);
          CHECK(a.velocity.y == 0.f);
          CHECK(a.magnitude == 0.f);
        }
        gatedCount += a.gated;
        mismatches += (a.gated != b.gated);
      }
    CHECK(mismatches == 0);
    // Non-trivial split: neither everything nor nothing is gated.
    CHECK(gatedCount > 0);
    CHECK(gatedCount < 49);
  }

  SECTION("a threshold above every magnitude gates the whole grid without losing a point")
  {
    const auto wide = runStretch(false, 0.2f);
    for(const auto& fv : wide)
    {
      CHECK(fv.status == 1); // still perfectly trackable -- this used to read 0
      CHECK(fv.gated == 1);
      CHECK(fv.magnitude == 0.f);
    }
  }
}
