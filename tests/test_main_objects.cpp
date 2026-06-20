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
    // Lost points must carry zero velocity.
    if(fv.status == 0)
    {
      CHECK(fv.velocity.x == 0.f);
      CHECK(fv.velocity.y == 0.f);
    }
  }
  CHECK(lostInFlat > 0);     // flat region marked lost
  CHECK(trackedInTextured > 0); // textured region tracked
}

TEST_CASE("OpticalFlowLK min_mag gate is symmetric on a non-square frame", "[flow]")
{
  // On a 96x48 frame, the magnitude gate is computed in normalised units (u/W, v/H) so a
  // diagonal motion of equal normalised extent in x and y is gated consistently. Here we drive
  // a pure-x motion and check that the same small motion is gated the same whether the frame
  // is wide (96x48) or tall (48x96): the normalised threshold must not depend on aspect.
  auto runFrame = [](int W, int H, float threshold) {
    cv::OpticalFlowLK obj;
    obj.inputs.grid.value = 8;
    obj.inputs.window.value = 5;
    obj.inputs.min_mag.value = threshold;
    obj.inputs.pyramid_levels.value = 3;

    Image f0 = makeShifted(W, H, 0.f);
    feed(obj.inputs.image, f0);
    obj();
    Image f1 = makeShifted(W, H, 2.f);
    feed(obj.inputs.image, f1);
    obj();

    int tracked = 0;
    for(auto& fv : obj.outputs.flow.value)
      if(fv.status == 1)
        ++tracked;
    return tracked;
  };

  // A 2px shift on a 96-wide frame -> ~0.0208 normalised. With a threshold below that, motion
  // passes; with a threshold above it, motion is gated. This is purely a function of the
  // normalised magnitude, independent of which axis the frame is longer on.
  const float small = 0.005f; // below 2/96
  const float large = 0.1f;   // above 2/96

  CHECK(runFrame(96, 48, small) > 0);  // passes on wide frame
  CHECK(runFrame(48, 96, small) > 0);  // passes on tall frame
  CHECK(runFrame(96, 48, large) == 0); // gated on wide frame
  CHECK(runFrame(48, 96, large) == 0); // gated on tall frame
}
