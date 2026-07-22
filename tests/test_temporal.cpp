// Tests for the temporal-statistics family: CumulativeMean (cv.jit.mean), TemporalStats
// (cv.jit.variance / cv.jit.stddev) and OnlineCovariance (cv.jit.covariance).
//
// Everything here is asserted against numbers computed by hand from the cv.jit sources, not
// against "looks plausible" tolerances — the whole point of these ports is that they reproduce
// cv.jit's *exact* arithmetic, including one update rule that is deliberately not textbook.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ScoreTextureModel.hpp"
#include "TestImage.hpp"

#include <CV/Cpu/CumulativeMean.hpp>
#include <CV/Cpu/OnlineCovariance.hpp>
#include <CV/Cpu/TemporalStats.hpp>

#include <cmath>
#include <vector>

using Catch::Approx;
using namespace cvtest;

namespace
{
// Rec.601 luma as the objects compute it, so hand-computed expectations line up bit-for-bit
// with the implementation instead of assuming 0.299+0.587+0.114 == 1 exactly.
// NOTE the scale: both objects work in [0,255] internally but their r32f outputs are
// NORMALISED to [0,1], because score interprets an r32f texture as [0,1] when it converts it
// to the RGBA8 that every texture input expects (see the contract at the top of
// CV/Cpu/CartoPol.hpp). Everything below is therefore asserted in normalised units and
// cross-checked against the [0,255] cv.jit figure via the `Luma scale` value port.
double luma(double v)
{
  return 0.299 * v + 0.587 * v + 0.114 * v;
}

// The same value on the normalised scale the r32f outputs actually carry. Written with the
// same `* (1/255)` the objects use, so this stays bit-comparable with them.
double luma01(double v)
{
  return luma(v) * (1.0 / 255.0);
}

// Feed one uniform grey frame through a texture object.
template <typename Obj>
void pushGray(Obj& obj, Image& img, std::uint8_t v)
{
  img.fillRect(0, 0, img.width, img.height, v);
  feed(obj.inputs.image, img);
  obj();
}
}

// ================================================================= CumulativeMean (cv.jit.mean)

TEST_CASE("CumulativeMean is a 1/N running average, not an EMA", "[cumulative_mean]")
{
  cv::CumulativeMean obj;
  Image img(2, 2, 0);

  // The defining property: 10, 20, 30 -> 10, 15, 20 (arithmetic mean of everything seen).
  // An exponential leak would produce 10, 10+k*10, ... for whatever rate k, never 15 then 20.
  const std::uint8_t inputs[3] = {10, 20, 30};
  const std::uint8_t expected[3] = {10, 15, 20};

  for(int k = 0; k < 3; ++k)
  {
    pushGray(obj, img, inputs[k]);
    REQUIRE(obj.outputs.frames.value == k + 1);

    const auto& out = obj.outputs.image.texture;
    REQUIRE(out.width == 2);
    REQUIRE(out.height == 2);
    for(int i = 0; i < 4; ++i)
    {
      CHECK(int(out.bytes[i * 4 + 0]) == int(expected[k]));
      CHECK(int(out.bytes[i * 4 + 1]) == int(expected[k]));
      CHECK(int(out.bytes[i * 4 + 2]) == int(expected[k]));
      CHECK(int(out.bytes[i * 4 + 3]) == 255); // alpha is constant 255 -> mean stays 255
    }
    // The un-quantised luma output carries the same value at full precision.
    for(int i = 0; i < 4; ++i)
      CHECK(obj.outputs.mean.texture.bytes[i] == Approx(luma01(expected[k])).epsilon(1e-6));
  }
}

TEST_CASE("CumulativeMean holds a constant stream exactly", "[cumulative_mean]")
{
  cv::CumulativeMean obj;
  Image img(3, 3, 0);

  for(int k = 0; k < 500; ++k)
  {
    pushGray(obj, img, 137);
    const auto& out = obj.outputs.image.texture;
    for(int i = 0; i < 9; ++i)
      REQUIRE(int(out.bytes[i * 4]) == 137);
  }
  CHECK(obj.outputs.frames.value == 500);
  CHECK(obj.outputs.mean.texture.bytes[0] == Approx(luma01(137)).epsilon(1e-6));
}

TEST_CASE("CumulativeMean does not drift over many frames", "[cumulative_mean]")
{
  // Ramp 1..100: the exact arithmetic mean is 50.5. An exponential moving average at any
  // fixed rate lags a ramp by a constant offset and would never land on 50.5.
  cv::CumulativeMean obj;
  Image img(1, 1, 0);

  for(int k = 1; k <= 100; ++k)
    pushGray(obj, img, static_cast<std::uint8_t>(k));

  CHECK(obj.outputs.frames.value == 100);
  CHECK(obj.outputs.mean.texture.bytes[0] == Approx(luma01(50.5)).epsilon(1e-6));
  // 50.5 quantised with round-to-nearest -> 51 (cv.jit truncates; see the note in the header).
  CHECK(int(obj.outputs.image.texture.bytes[0]) == 51);
}

TEST_CASE("CumulativeMean reset restarts the averaging", "[cumulative_mean]")
{
  cv::CumulativeMean obj;
  Image img(2, 2, 0);

  pushGray(obj, img, 10);
  pushGray(obj, img, 20);
  REQUIRE(int(obj.outputs.image.texture.bytes[0]) == 15);

  SECTION("via the reset message")
  {
    obj.reset();
    CHECK(obj.outputs.frames.value == 2); // not yet recomputed
    pushGray(obj, img, 40);
    // cv.jit.mean does not clear its buffer on reset: with index back to 1 the next frame
    // runs with a = 1, b = 0 and replaces the accumulator wholesale.
    CHECK(obj.outputs.frames.value == 1);
    CHECK(int(obj.outputs.image.texture.bytes[0]) == 40);
    pushGray(obj, img, 60);
    CHECK(int(obj.outputs.image.texture.bytes[0]) == 50);
  }

  SECTION("via the Reset toggle (rising edge)")
  {
    obj.inputs.reset.value = true;
    pushGray(obj, img, 40);
    CHECK(obj.outputs.frames.value == 1);
    CHECK(int(obj.outputs.image.texture.bytes[0]) == 40);

    // Held high: no further reset, the average keeps accumulating.
    pushGray(obj, img, 60);
    CHECK(obj.outputs.frames.value == 2);
    CHECK(int(obj.outputs.image.texture.bytes[0]) == 50);

    // Dropping the toggle must be *observed* by a tick before it can re-arm.
    obj.inputs.reset.value = false;
    pushGray(obj, img, 80);
    CHECK(obj.outputs.frames.value == 3);
    CHECK(int(obj.outputs.image.texture.bytes[0]) == 60); // (40+60+80)/3

    // Rising again re-triggers.
    obj.inputs.reset.value = true;
    pushGray(obj, img, 90);
    CHECK(obj.outputs.frames.value == 1);
    CHECK(int(obj.outputs.image.texture.bytes[0]) == 90);
  }
}

TEST_CASE("CumulativeMean handles dimension changes and empty input", "[cumulative_mean]")
{
  cv::CumulativeMean obj;

  Image a(4, 4, 0);
  pushGray(obj, a, 100);
  pushGray(obj, a, 200);
  REQUIRE(obj.outputs.frames.value == 2);
  REQUIRE(int(obj.outputs.image.texture.bytes[0]) == 150);

  // A geometry change reallocates and restarts the counter (cv.jit does the same).
  Image b(8, 6, 0);
  pushGray(obj, b, 60);
  CHECK(obj.outputs.frames.value == 1);
  CHECK(obj.outputs.image.texture.width == 8);
  CHECK(obj.outputs.image.texture.height == 6);
  for(int i = 0; i < 8 * 6; ++i)
    CHECK(int(obj.outputs.image.texture.bytes[i * 4]) == 60);

  // Shrinking back also works.
  pushGray(obj, a, 90);
  CHECK(obj.outputs.frames.value == 1);
  CHECK(int(obj.outputs.image.texture.bytes[0]) == 90);

  // Null / zero-sized input must be ignored, not crash.
  obj.inputs.image.texture.bytes = nullptr;
  obj.inputs.image.texture.width = 4;
  obj.inputs.image.texture.height = 4;
  obj.inputs.image.texture.changed = true;
  obj();
  CHECK(obj.outputs.frames.value == 1);

  Image empty(0, 0, 0);
  obj.inputs.image.texture.bytes = nullptr;
  obj.inputs.image.texture.width = 0;
  obj.inputs.image.texture.height = 0;
  obj.inputs.image.texture.changed = true;
  obj();
  CHECK(obj.outputs.frames.value == 1);
}

// ============================================== TemporalStats (cv.jit.variance / cv.jit.stddev)

TEST_CASE("TemporalStats variance of a constant stream is exactly zero", "[temporal_stats]")
{
  cv::TemporalStats obj;
  Image img(4, 4, 0);

  for(int k = 0; k < 50; ++k)
  {
    pushGray(obj, img, 200);
    for(int i = 0; i < 16; ++i)
    {
      REQUIRE(obj.outputs.variance.texture.bytes[i] == 0.f);
      REQUIRE(obj.outputs.stddev.texture.bytes[i] == 0.f);
      REQUIRE(obj.outputs.mean.texture.bytes[i] == Approx(luma01(200)).epsilon(1e-6));
    }
  }
  CHECK(obj.outputs.frames.value == 50);
}

TEST_CASE("TemporalStats matches the hand-computed cv.jit patch", "[temporal_stats]")
{
  // The cv.jit.variance patch updates the mean with the *current* frame first, then feeds
  // (x - mean)^2 into a second cv.jit.mean. With an alternating 0 / L stream, L being white
  // on the NORMALISED scale the r32f outputs use (L = luma(255)/255 = 1):
  //   N=1 x=0: mu=0,       d=0,     var=0
  //   N=2 x=L: mu=L/2,     d=L/2,   var=(L^2/4)/2            = L^2/8
  //   N=3 x=0: mu=L/3,     d=-L/3,  var=(L^2/9)/3 + (L^2/8)(2/3) = 13 L^2/108
  //   N=4 x=L: mu=L/2,     d=L/2,   var=(L^2/4)/4 + (13L^2/108)(3/4) = 11 L^2/72
  cv::TemporalStats obj;
  Image img(2, 2, 0);

  const double L = luma01(255);
  const double L2 = L * L;
  const double expMean[4] = {0.0, L / 2.0, L / 3.0, L / 2.0};
  const double expVar[4]
      = {0.0, L2 / 8.0, 13.0 * L2 / 108.0, 11.0 * L2 / 72.0};

  for(int k = 0; k < 4; ++k)
  {
    pushGray(obj, img, (k % 2 == 0) ? 0 : 255);
    REQUIRE(obj.outputs.frames.value == k + 1);
    // The contract: whatever the statistic, the texture stays inside [0,1].
    REQUIRE(r32f_in_unit_range(obj.outputs.mean));
    REQUIRE(r32f_in_unit_range(obj.outputs.variance));
    REQUIRE(r32f_in_unit_range(obj.outputs.stddev));
    for(int i = 0; i < 4; ++i)
    {
      CHECK(obj.outputs.mean.texture.bytes[i] == Approx(expMean[k]).epsilon(1e-5));
      CHECK(obj.outputs.variance.texture.bytes[i] == Approx(expVar[k]).epsilon(1e-5));
      // stddev == sqrt(variance), elementwise.
      CHECK(
          obj.outputs.stddev.texture.bytes[i]
          == Approx(std::sqrt(expVar[k])).epsilon(1e-5));
    }
  }

  // And the cv.jit-scale figure is still recoverable through `Luma scale`: at N=4 the
  // variance of a 0/255 square wave is the well-known 16256.25 = 127.5^2 scaled by the
  // running-average transient, i.e. 11*255^2/72 = 9934.375.
  const double scale = obj.outputs.luma_scale.value;
  CHECK(scale == Approx(255.0));
  // A variance scales QUADRATICALLY with the unit change; the mean and stddev linearly.
  CHECK(obj.outputs.variance.texture.bytes[0] * scale * scale == Approx(9934.375).epsilon(1e-5));
  CHECK(obj.outputs.mean.texture.bytes[0] * scale == Approx(127.5).epsilon(1e-5));
  CHECK(
      obj.outputs.stddev.texture.bytes[0] * scale
      == Approx(std::sqrt(9934.375)).epsilon(1e-5));
  CHECK(11.0 * 255.0 * 255.0 / 72.0 == Approx(9934.375));
}

TEST_CASE("TemporalStats stddev is the elementwise sqrt of variance", "[temporal_stats]")
{
  cv::TemporalStats obj;
  Image img(3, 3, 0);

  for(int k = 0; k < 20; ++k)
  {
    img.fillRect(0, 0, 3, 3, static_cast<std::uint8_t>((k * 37) % 256));
    // Make each pixel see a different history so the check is not trivially uniform.
    for(int y = 0; y < 3; ++y)
      for(int x = 0; x < 3; ++x)
        img.setGray(x, y, static_cast<std::uint8_t>((k * 37 + (y * 3 + x) * 11) % 256));
    feed(obj.inputs.image, img);
    obj();

    for(int i = 0; i < 9; ++i)
    {
      const float v = obj.outputs.variance.texture.bytes[i];
      REQUIRE(v >= 0.f);
      REQUIRE(obj.outputs.stddev.texture.bytes[i] == Approx(std::sqrt(v)).epsilon(1e-6));
    }
  }
}

TEST_CASE("TemporalStats is per-pixel independent", "[temporal_stats]")
{
  // Pixel (0,0) is constant, pixel (1,0) alternates: their variances must differ.
  cv::TemporalStats obj;
  Image img(2, 1, 0);

  for(int k = 0; k < 4; ++k)
  {
    img.setGray(0, 0, 100);
    img.setGray(1, 0, (k % 2 == 0) ? 0 : 255);
    feed(obj.inputs.image, img);
    obj();
  }

  const double L = luma01(255);
  CHECK(obj.outputs.variance.texture.bytes[0] == 0.f);
  CHECK(obj.outputs.stddev.texture.bytes[0] == 0.f);
  CHECK(obj.outputs.mean.texture.bytes[0] == Approx(luma01(100)).epsilon(1e-6));

  CHECK(
      obj.outputs.variance.texture.bytes[1]
      == Approx(11.0 * L * L / 72.0).epsilon(1e-5));
  CHECK(obj.outputs.variance.texture.bytes[1] > obj.outputs.variance.texture.bytes[0]);
}

TEST_CASE("TemporalStats reset clears both accumulators", "[temporal_stats]")
{
  cv::TemporalStats obj;
  Image img(2, 2, 0);

  pushGray(obj, img, 0);
  pushGray(obj, img, 255);
  // Normalised units: L^2/8 = 0.125 for a 0 -> white pair (it was 2031.3 on the old
  // [0,255] scale, which score's r32f conversion turned into solid white).
  REQUIRE(obj.outputs.variance.texture.bytes[0] == Approx(0.125f).epsilon(1e-5));

  SECTION("message")
  {
    obj.reset();
    pushGray(obj, img, 77);
    CHECK(obj.outputs.frames.value == 1);
    CHECK(obj.outputs.variance.texture.bytes[0] == 0.f);
    CHECK(obj.outputs.stddev.texture.bytes[0] == 0.f);
    CHECK(obj.outputs.mean.texture.bytes[0] == Approx(luma01(77)).epsilon(1e-6));
  }

  SECTION("toggle")
  {
    obj.inputs.reset.value = true;
    pushGray(obj, img, 77);
    CHECK(obj.outputs.frames.value == 1);
    CHECK(obj.outputs.variance.texture.bytes[0] == 0.f);
    CHECK(obj.outputs.mean.texture.bytes[0] == Approx(luma01(77)).epsilon(1e-6));
  }
}

TEST_CASE("TemporalStats survives dimension changes and empty input", "[temporal_stats]")
{
  cv::TemporalStats obj;

  Image a(4, 4, 0);
  pushGray(obj, a, 0);
  pushGray(obj, a, 255);
  REQUIRE(obj.outputs.frames.value == 2);

  Image b(2, 7, 0);
  pushGray(obj, b, 128);
  CHECK(obj.outputs.frames.value == 1);
  CHECK(obj.outputs.variance.texture.width == 2);
  CHECK(obj.outputs.variance.texture.height == 7);
  for(int i = 0; i < 2 * 7; ++i)
    CHECK(obj.outputs.variance.texture.bytes[i] == 0.f);

  obj.inputs.image.texture.bytes = nullptr;
  obj.inputs.image.texture.width = 0;
  obj.inputs.image.texture.height = 0;
  obj.inputs.image.texture.changed = true;
  obj();
  CHECK(obj.outputs.frames.value == 1);
}

// ============================================================= the r32f [0,1] contract
//
// score converts an r32f texture output to RGBA8 by interpreting the float as [0,1]
// (`gray = qBound(0, int(v*255.f), 255)`, Crousti/TextureConversion.hpp). Both objects used
// to emit [0,255] luma units on their r32f ports, so ANY pixel above luma 1/255 arrived at
// the next object as pure white — every one of these objects' float outputs was unusable in
// a patch. These tests pin the normalisation down, at both ends of the range.
TEST_CASE("Temporal r32f outputs stay inside [0,1] for extreme inputs", "[contract]")
{
  SECTION("CumulativeMean on a white frame")
  {
    cv::CumulativeMean obj;
    Image img(4, 4, 0);
    pushGray(obj, img, 255);
    REQUIRE(r32f_in_unit_range(obj.outputs.mean));
    // White is the TOP of the range, not 255x past it.
    CHECK(obj.outputs.mean.texture.bytes[0] == Approx(1.f).epsilon(1e-6));
    CHECK(obj.outputs.luma_scale.value == Approx(255.f));
    // ...and it survives score's real conversion as white rather than being clipped there.
    Image conv = score_r32f_to_rgba8(obj.outputs.mean);
    CHECK(int(conv.px[0]) == 255);

    // A mid grey must come back as a mid grey, not as white.
    cv::CumulativeMean mid;
    Image img2(4, 4, 0);
    pushGray(mid, img2, 128);
    CHECK(mid.outputs.mean.texture.bytes[0] == Approx(128.f / 255.f).epsilon(1e-6));
    Image conv2 = score_r32f_to_rgba8(mid.outputs.mean);
    CHECK(int(conv2.px[0]) == 128);
  }

  SECTION("TemporalStats at maximum variance")
  {
    // A pixel alternating pure black / pure white is the worst case there is: for a signal
    // in [0,1] the variance cannot exceed 0.25 and the stddev cannot exceed 0.5, so this is
    // the tightest the normalisation is ever squeezed.
    //
    // The exact converged value is NOT 0.25, because cv.jit takes d_k = x_k - mu_k with mu
    // the running mean *at frame k* (see the patch trace at the top of this section), so the
    // early frames contribute a smaller d. Tracing the recurrence over 200 frames:
    //     var = (1/200) * [ 100 * 0.25  +  sum over odd k<=199 of ((k-1)/(2k))^2 ]
    //         = 0.2433281452...   ->   stddev 0.4932830275...
    // A textbook variance-about-the-final-mean would give exactly 0.25 here, so this number
    // also pins the cv.jit update ordering down.
    cv::TemporalStats obj;
    Image img(2, 2, 0);
    for(int k = 0; k < 200; ++k)
      pushGray(obj, img, (k % 2 == 0) ? 0 : 255);

    REQUIRE(r32f_in_unit_range(obj.outputs.mean));
    REQUIRE(r32f_in_unit_range(obj.outputs.variance));
    REQUIRE(r32f_in_unit_range(obj.outputs.stddev));
    CHECK(obj.outputs.mean.texture.bytes[0] == Approx(0.5f).epsilon(1e-4));
    CHECK(obj.outputs.variance.texture.bytes[0] == Approx(0.24332815).epsilon(1e-6));
    CHECK(obj.outputs.stddev.texture.bytes[0] == Approx(0.49328303).epsilon(1e-6));
    CHECK(obj.outputs.luma_scale.value == Approx(255.f));
    // The theoretical ceilings the normalisation relies on.
    CHECK(obj.outputs.variance.texture.bytes[0] <= 0.25f);
    CHECK(obj.outputs.stddev.texture.bytes[0] <= 0.5f);

    // The same statistic on cv.jit's [0,255] scale, via `Luma scale`: a variance scales
    // quadratically, so 0.2433 * 255^2 = 15822.4 (just under the 127.5^2 = 16256.25 ceiling).
    const double s = obj.outputs.luma_scale.value;
    CHECK(obj.outputs.variance.texture.bytes[0] * s * s == Approx(15822.41).epsilon(1e-5));
    CHECK(obj.outputs.variance.texture.bytes[0] * s * s < 127.5 * 127.5);

    // Through score's conversion, the three fields are distinguishable greys — not three
    // identical white rectangles, which is what the old [0,255] output produced.
    Image cm = score_r32f_to_rgba8(obj.outputs.mean);
    Image cv_ = score_r32f_to_rgba8(obj.outputs.variance);
    CHECK(int(cm.px[0]) == 127);  // 0.5      -> int(127.5)  truncates to 127
    CHECK(int(cv_.px[0]) == 62);  // 0.243328 -> int(62.049) truncates to 62
    CHECK(int(cm.px[0]) != int(cv_.px[0]));
  }
}

// ========================================================== OnlineCovariance (cv.jit.covariance)

TEST_CASE("OnlineCovariance reproduces cv.jit's exact update ordering", "[online_covariance]")
{
  // Hand trace of cv_jit_covariance_calculate_ndim over three 2-element frames.
  // The critical detail is that `n` is decremented *between* the mean loop and the var loop.
  //
  // frame 1: [1, 2]     n=1  -> mean=[1,2]          var=[0,0]
  // frame 2: [3, 4]     n=2  nn=1/2
  //                     mean=[1*.5+3/2, 2*.5+4/2] = [2, 3]
  //                     n-- -> 1, nn = 0/1 = 0
  //                     var =[0*0+(3-2)/1, 0*0+(4-3)/1] = [1, 1]
  // frame 3: [5, 10]    n=3  nn=2/3
  //                     mean=[2*2/3+5/3, 3*2/3+10/3] = [3, 16/3]
  //                     n-- -> 2, nn = 1/2
  //                     var =[1*.5+(5-3)/2, 1*.5+(10-16/3)/2] = [1.5, 17/6]
  //
  // A textbook implementation (no `n--`, nn stays 2/3 and the divisor stays 3) would give
  // var[0] = 1*(2/3) + 2/3 = 4/3 = 1.3333, NOT 1.5 — that is what this test pins down.
  cv::OnlineCovariance obj;

  auto push = [&](std::vector<float> v) {
    obj.inputs.vec.value = std::move(v);
    obj();
  };

  // ---- frame 1
  push({1.f, 2.f});
  REQUIRE(obj.outputs.frames.value == 1);
  REQUIRE(obj.outputs.size.value == 2);
  REQUIRE(obj.outputs.mean.value.size() == 2u);
  CHECK(obj.outputs.mean.value[0] == Approx(1.0));
  CHECK(obj.outputs.mean.value[1] == Approx(2.0));
  // n == 1 initialises the deviation accumulator to zero.
  CHECK(obj.outputs.deviation.value[0] == 0.f);
  CHECK(obj.outputs.deviation.value[1] == 0.f);
  REQUIRE(obj.outputs.matrix.value.size() == 4u);
  for(float v : obj.outputs.matrix.value)
    CHECK(v == 0.f);

  // ---- frame 2
  push({3.f, 4.f});
  REQUIRE(obj.outputs.frames.value == 2);
  CHECK(obj.outputs.mean.value[0] == Approx(2.0));
  CHECK(obj.outputs.mean.value[1] == Approx(3.0));
  CHECK(obj.outputs.deviation.value[0] == Approx(1.0));
  CHECK(obj.outputs.deviation.value[1] == Approx(1.0));
  for(float v : obj.outputs.matrix.value)
    CHECK(v == Approx(1.0));

  // ---- frame 3
  push({5.f, 10.f});
  REQUIRE(obj.outputs.frames.value == 3);
  CHECK(obj.outputs.mean.value[0] == Approx(3.0));
  CHECK(obj.outputs.mean.value[1] == Approx(16.0 / 3.0));

  // THE discriminating assertions: 1.5 and 17/6, not the textbook 4/3 and 26/9.
  CHECK(obj.outputs.deviation.value[0] == Approx(1.5));
  CHECK(obj.outputs.deviation.value[1] == Approx(17.0 / 6.0));
  CHECK(obj.outputs.deviation.value[0] != Approx(4.0 / 3.0));

  // Row-major outer product var * var^T.
  const double v0 = 1.5, v1 = 17.0 / 6.0;
  CHECK(obj.outputs.matrix.value[0] == Approx(v0 * v0)); // (0,0) = 2.25
  CHECK(obj.outputs.matrix.value[1] == Approx(v0 * v1)); // (0,1) = 4.25
  CHECK(obj.outputs.matrix.value[2] == Approx(v1 * v0)); // (1,0) = 4.25
  CHECK(obj.outputs.matrix.value[3] == Approx(v1 * v1)); // (1,1) = 8.027777...
  CHECK(obj.outputs.matrix.value[0] == Approx(2.25));
  CHECK(obj.outputs.matrix.value[1] == Approx(4.25));
  CHECK(obj.outputs.matrix.value[3] == Approx(289.0 / 36.0));
}

TEST_CASE("OnlineCovariance output is symmetric and rank-1", "[online_covariance]")
{
  cv::OnlineCovariance obj;
  const int N = 4;

  for(int k = 0; k < 12; ++k)
  {
    std::vector<float> v(N);
    for(int i = 0; i < N; ++i)
      v[i] = static_cast<float>(std::sin(0.3 * k + 0.7 * i) * 10.0 + i);
    obj.inputs.vec.value = v;
    obj();
  }

  const auto& m = obj.outputs.matrix.value;
  REQUIRE(m.size() == std::size_t(N * N));

  // Symmetry: m[j][i] == m[i][j].
  for(int j = 0; j < N; ++j)
    for(int i = 0; i < N; ++i)
      CHECK(m[j * N + i] == Approx(m[i * N + j]).margin(1e-5));

  // Rank 1: every 2x2 minor vanishes (m = v v^T).
  for(int j = 0; j + 1 < N; ++j)
    for(int i = 0; i + 1 < N; ++i)
    {
      const double det = double(m[j * N + i]) * double(m[(j + 1) * N + i + 1])
                         - double(m[j * N + i + 1]) * double(m[(j + 1) * N + i]);
      CHECK(det == Approx(0.0).margin(1e-3));
    }

  // ...and it really is the outer product of the exposed deviation vector.
  const auto& d = obj.outputs.deviation.value;
  for(int j = 0; j < N; ++j)
    for(int i = 0; i < N; ++i)
      CHECK(m[j * N + i] == Approx(double(d[j]) * double(d[i])).margin(1e-5));
}

TEST_CASE("OnlineCovariance reset zeroes the state", "[online_covariance]")
{
  cv::OnlineCovariance obj;

  obj.inputs.vec.value = {1.f, 2.f};
  obj();
  obj.inputs.vec.value = {3.f, 4.f};
  obj();
  REQUIRE(obj.outputs.deviation.value[0] == Approx(1.0));

  SECTION("message")
  {
    obj.reset();
    obj.inputs.vec.value = {7.f, 9.f};
    obj();
    CHECK(obj.outputs.frames.value == 1);
    CHECK(obj.outputs.mean.value[0] == Approx(7.0));
    CHECK(obj.outputs.mean.value[1] == Approx(9.0));
    CHECK(obj.outputs.deviation.value[0] == 0.f);
    CHECK(obj.outputs.deviation.value[1] == 0.f);
    for(float v : obj.outputs.matrix.value)
      CHECK(v == 0.f);
  }

  SECTION("toggle")
  {
    obj.inputs.reset.value = true;
    obj.inputs.vec.value = {7.f, 9.f};
    obj();
    CHECK(obj.outputs.frames.value == 1);
    CHECK(obj.outputs.deviation.value[0] == 0.f);

    // Held high -> no re-trigger.
    obj.inputs.vec.value = {9.f, 11.f};
    obj();
    CHECK(obj.outputs.frames.value == 2);
    CHECK(obj.outputs.mean.value[0] == Approx(8.0));
    CHECK(obj.outputs.deviation.value[0] == Approx(1.0));
  }
}

TEST_CASE("OnlineCovariance clears state when the input length changes", "[online_covariance]")
{
  cv::OnlineCovariance obj;

  obj.inputs.vec.value = {1.f, 2.f};
  obj();
  obj.inputs.vec.value = {3.f, 4.f};
  obj();
  REQUIRE(obj.outputs.frames.value == 2);

  // New length -> mean/var cleared, n back to 0 then straight to 1.
  obj.inputs.vec.value = {5.f, 6.f, 7.f};
  obj();
  CHECK(obj.outputs.frames.value == 1);
  CHECK(obj.outputs.size.value == 3);
  REQUIRE(obj.outputs.mean.value.size() == 3u);
  CHECK(obj.outputs.mean.value[0] == Approx(5.0));
  CHECK(obj.outputs.mean.value[1] == Approx(6.0));
  CHECK(obj.outputs.mean.value[2] == Approx(7.0));
  for(float v : obj.outputs.deviation.value)
    CHECK(v == 0.f);
  REQUIRE(obj.outputs.matrix.value.size() == 9u);
  for(float v : obj.outputs.matrix.value)
    CHECK(v == 0.f);

  // Shrinking works too, and the accumulation restarts cleanly afterwards.
  obj.inputs.vec.value = {2.f};
  obj();
  CHECK(obj.outputs.frames.value == 1);
  CHECK(obj.outputs.size.value == 1);
  REQUIRE(obj.outputs.matrix.value.size() == 1u);
  CHECK(obj.outputs.matrix.value[0] == 0.f);

  obj.inputs.vec.value = {4.f};
  obj();
  CHECK(obj.outputs.frames.value == 2);
  // n=2: mean = 2*0.5 + 4/2 = 3 ; n-- -> 1, nn = 0 ; var = (4-3)/1 = 1
  CHECK(obj.outputs.mean.value[0] == Approx(3.0));
  CHECK(obj.outputs.deviation.value[0] == Approx(1.0));
  CHECK(obj.outputs.matrix.value[0] == Approx(1.0));
}

TEST_CASE("OnlineCovariance handles an empty input vector", "[online_covariance]")
{
  cv::OnlineCovariance obj;

  obj.inputs.vec.value = {};
  obj();
  CHECK(obj.outputs.frames.value == 0);
  CHECK(obj.outputs.size.value == 0);
  CHECK(obj.outputs.matrix.value.empty());
  CHECK(obj.outputs.mean.value.empty());
  CHECK(obj.outputs.deviation.value.empty());

  // Accumulate, then go empty, then come back: state must have been dropped.
  obj.inputs.vec.value = {1.f, 2.f};
  obj();
  obj.inputs.vec.value = {3.f, 4.f};
  obj();
  REQUIRE(obj.outputs.frames.value == 2);

  obj.inputs.vec.value = {};
  obj();
  CHECK(obj.outputs.frames.value == 0);
  CHECK(obj.outputs.matrix.value.empty());

  obj.inputs.vec.value = {10.f, 20.f};
  obj();
  CHECK(obj.outputs.frames.value == 1);
  CHECK(obj.outputs.mean.value[0] == Approx(10.0));
  CHECK(obj.outputs.deviation.value[0] == 0.f);
}

TEST_CASE("OnlineCovariance deviation can go negative", "[online_covariance]")
{
  // cv.jit's `var` is a running mean of *signed* deviations, so a decreasing signal drives
  // it below zero — which a real variance could never do. Pinning that down guards against
  // anyone "fixing" it into a textbook variance.
  cv::OnlineCovariance obj;

  obj.inputs.vec.value = {10.f};
  obj();
  obj.inputs.vec.value = {0.f};
  obj();
  // n=2: mean = 10*0.5 + 0/2 = 5 ; n-- -> 1, nn = 0 ; var = (0-5)/1 = -5
  CHECK(obj.outputs.mean.value[0] == Approx(5.0));
  CHECK(obj.outputs.deviation.value[0] == Approx(-5.0));
  // The matrix is still non-negative on the diagonal: (-5)*(-5) = 25.
  CHECK(obj.outputs.matrix.value[0] == Approx(25.0));
}
