// Tests for cv::HoughTransform (cv.jit.hough -- the classical transform with REAL rho/theta
// resolution attributes) and cv::Lines (cv.jit.lines -- Canny + probabilistic Hough,
// emitting line segments), plus HoughTransform -> Extrema -> HoughLines end to end.
//
// Every expectation below is derived by hand from the documented formulas; the derivations
// are written out in the comment blocks so they can be re-checked without running anything.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ScoreTextureModel.hpp"
#include "TestImage.hpp"

#include <CV/Cpu/Extrema.hpp>
#include <CV/Cpu/HoughLines.hpp>
#include <CV/Cpu/HoughTransform.hpp>
#include <CV/Cpu/Lines.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

using Catch::Approx;
using namespace cvtest;

namespace
{
inline constexpr double pi = std::numbers::pi_v<double>;

// ---------------------------------------------------------------- accumulator utilities

struct AccPeak
{
  int x{-1};   // rho bin (column)
  int y{-1};   // theta bin (row)
  float v{-1}; // votes, as stored -- i.e. CLAMPED to 255
  int ties{};  // how many bins share that value
};

// NOTE: the accumulator texture is r8 and holds min(votes, 255); every case below is
// designed to stay under that ceiling except the one test that deliberately crosses it,
// which reads the true maximum from the `Max count` outlet instead.
AccPeak argmax(const cv::HoughTransform& h)
{
  const auto& t = h.outputs.accum.texture;
  AccPeak p;
  for(int y = 0; y < t.height; ++y)
  {
    for(int x = 0; x < t.width; ++x)
    {
      const float v = t.bytes[y * t.width + x];
      if(v > p.v)
      {
        p = AccPeak{.x = x, .y = y, .v = v, .ties = 1};
      }
      else if(v == p.v)
      {
        ++p.ties;
      }
    }
  }
  return p;
}

std::vector<float> accumulatorOf(const cv::HoughTransform& h)
{
  const auto& t = h.outputs.accum.texture;
  return std::vector<float>(
      t.bytes, t.bytes + static_cast<std::size_t>(t.width) * t.height);
}

// What score ACTUALLY does to the accumulator texture on its way to cv::Extrema, whose
// input is RGBA8. Modelling this wrong is what hid the fact that the
// HoughTransform -> Extrema -> HoughLines chain could not run in the host at all: this file
// used to pretend the conversion was "one vote per grey level", i.e. int(v + 0.5).
//
// R32F -> RGBA8 (TextureConversion.hpp:350-364) reads the float as a NORMALISED [0,1]
// value:   gray = qBound(0, int(v * 255.0f), 255)
// so an accumulator holding raw vote counts saturates to 255 at ONE vote and the whole
// accumulator becomes a plateau -- on which Extrema's strict '>' finds nothing. That step is
// cvtest::score_r32f_to_gray8 (tests/ScoreTextureModel.hpp); this wrapper only exists to run
// it over a buffer that is no longer an r32f port.
Image r32fAsImage(int w, int h, const float* v)
{
  Image img{w, h};
  for(int y = 0; y < h; ++y)
    for(int x = 0; x < w; ++x)
      img.setGray(x, y, score_r32f_to_gray8(v[y * w + x]));
  return img;
}

// R8 -> RGBA8 (TextureConversion.hpp:257-272) is a plain byte copy into r, g and b, so the
// Rec.601 luma cv::Extrema decodes is exactly the byte that was written. This is why the
// accumulator is now an r8 texture.
Image accumulatorAsImage(const cv::HoughTransform& h)
{
  const auto& t = h.outputs.accum.texture;
  Image img{t.width, t.height};
  for(int y = 0; y < t.height; ++y)
    for(int x = 0; x < t.width; ++x)
      img.setGray(x, y, t.bytes[y * t.width + x]);
  return img;
}

void houghRun(cv::HoughTransform& h, Image& img, double rho, double theta, float thr = 0.f)
{
  feed(h.inputs.image, img);
  h.inputs.rho.value = rho;
  h.inputs.theta.value = theta;
  h.inputs.threshold.value = thr;
  h();
}

// ---------------------------------------------------------------------- Lines utilities

void linesDefaults(cv::Lines& l)
{
  l.inputs.threshold.value = 150.;
  l.inputs.resolution.value = 1;
  l.inputs.sensitivity.value = 50;
  l.inputs.gap.value = 2.;
  l.inputs.length.value = 10.;
  l.inputs.maxlines.value = 4096;
}

// A single 0 -> 255 horizontal step: rows [y0, H) are white, everything above is black.
//
// Canny turns this into EXACTLY ONE horizontal edge, at row y0 - 1. Derivation: with an
// unnormalised 3x3 Sobel, |gy| is 4*255 = 1020 on BOTH rows y0-1 and y0 (each sees a black
// row on one side and a white row on the other) and 0 everywhere else. The vertical NMS
// test is `m > mag[above] && m >= mag[below]`, so row y0-1 (above = 0) survives and row y0
// (above = 1020) does not. The outermost 1-pixel border is always dropped, so the edge runs
// from x = 1 to x = W - 2.
Image stepImage(int w, int h, int y0)
{
  Image img{w, h};
  img.fillRect(0, y0, w, h - y0, 255);
  return img;
}
}

// ============================================================== cv::HoughTransform

TEST_CASE("HoughTransform: accumulator peak of a straight line is hand-computable",
          "[hough][houghtransform]")
{
  // 16x16 image, a single VERTICAL line of 16 pixels at x = 8. rho = 1, theta = pi/4
  // (both exactly representable: pi/4 is pi scaled by a power of two).
  //
  //   numangle = (int)(pi / (pi/4))            = 4
  //   numrho   = (int)(((16 + 16) * 2 + 1) / 1) = 65
  //   offset   = (65 - 1) / 2                   = 32
  //
  // Angle bin 0 is theta = 0 -> cos = 1, sin = 0, so for every pixel (j = 8, i = 0..15):
  //   r = round((8 * 1 + i * 0) / 1) + 32 = 8 + 32 = 40
  // i.e. all 16 pixels land in the SAME bin: accum[0][40] = 16.
  //
  // No other bin can reach 16. Bin 2 (theta = pi/2) gives r = round(i) + 32, one vote per
  // row. Bins 1 and 3 (theta = pi/4, 3pi/4) give r = round(0.7071*(8 +/- i)) + 32, whose
  // rounding collides at most in pairs -> 2 votes max.
  cv::HoughTransform h;
  Image img{16, 16};
  for(int y = 0; y < 16; ++y)
    img.setGray(8, y, 255);

  houghRun(h, img, 1.0, pi / 4);

  CHECK(h.outputs.acc_width.value == 65);
  CHECK(h.outputs.acc_height.value == 4);
  CHECK(h.outputs.accum.texture.width == 65);
  CHECK(h.outputs.accum.texture.height == 4);

  const auto p = argmax(h);
  CHECK(p.x == 40); // rho bin
  CHECK(p.y == 0);  // theta bin
  CHECK(p.v == Approx(16.f));
  CHECK(p.ties == 1); // the maximum is unique

  // Total votes = (on pixels) * numangle: every pixel votes exactly once per angle.
  const auto acc = accumulatorOf(h);
  float total = 0.f;
  for(float v : acc)
    total += v;
  CHECK(total == Approx(16.f * 4.f));
}

TEST_CASE("HoughTransform: a non-square image places the peak at the true pixel rho",
          "[hough][houghtransform]")
{
  // The case CV/Shaders/Analysis/Hough.cs gets wrong: its accumulator is a fixed
  // 256 x 180 grid with a NORMALISED rho axis, so on a 64 x 16 image the rho bin no longer
  // corresponds to a distance in pixels. Here it does.
  //
  // 64 x 16, one HORIZONTAL line of 64 pixels at y = 8. rho = 1, theta = pi/32.
  //   numangle = (int)(pi / (pi/32))            = 32
  //   numrho   = (int)(((64 + 16) * 2 + 1) / 1) = 161
  //   offset   = (161 - 1) / 2                  = 80
  // Angle bin 16 is theta = 16*pi/32 = pi/2 -> cos = 0, sin = 1, so for every pixel
  // (j = 0..63, i = 8):
  //   r = round(0 * j + 1 * 8) + 80 = 8 + 80 = 88
  // -> accum[16][88] = 64, and 88 - 80 = 8 is literally the line's distance in PIXELS.
  cv::HoughTransform h;
  Image img{64, 16};
  for(int x = 0; x < 64; ++x)
    img.setGray(x, 8, 255);

  houghRun(h, img, 1.0, pi / 32);

  CHECK(h.outputs.acc_width.value == 161);
  CHECK(h.outputs.acc_height.value == 32);

  const auto p = argmax(h);
  CHECK(p.x == 88);
  CHECK(p.y == 16);
  CHECK(p.v == Approx(64.f));
  CHECK(p.ties == 1);
}

TEST_CASE("HoughTransform: numangle / numrho follow the cv.jit formulas",
          "[hough][houghtransform]")
{
  // numangle = (int)(pi / theta), numrho = (int)(((W + H) * 2 + 1) / rho). BOTH casts
  // TRUNCATE. A single lit pixel is enough -- we only look at the derived dimensions.
  auto dims = [](int w, int h, double rho, double theta) {
    cv::HoughTransform obj;
    Image img{w, h};
    img.setGray(0, 0, 255);
    houghRun(obj, img, rho, theta);
    return std::pair{obj.outputs.acc_width.value, obj.outputs.acc_height.value};
  };

  SECTION("square")
  {
    // (16 + 16) * 2 + 1 = 65
    CHECK(dims(16, 16, 1.0, pi / 4) == std::pair{65, 4});
    CHECK(dims(16, 16, 4.0, pi / 8) == std::pair{16, 8}); // (int)(65/4) = 16
  }

  SECTION("non-square 64 x 16 and 16 x 64 agree -- numrho depends on W + H only")
  {
    // (64 + 16) * 2 + 1 = 161
    CHECK(dims(64, 16, 1.0, pi / 16) == std::pair{161, 16});
    CHECK(dims(16, 64, 1.0, pi / 16) == std::pair{161, 16});
    CHECK(dims(64, 16, 20.0, pi / 16) == std::pair{8, 16}); // (int)(161/20) = 8
  }

  SECTION("non-square 100 x 20")
  {
    // (100 + 20) * 2 + 1 = 241; (int)(241 / 2) = 120
    CHECK(dims(100, 20, 2.0, pi / 4) == std::pair{120, 4});
    // (int)(241 / 3) = 80
    CHECK(dims(100, 20, 3.0, pi / 2) == std::pair{80, 2});
  }

  SECTION("the cv.jit defaults, including the 35-not-36 truncation quirk")
  {
    // Default theta = 5 * 0.01745329252 = 0.0872664626 rad. cv.jit's DEG constant is very
    // slightly LARGER than pi/180, so 36 * theta = 3.1415926536 > pi and therefore
    //   pi / theta = 35.999999996  ->  (int) -> 35, NOT 36.
    // A textbook implementation using pi/180 would produce 36 here. This is deliberate.
    cv::HoughTransform obj;
    Image img{64, 48};
    img.setGray(0, 0, 255);
    feed(obj.inputs.image, img);
    obj(); // leave rho / theta at their port defaults (4.0 and 5 degrees)

    CHECK(obj.inputs.rho.value == Approx(4.0));
    CHECK(obj.inputs.theta.value == Approx(0.0872664626));
    // numrho = (int)(((64 + 48) * 2 + 1) / 4) = (int)(225 / 4) = (int)56.25 = 56
    CHECK(obj.outputs.acc_width.value == 56);
    CHECK(obj.outputs.acc_height.value == 35);
  }

  SECTION("out-of-range rho / theta are clamped VISIBLY at calc time")
  {
    // cv.jit clamps into locals and never writes back, so its reported attribute and its
    // effective value diverge. Ours clamp to [1, 20] and [pi/360, pi/2]:
    //   rho 0.5 -> 1     : numrho   = (int)(((64+48)*2+1) / 1)   = 225
    //   theta 1e-4 -> pi/360 : numangle = (int)(pi / (pi/360))   = 360 (the cap)
    CHECK(dims(64, 48, 0.5, 0.0001) == std::pair{225, 360});
    //   rho 100 -> 20    : numrho   = (int)(225 / 20) = 11
    //   theta 3 -> pi/2  : numangle = (int)(pi / (pi/2)) = 2 (the floor)
    CHECK(dims(64, 48, 100.0, 3.0) == std::pair{11, 2});
  }
}

TEST_CASE("HoughTransform: theta changes the angular resolution", "[hough][houghtransform]")
{
  // Same 16x16 vertical line at x = 8; only theta moves. The accumulator HEIGHT is the
  // angular resolution, and the peak stays in theta bin 0 (theta = 0 is the normal of a
  // vertical line) with all 16 votes, at the same rho bin 40, whatever the resolution.
  Image img{16, 16};
  for(int y = 0; y < 16; ++y)
    img.setGray(8, y, 255);

  cv::HoughTransform coarse;
  houghRun(coarse, img, 1.0, pi / 4);
  CHECK(coarse.outputs.acc_height.value == 4);

  cv::HoughTransform fine;
  houghRun(fine, img, 1.0, pi / 32);
  CHECK(fine.outputs.acc_height.value == 32);

  // 8x more angular bins, same width, so 8x more cells.
  CHECK(fine.outputs.acc_width.value == coarse.outputs.acc_width.value);
  CHECK(accumulatorOf(fine).size() == 8 * accumulatorOf(coarse).size());

  const auto pc = argmax(coarse);
  const auto pf = argmax(fine);
  CHECK(pc.x == 40);
  CHECK(pc.y == 0);
  CHECK(pf.x == 40);
  CHECK(pf.y == 0);
  CHECK(pc.v == Approx(16.f));
  CHECK(pf.v == Approx(16.f));

  // The finer grid really is finer: a 45-degree neighbour of the peak exists in the fine
  // accumulator (bin 8 = pi/4) and is a DIFFERENT row from the coarse bin 1.
  CHECK(fine.outputs.acc_height.value / coarse.outputs.acc_height.value == 8);
}

TEST_CASE("HoughTransform: an empty image gives an all-zero accumulator",
          "[hough][houghtransform]")
{
  cv::HoughTransform h;
  Image img{32, 24}; // all black, alpha 255
  houghRun(h, img, 2.0, pi / 8);

  // Dimensions are still derived and reported: (32+24)*2+1 = 113, (int)(113/2) = 56.
  CHECK(h.outputs.acc_width.value == 56);
  CHECK(h.outputs.acc_height.value == 8);

  const auto acc = accumulatorOf(h);
  REQUIRE(acc.size() == 56u * 8u);
  CHECK(std::all_of(acc.begin(), acc.end(), [](float v) { return v == 0.f; }));
}

TEST_CASE("HoughTransform: voting is BINARY -- the pixel value is never a weight",
          "[hough][houghtransform]")
{
  // cv.jit tests `*ip != 0`. The same line drawn at grey 40 and at grey 255 must give
  // bit-identical accumulators. An implementation that accumulated luminance (the obvious
  // "improvement") would give 40x different counts and fail this.
  auto lineImage = [](std::uint8_t v) {
    Image img{24, 24};
    for(int y = 4; y < 20; ++y)
      img.setGray(11, y, v);
    return img;
  };

  Image dim = lineImage(40);
  Image bright = lineImage(255);

  cv::HoughTransform a;
  houghRun(a, dim, 1.0, pi / 8);
  cv::HoughTransform b;
  houghRun(b, bright, 1.0, pi / 8);

  const auto accA = accumulatorOf(a);
  const auto accB = accumulatorOf(b);
  REQUIRE(accA.size() == accB.size());
  CHECK(accA == accB);

  // ... and the peak is the 16-pixel line, once, not 16*40 or 16*255.
  const auto p = argmax(a);
  CHECK(p.v == Approx(16.f));

  // `Threshold` is a BINARISER, not a weight: at 100 the grey-40 line disappears entirely
  // while the grey-255 line still votes exactly as before.
  cv::HoughTransform c;
  houghRun(c, dim, 1.0, pi / 8, 100.f);
  const auto accC = accumulatorOf(c);
  CHECK(std::all_of(accC.begin(), accC.end(), [](float v) { return v == 0.f; }));

  cv::HoughTransform d;
  houghRun(d, bright, 1.0, pi / 8, 100.f);
  CHECK(accumulatorOf(d) == accB);
}

TEST_CASE("HoughTransform: degenerate sizes and a mid-stream dimension change",
          "[hough][houghtransform]")
{
  cv::HoughTransform h;

  // 1x1: (1+1)*2+1 = 5 -> numrho = (int)(5/1) = 5, numangle = 4.
  Image one{1, 1};
  one.setGray(0, 0, 255);
  houghRun(h, one, 1.0, pi / 4);
  CHECK(h.outputs.acc_width.value == 5);
  CHECK(h.outputs.acc_height.value == 4);
  // The single pixel is at (0,0): r = round(0) + (5-1)/2 = 2 for every angle.
  {
    const auto acc = accumulatorOf(h);
    REQUIRE(acc.size() == 20u);
    for(int n = 0; n < 4; ++n)
      CHECK(acc[static_cast<std::size_t>(n) * 5 + 2] == 1.f);
  }

  // rho so coarse that the accumulator collapses to nothing: (int)(5 / 20) = 0. The object
  // must publish an EMPTY texture, not silently leave the previous frame's 5 x 4 one
  // attached while `Acc width` says 0 -- a downstream peak picker fed the stale texture
  // would keep reporting peaks from a frame that no longer exists.
  houghRun(h, one, 20.0, pi / 4);
  CHECK(h.outputs.acc_width.value == 0);
  CHECK(h.outputs.accum.texture.width == 0);
  CHECK(h.outputs.accum.texture.height == 0);
  CHECK(h.outputs.max_count.value == 0);

  // Dimension change mid-stream: buffers must be reallocated, no stale votes.
  Image big{40, 8};
  for(int x = 0; x < 40; ++x)
    big.setGray(x, 3, 255);
  houghRun(h, big, 1.0, pi / 16);
  // (40 + 8) * 2 + 1 = 97; theta bin 8 = pi/2 -> r = 3 + (97-1)/2 = 51, 40 votes.
  CHECK(h.outputs.acc_width.value == 97);
  CHECK(h.outputs.acc_height.value == 16);
  const auto p = argmax(h);
  CHECK(p.x == 51);
  CHECK(p.y == 8);
  CHECK(p.v == Approx(40.f));

  Image empty{40, 8};
  houghRun(h, empty, 1.0, pi / 16);
  const auto acc = accumulatorOf(h);
  CHECK(std::all_of(acc.begin(), acc.end(), [](float v) { return v == 0.f; }));
}

TEST_CASE("HoughTransform: the accumulator is R8, because R32F cannot survive the host",
          "[hough][houghtransform]")
{
  // REGRESSION TEST FOR THE BUG THAT MADE THE WHOLE CHAIN UNUSABLE IN score.
  //
  // The accumulator used to be an r32f texture holding raw counts. cv::Extrema takes RGBA8,
  // and score's converter reads an R32F texel as a normalised [0,1] colour
  // (TextureConversion.hpp:350-364), so EVERY non-empty bin arrived as 255: a saturated
  // plateau, on which Extrema's strict '>' finds nothing at all, forever.
  //
  // R8 -> RGBA8 is a byte copy instead, and 0..255 is exactly what Extrema's luma decode
  // expects. This test pins BOTH halves: the texture is r8 and carries the count verbatim,
  // and the old r32f route provably destroys it.
  static_assert(
      std::is_same_v<
          std::remove_cvref_t<decltype(*std::declval<cv::HoughTransform&>()
                                            .outputs.accum.texture.bytes)>,
          unsigned char>,
      "the accumulator must be an 8-bit texture -- see the header");

  // 64 x 16, a horizontal line of 64 pixels at y = 8, rho = 1, theta = pi/32:
  //   numrho = 161, numangle = 32, offset = 80, peak accum[16][88] = 64
  // (derived in "a non-square image places the peak at the true pixel rho" above). The peak
  // is deliberately NOT in theta bin 0: Extrema never tests the border row.
  //
  // Theta bin 0 (theta = 0 -> cos = 1, sin = 0) puts one vote in each of bins 80..143, so
  // accum[0][80] = 1 is a bin with a SINGLE vote to contrast the peak against.
  cv::HoughTransform h;
  Image img{64, 16};
  for(int x = 0; x < 64; ++x)
    img.setGray(x, 8, 255);
  houghRun(h, img, 1.0, pi / 32);

  const auto& t = h.outputs.accum.texture;
  REQUIRE(t.width == 161);
  REQUIRE(t.height == 32);
  CHECK(static_cast<int>(t.bytes[16 * 161 + 88]) == 64); // the count, not a normalised value
  CHECK(static_cast<int>(t.bytes[0 * 161 + 80]) == 1);
  CHECK(h.outputs.max_count.value == 64);

  // Through the R8 path the counts survive verbatim and Extrema finds the peak. Threshold
  // 30 isolates it: for any theta bin n != 16, consecutive x move the rho index by
  // |cos(theta_n)| >= sin(pi/32) = 0.098, so at most 1/0.098 + 1 = 11 pixels share a bin.
  {
    Image conv = accumulatorAsImage(h);
    CHECK(conv.px[(16 * 161 + 88) * 4] == 64);
    CHECK(conv.px[(0 * 161 + 80) * 4] == 1);
    cv::Extrema ex;
    feed(ex.inputs.image, conv);
    ex.inputs.threshold.value = 30;
    ex.inputs.maxpoints.value = 64;
    ex.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours8;
    ex();
    REQUIRE(ex.outputs.peaks.value.size() == 1);
    CHECK(ex.outputs.peaks.value[0].x == 88);
    CHECK(ex.outputs.peaks.value[0].y == 16);
    CHECK(ex.outputs.peaks.value[0].value == Approx(64.f));
  }

  // Through the OLD R32F path all count information is destroyed: 64 votes and 1 vote both
  // arrive as 255, and the true peak is no longer a local maximum at all (its neighbours in
  // theta bins 15 and 17 are non-zero, hence also 255, and strict '>' fails).
  {
    std::vector<float> raw;
    for(int i = 0; i < t.width * t.height; ++i)
      raw.push_back(static_cast<float>(t.bytes[i]));
    Image conv = r32fAsImage(t.width, t.height, raw.data());
    CHECK(conv.px[(16 * 161 + 88) * 4] == 255);
    CHECK(conv.px[(0 * 161 + 80) * 4] == 255); // a single vote, indistinguishable from 64

    cv::Extrema ex;
    feed(ex.inputs.image, conv);
    ex.inputs.threshold.value = 30;
    ex.inputs.maxpoints.value = 4096;
    ex.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours8;
    ex();
    for(const auto& p : ex.outputs.peaks.value)
      CHECK_FALSE((p.x == 88 && p.y == 16)); // the real line is never reported
  }
}

TEST_CASE("HoughTransform: Max count is the UNCLAMPED peak; the texture stops at 255",
          "[hough][houghtransform]")
{
  // 400 x 8, a horizontal run of 320 pixels at y = 3, rho = 1, theta = pi/16.
  //   numrho   = (int)(((400 + 8) * 2 + 1) / 1) = 817,  offset = (817 - 1) / 2 = 408
  //   theta bin 8 = pi/2 -> cos = 6.1e-17, sin = 1
  //   r = round(6.1e-17 * j + 3) + 408 = 411  for every one of the 320 pixels
  // so the true count in bin (411, 8) is 320, which does NOT fit in a byte.
  cv::HoughTransform h;
  Image img{400, 8};
  for(int x = 0; x < 320; ++x)
    img.setGray(x, 3, 255);
  houghRun(h, img, 1.0, pi / 16);

  REQUIRE(h.outputs.acc_width.value == 817);
  REQUIRE(h.outputs.acc_height.value == 16);

  const auto& t = h.outputs.accum.texture;
  CHECK(static_cast<int>(t.bytes[8 * 817 + 411]) == 255); // clamped in the texture
  CHECK(h.outputs.max_count.value == 320);                // reported in full on the outlet

  // Below the ceiling the two agree exactly.
  Image shortLine{400, 8};
  for(int x = 0; x < 100; ++x)
    shortLine.setGray(x, 3, 255);
  houghRun(h, shortLine, 1.0, pi / 16);
  CHECK(static_cast<int>(h.outputs.accum.texture.bytes[8 * 817 + 411]) == 100);
  CHECK(h.outputs.max_count.value == 100);
}

TEST_CASE("HoughTransform: the geometry outlets publish the EXACT steps, not pi/accH",
          "[hough][houghtransform]")
{
  // The cv.jit defaults on a 640 x 480 frame -- the configuration in which re-deriving the
  // steps from the accumulator size is worst.
  //   theta    = 5 * 0.01745329252 = 0.0872664626
  //   numangle = (int)(pi / theta) = (int)35.999999996 = 35   <- TRUNCATION
  //   pi / 35  = 0.08975979010256552, i.e. 2.857% LARGER than the step actually used;
  //              over the 34th theta bin that is 4.857 degrees of angular error.
  //   numrho   = (int)(((640 + 480) * 2 + 1) / 4) = (int)560.25 = 560  <- EVEN
  //   (numrho - 1) / 2 = 279 as an INTEGER, but 279.5 as a float: half a bin = 2 px.
  cv::HoughTransform h;
  Image img{640, 480};
  img.setGray(0, 0, 255);
  feed(h.inputs.image, img);
  h(); // port defaults: rho = 4, theta = 5 degrees

  CHECK(h.outputs.acc_width.value == 560);
  CHECK(h.outputs.acc_height.value == 35);
  CHECK(h.outputs.src_width.value == 640);
  CHECK(h.outputs.src_height.value == 480);
  CHECK(h.outputs.theta_step.value == Approx(0.0872664626).epsilon(1e-12));
  CHECK(h.outputs.rho_step.value == Approx(4.0));

  // The published step is NOT what pi/accH would give, and the difference is large.
  const double derived = pi / h.outputs.acc_height.value;
  CHECK(derived == Approx(0.08975979010256552).epsilon(1e-12));
  CHECK((derived - h.outputs.theta_step.value) / h.outputs.theta_step.value
        == Approx(0.0285714285).epsilon(1e-6));
  // Same for the rho offset: integer 279, not 279.5.
  CHECK((h.outputs.acc_width.value - 1) / 2 == 279);
  CHECK((h.outputs.acc_width.value - 1) / 2.0 == Approx(279.5));
}

// ============================================================================ cv::Lines

TEST_CASE("Lines: a single horizontal segment is recovered with its endpoints",
          "[lines]")
{
  // 64 x 40 step image, white from row 20 down. Canny -> one horizontal edge on row 19,
  // x in [1, 62] (the 1-pixel border is never an edge). See stepImage() for the derivation.
  //
  // With rho = 1 and theta = pi/180 the whole run lands in a single accumulator cell
  // (theta bin 90 -> r = round(19) + offset), which reaches sensitivity = 50 after 50 of
  // the 62 points; the walk then recovers the entire run in both directions.
  cv::Lines l;
  Image img = stepImage(64, 40, 20);
  feed(l.inputs.image, img);
  linesDefaults(l);
  l();

  REQUIRE(l.outputs.lines.value.size() == 1);
  CHECK(l.outputs.count.value == 1);
  const auto& s = l.outputs.lines.value[0];
  CHECK(s.x1 == Approx(1.f).margin(1.f));
  CHECK(s.y1 == Approx(19.f).margin(1.f));
  CHECK(s.x2 == Approx(62.f).margin(1.f));
  CHECK(s.y2 == Approx(19.f).margin(1.f));

  // Deterministic: the randomised point order is reseeded every frame, so re-running on
  // the same image gives exactly the same list.
  const auto first = l.outputs.lines.value;
  feed(l.inputs.image, img);
  l();
  CHECK(l.outputs.lines.value.size() == first.size());
  CHECK(l.outputs.lines.value[0].x1 == first[0].x1);
  CHECK(l.outputs.lines.value[0].x2 == first[0].x2);
}

TEST_CASE("Lines: two parallel segments give two entries", "[lines]")
{
  // A 9-row white bar (rows 12..20) has two 0<->255 transitions, so Canny yields two
  // parallel horizontal edges: one at row 11 (top, by the same NMS argument as stepImage)
  // and one at row 20 (bottom). They occupy different rho bins (11 vs 20 with rho = 1), so
  // the accumulator keeps them apart and both are extracted.
  cv::Lines l;
  Image img{64, 40};
  img.fillRect(0, 12, 64, 9, 255);
  feed(l.inputs.image, img);
  linesDefaults(l);
  l();

  REQUIRE(l.outputs.lines.value.size() == 2);
  CHECK(l.outputs.count.value == 2);

  std::vector<float> rows;
  for(const auto& s : l.outputs.lines.value)
  {
    CHECK(s.y1 == Approx(s.y2)); // both horizontal
    CHECK(s.x1 == Approx(1.f).margin(1.f));
    CHECK(s.x2 == Approx(62.f).margin(1.f));
    rows.push_back(s.y1);
  }
  std::sort(rows.begin(), rows.end());
  CHECK(rows[0] == Approx(11.f).margin(1.f));
  CHECK(rows[1] == Approx(20.f).margin(1.f));
}

TEST_CASE("Lines: `length` (minLineLength) rejects short segments", "[lines]")
{
  // 80 x 60, a white block occupying x in [0,9], y in [40,59] -- it touches the left and
  // bottom borders, so Canny produces exactly two edges:
  //   * a SHORT horizontal one on row 39, x in [1, 8]   -> |dx| = 7,  |dy| = 0
  //   * a LONG  vertical   one in column 9, y in [40,58] -> |dx| = 0, |dy| = 18
  // cv::HoughLinesP's acceptance test is Chebyshev: |dx| >= length || |dy| >= length.
  Image img{80, 60};
  img.fillRect(0, 40, 10, 20, 255);

  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 8; // both runs are shorter than the default 50 votes
    l.inputs.length.value = 5.;
    l();
    CHECK(l.outputs.lines.value.size() == 2); // 7 >= 5 and 18 >= 5
  }
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 8;
    l.inputs.length.value = 12.; // 7 < 12 -> the horizontal run is dropped
    l();
    REQUIRE(l.outputs.lines.value.size() == 1);
    const auto& s = l.outputs.lines.value[0];
    CHECK(std::abs(s.y2 - s.y1) == Approx(18.f).margin(1.f)); // the vertical one survived
    CHECK(s.x1 == Approx(s.x2));
  }
  {
    // Raising `length` past the longest run kills everything.
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 8;
    l.inputs.length.value = 25.;
    l();
    CHECK(l.outputs.lines.value.empty());
  }
  {
    // And on the long 62-pixel step edge, length = 100 rejects it too.
    cv::Lines l;
    Image step = stepImage(64, 40, 20);
    feed(l.inputs.image, step);
    linesDefaults(l);
    l.inputs.length.value = 100.;
    l();
    CHECK(l.outputs.lines.value.empty());
    CHECK(l.outputs.count.value == 0);
  }
}

TEST_CASE("Lines: `gap` bridges a small break but not a large one", "[lines]")
{
  // Same 64 x 40 step image, but a black vertical band punches a hole in the edge:
  //   band of 2 columns at x = 28  -> Canny row 19 becomes (1-26) + (31-62): a 4-cell break
  //   band of 6 columns at x = 28  -> Canny row 19 becomes (1-26) + (35-62): an 8-cell break
  // The walk tolerates a run of G missing cells iff G <= maxLineGap, so gap = 4 must bridge
  // the first and NOT the second. When the break is not bridged, only the half containing
  // the seed point survives (its points are then removed and un-voted, dropping the shared
  // accumulator cell back below `sensitivity`), so the result is a single SHORT segment.
  auto brokenStep = [](int bandWidth) {
    Image img = stepImage(64, 40, 20);
    img.fillRect(28, 0, bandWidth, 40, 0);
    return img;
  };

  Image small = brokenStep(2); // 4-cell break
  Image large = brokenStep(6); // 8-cell break

  {
    cv::Lines l;
    feed(l.inputs.image, small);
    linesDefaults(l);
    l.inputs.gap.value = 4.;
    l();
    REQUIRE(l.outputs.lines.value.size() == 1);
    const auto& s = l.outputs.lines.value[0];
    CHECK(s.x1 == Approx(1.f).margin(1.f));
    CHECK(s.x2 == Approx(62.f).margin(1.f)); // bridged: spans the whole row
    CHECK(s.y1 == Approx(19.f).margin(1.f));
  }
  {
    cv::Lines l;
    feed(l.inputs.image, small);
    linesDefaults(l);
    l.inputs.gap.value = 3.; // one short of the 4-cell break
    l();
    REQUIRE(l.outputs.lines.value.size() == 1);
    const auto& s = l.outputs.lines.value[0];
    CHECK(std::abs(s.x2 - s.x1) < 40.f); // only the left half
  }
  {
    cv::Lines l;
    feed(l.inputs.image, large);
    linesDefaults(l);
    l.inputs.gap.value = 4.; // the very setting that bridged the small break
    l();
    REQUIRE(l.outputs.lines.value.size() == 1);
    const auto& s = l.outputs.lines.value[0];
    CHECK(std::abs(s.x2 - s.x1) < 40.f); // NOT bridged
  }
  {
    cv::Lines l;
    feed(l.inputs.image, large);
    linesDefaults(l);
    l.inputs.gap.value = 8.; // exactly the break size -> bridged
    l();
    REQUIRE(l.outputs.lines.value.size() == 1);
    const auto& s = l.outputs.lines.value[0];
    CHECK(s.x1 == Approx(1.f).margin(1.f));
    CHECK(s.x2 == Approx(62.f).margin(1.f));
  }
}

TEST_CASE("Lines: `sensitivity` gates weak lines", "[lines]")
{
  // The step edge is 62 pixels long, so its accumulator cell tops out at 62 votes.
  Image img = stepImage(64, 40, 20);

  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 50; // 62 >= 50 -> found
    l();
    CHECK(l.outputs.lines.value.size() == 1);
  }
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 62; // exactly reachable -> still found
    l();
    CHECK(l.outputs.lines.value.size() == 1);
  }
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 63; // one vote more than the line can ever produce
    l();
    CHECK(l.outputs.lines.value.empty());
    CHECK(l.outputs.count.value == 0);
  }
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.sensitivity.value = 200;
    l();
    CHECK(l.outputs.lines.value.empty());
  }
}

TEST_CASE("Lines: empty and degenerate images yield no segments", "[lines]")
{
  {
    cv::Lines l;
    Image img{64, 40}; // all black -> no edges at all
    feed(l.inputs.image, img);
    linesDefaults(l);
    l();
    CHECK(l.outputs.lines.value.empty());
    CHECK(l.outputs.count.value == 0);
  }
  {
    cv::Lines l;
    Image img{64, 40, 255}; // uniformly white -> no gradient, no edges
    feed(l.inputs.image, img);
    linesDefaults(l);
    l();
    CHECK(l.outputs.lines.value.empty());
  }
  {
    cv::Lines l;
    Image img{1, 1};
    img.setGray(0, 0, 255);
    feed(l.inputs.image, img);
    linesDefaults(l);
    l();
    CHECK(l.outputs.lines.value.empty());
  }
  {
    cv::Lines l;
    Image img{2, 2};
    img.setGray(0, 0, 255);
    feed(l.inputs.image, img);
    linesDefaults(l);
    l();
    CHECK(l.outputs.lines.value.empty());
  }
  {
    // A previous non-empty result must be cleared by the next (empty) frame.
    cv::Lines l;
    Image step = stepImage(64, 40, 20);
    feed(l.inputs.image, step);
    linesDefaults(l);
    l();
    REQUIRE(l.outputs.lines.value.size() == 1);

    Image blank{64, 40};
    feed(l.inputs.image, blank);
    l();
    CHECK(l.outputs.lines.value.empty());
    CHECK(l.outputs.count.value == 0);
  }
}

TEST_CASE("Lines: resolution and max lines", "[lines]")
{
  Image img = stepImage(64, 40, 20);

  // `resolution` sets rho = resolution px AND theta = resolution degrees at once. The
  // horizontal edge stays exactly on a theta bin for every resolution in [1,10] (180 is
  // divisible by 1,2,5,10 among them, and by 3,4,6,9 too), so it is found either way --
  // this pins that a coarser grid does not lose the line or move its endpoints.
  for(int r : {1, 2, 5, 10})
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.resolution.value = r;
    l();
    INFO("resolution = " << r);
    REQUIRE(l.outputs.lines.value.size() == 1);
    CHECK(l.outputs.lines.value[0].x1 == Approx(1.f).margin(1.f));
    CHECK(l.outputs.lines.value[0].x2 == Approx(62.f).margin(1.f));
  }

  // Out-of-range resolution is clamped to [1,10] at calc time, not passed through.
  for(int r : {-5, 0, 99})
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.resolution.value = r;
    l();
    INFO("resolution = " << r);
    CHECK(l.outputs.lines.value.size() == 1);
  }

  // `Max lines` caps the list (our addition; cv.jit passes no cap at all).
  {
    cv::Lines l;
    Image bar{64, 40};
    bar.fillRect(0, 12, 64, 9, 255);
    feed(l.inputs.image, bar);
    linesDefaults(l);
    l();
    REQUIRE(l.outputs.lines.value.size() == 2);

    feed(l.inputs.image, bar);
    l.inputs.maxlines.value = 1;
    l();
    CHECK(l.outputs.lines.value.size() == 1);
    CHECK(l.outputs.count.value == 1);
  }
}

TEST_CASE("Lines: `threshold` drives the Canny band", "[lines]")
{
  // A step of 60 grey levels gives an unnormalised Sobel |gy| of 4*60 = 240.
  //   threshold = 150 -> band [140, 160]: 240 > 160, a strong edge -> line found.
  //   threshold = 255 -> band [245, 255]: 240 < 245, below `low` -> nothing at all.
  Image img{64, 40};
  img.fillRect(0, 20, 64, 20, 60);

  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l();
    CHECK(l.outputs.lines.value.size() == 1);
  }
  {
    cv::Lines l;
    feed(l.inputs.image, img);
    linesDefaults(l);
    l.inputs.threshold.value = 255.;
    l();
    CHECK(l.outputs.lines.value.empty());
  }
}

TEST_CASE("Lines and HoughLines share the same segment element type", "[lines]")
{
  // The whole point of reusing cv::line_segment: a std::vector<line_segment> produced by
  // one object is assignable to anything consuming the other's output.
  static_assert(
      std::is_same_v<
          decltype(std::declval<cv::Lines&>().outputs.lines.value)::value_type,
          decltype(std::declval<cv::HoughLines&>().outputs.lines.value)::value_type>);

  cv::Lines l;
  Image img = stepImage(64, 40, 20);
  feed(l.inputs.image, img);
  linesDefaults(l);
  l();
  REQUIRE(l.outputs.lines.value.size() == 1);

  std::vector<cv::line_segment> sink = l.outputs.lines.value; // compiles == interchangeable
  CHECK(sink.size() == 1);
}

// ================================================== HoughTransform -> Extrema -> HoughLines

TEST_CASE("Chain: HoughTransform -> Extrema -> HoughLines recovers the line",
          "[hough][houghtransform][extrema][hough2lines][chain]")
{
  // 64 x 64 image, one horizontal line of 64 pixels at y = 32. rho = 1, theta = pi/64.
  //
  // HoughTransform:
  //   numangle = (int)(pi / (pi/64))            = 64
  //   numrho   = (int)(((64 + 64) * 2 + 1) / 1) = 257,  offset = (257-1)/2 = 128
  //   theta bin 32 = 32*pi/64 = pi/2 -> cos = 0, sin = 1
  //   r = round(0*j + 1*32) + 128 = 160  ->  accum[32][160] = 64
  //
  // Extrema (fed the accumulator as one grey level per vote) must report exactly that bin,
  // and echo the accumulator size so HoughLines can be wired from its outputs.
  //
  // HoughLines, with w = h = 64, accW = 257, accH = 64:
  //   dTheta    = pi/64,  theta = 32 * pi/64 = pi/2  -> sin = 1, cos = 0
  //   rho_scale = (2*(64+64)+1)/257 = 257/257 = 1
  //   rho       = (160 - (257-1)/2) * 1 = 32
  //   y1 = rho/sin = 32, y2 = (rho - 64*cos)/sin = 32
  // -> the segment (0, 32) -> (64, 32), i.e. the line we drew.
  cv::HoughTransform h;
  Image img{64, 64};
  for(int x = 0; x < 64; ++x)
    img.setGray(x, 32, 255);
  houghRun(h, img, 1.0, pi / 64);

  REQUIRE(h.outputs.acc_width.value == 257);
  REQUIRE(h.outputs.acc_height.value == 64);
  {
    const auto p = argmax(h);
    REQUIRE(p.x == 160);
    REQUIRE(p.y == 32);
    REQUIRE(p.v == Approx(64.f));
  }

  Image acc = accumulatorAsImage(h);
  cv::Extrema ex;
  feed(ex.inputs.image, acc);
  // 30 rather than the default 20: the quantisation of the other angle bins leaves one
  // secondary local max at 21 votes. Anything above that isolates the true line.
  ex.inputs.threshold.value = 30;
  ex.inputs.maxpoints.value = 64;
  ex.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours8;
  // Geometry pass-through, wired from HoughTransform's outlets.
  ex.inputs.theta_step.value = h.outputs.theta_step.value;
  ex.inputs.rho_step.value = h.outputs.rho_step.value;
  ex.inputs.src_width.value = h.outputs.src_width.value;
  ex.inputs.src_height.value = h.outputs.src_height.value;
  ex();

  CHECK(ex.outputs.theta_step.value == Approx(pi / 64));
  CHECK(ex.outputs.rho_step.value == Approx(1.0));
  CHECK(ex.outputs.src_width.value == 64);
  CHECK(ex.outputs.src_height.value == 64);

  REQUIRE(ex.outputs.peaks.value.size() == 1);
  CHECK(ex.outputs.peaks.value[0].x == 160);
  CHECK(ex.outputs.peaks.value[0].y == 32);
  CHECK(ex.outputs.peaks.value[0].value == Approx(64.f).margin(1.f));
  // Wired straight from HoughTransform's own reported size.
  CHECK(ex.outputs.acc_width.value == h.outputs.acc_width.value);
  CHECK(ex.outputs.acc_height.value == h.outputs.acc_height.value);

  cv::HoughLines hl;
  hl.inputs.peaks.value = ex.outputs.peaks.value;
  hl.inputs.width.value = ex.outputs.src_width.value;
  hl.inputs.height.value = ex.outputs.src_height.value;
  hl.inputs.acc_width.value = ex.outputs.acc_width.value;
  hl.inputs.acc_height.value = ex.outputs.acc_height.value;
  hl.inputs.theta_step.value = ex.outputs.theta_step.value;
  hl.inputs.rho_step.value = ex.outputs.rho_step.value;
  hl.inputs.threshold.value = 0;
  hl.inputs.maxlines.value = 64;
  hl();

  REQUIRE(hl.outputs.lines.value.size() == 1);
  const auto& s = hl.outputs.lines.value[0];
  CHECK(s.x1 == Approx(0.f));
  CHECK(s.y1 == Approx(32.f).margin(1e-3));
  CHECK(s.x2 == Approx(64.f));
  CHECK(s.y2 == Approx(32.f).margin(1e-3));

  // And the very same picture through cv::Lines gives the segment's real extent instead of
  // a full-width infinite line: the two paths agree on where the line is.
  cv::Lines lines;
  feed(lines.inputs.image, img);
  linesDefaults(lines);
  lines();
  REQUIRE(!lines.outputs.lines.value.empty());
  for(const auto& seg : lines.outputs.lines.value)
  {
    // Canny turns the 1-pixel bright line into two edges, one on each side of it.
    CHECK(std::abs(seg.y1 - 32.f) <= 1.5f);
    CHECK(seg.y1 == Approx(seg.y2));
  }
}

TEST_CASE("Chain: the cv.jit DEFAULTS on a 640 x 480 frame (the configuration that broke)",
          "[hough][houghtransform][extrema][hough2lines][chain]")
{
  // THE ADVERSARIAL CASE. The other chain test picks theta = pi/64 (so pi/accH happens to
  // be exact) and accW = 257 (odd, so an integer and a float (accW-1)/2 agree). Both of
  // those coincidences hide real bugs. This one uses the shipped defaults on a non-square
  // frame, where neither holds:
  //
  //   theta    = 5 * 0.01745329252 = 0.0872664626 rad
  //   numangle = (int)(pi / theta) = (int)35.999999996 = 35   -> pi/35 != theta
  //   rho      = 4
  //   numrho   = (int)(((640 + 480) * 2 + 1) / 4) = (int)560.25 = 560   -> EVEN
  //   offset   = (560 - 1) / 2 = 279 as an integer (a float would say 279.5)
  //
  // The picture is a 200-pixel horizontal run at y = 100, x in [0, 199]. 200 stays under
  // the 255-vote texture ceiling, so nothing is clamped.
  //
  //   theta bin 18 = 18 * 0.0872664626 = 1.5707963268 rad. cv.jit's DEG constant is a hair
  //   larger than pi/180, so this is pi/2 + 5.1e-12: sin = 1, cos = -5.1034e-12.
  //   r = round((j * -5.1e-12 + 100 * 1) / 4) + 279 = round(25) + 279 = 304 for all 200
  //   pixels  ->  accum[18][304] = 200.
  //
  //   Every other theta bin spreads those 200 votes: consecutive j move the rho index by
  //   |cos(theta_n)| / 4, and the largest |cos| among the neighbours of bin 18 is
  //   sin(0.0872664626) = 0.08715, so at most 4/0.08715 + 1 = 46 pixels can share a bin.
  //   An Extrema threshold of 150 therefore isolates bin (304, 18) exactly.
  //
  // HoughLines with the EXACT steps wired through:
  //   theta = 18 * 0.0872664626 = 1.5707963268 -> sin = 1, cos = -5.1034e-12
  //   rho   = (304 - 279) * 4 = 100            (integer offset, real rho step)
  //   y1 = rho / sin           = 100
  //   y2 = (rho - 640*cos)/sin = 100 + 3.27e-9 = 100
  // -> exactly the line that was drawn.
  //
  // With the steps LEFT UNSET, HoughLines can only fall back to inverting the accumulator
  // size -- which is what it used to do unconditionally -- and the same peak decodes to
  //   dTheta    = pi/35   = 0.08975979010256552, theta = 1.6156762218461793
  //                         sin = 0.9989930665413147, cos = -0.04486483035051486
  //   rho_scale = 2241/560 = 4.001785714285714
  //   rho       = (304 - 279) * 4.001785714285714 = 100.04464285714286
  //   y1 = rho/sin            = 100.14548429...
  //   y2 = (rho - 640*cos)/sin = 128.88790873...
  // i.e. a "horizontal" line TILTED by 28.7 px across the frame. (The rho offset is still
  // the integer 279 even in the fallback -- that half-bin was a separate bug and it is
  // fixed unconditionally; only the STEPS are guessed here.) The numbers are asserted so
  // nobody can quietly make the derivation the primary path again.
  cv::HoughTransform h;
  Image img{640, 480};
  for(int x = 0; x < 200; ++x)
    img.setGray(x, 100, 255);
  feed(h.inputs.image, img);
  h(); // rho and theta stay at their port defaults

  REQUIRE(h.outputs.acc_width.value == 560);
  REQUIRE(h.outputs.acc_height.value == 35);
  REQUIRE(h.outputs.src_width.value == 640);
  REQUIRE(h.outputs.src_height.value == 480);
  REQUIRE(h.outputs.max_count.value == 200); // under the 255 ceiling: nothing is clamped
  {
    const auto p = argmax(h);
    REQUIRE(p.x == 304);
    REQUIRE(p.y == 18);
    REQUIRE(p.v == Approx(200.f));
    REQUIRE(p.ties == 1);
  }

  Image acc = accumulatorAsImage(h);
  cv::Extrema ex;
  feed(ex.inputs.image, acc);
  ex.inputs.threshold.value = 150; // > the 46-vote ceiling of every other theta bin
  ex.inputs.maxpoints.value = 64;
  ex.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours8;
  ex.inputs.theta_step.value = h.outputs.theta_step.value;
  ex.inputs.rho_step.value = h.outputs.rho_step.value;
  ex.inputs.src_width.value = h.outputs.src_width.value;
  ex.inputs.src_height.value = h.outputs.src_height.value;
  ex();

  REQUIRE(ex.outputs.peaks.value.size() == 1);
  CHECK(ex.outputs.peaks.value[0].x == 304);
  CHECK(ex.outputs.peaks.value[0].y == 18);
  CHECK(ex.outputs.peaks.value[0].value == Approx(200.f));
  CHECK(ex.outputs.theta_step.value == Approx(0.0872664626).epsilon(1e-12));
  CHECK(ex.outputs.rho_step.value == Approx(4.0));
  CHECK(ex.outputs.src_width.value == 640);
  CHECK(ex.outputs.src_height.value == 480);

  cv::HoughLines hl;
  hl.inputs.peaks.value = ex.outputs.peaks.value;
  hl.inputs.width.value = ex.outputs.src_width.value;
  hl.inputs.height.value = ex.outputs.src_height.value;
  hl.inputs.acc_width.value = ex.outputs.acc_width.value;
  hl.inputs.acc_height.value = ex.outputs.acc_height.value;
  hl.inputs.theta_step.value = ex.outputs.theta_step.value;
  hl.inputs.rho_step.value = ex.outputs.rho_step.value;
  hl.inputs.threshold.value = 0;
  hl.inputs.maxlines.value = 64;
  hl();

  REQUIRE(hl.outputs.lines.value.size() == 1);
  {
    const auto& s = hl.outputs.lines.value[0];
    CHECK(s.x1 == Approx(0.f));
    CHECK(s.y1 == Approx(100.f).margin(1e-4));
    CHECK(s.x2 == Approx(640.f));
    CHECK(s.y2 == Approx(100.f).margin(1e-4)); // horizontal, as drawn
  }

  // Now unset the steps: HoughLines falls back to the cv.jit-style inversion and gets the
  // documented WRONG answer. Pinning it keeps the fallback honest and the regression loud.
  hl.inputs.theta_step.value = 0.;
  hl.inputs.rho_step.value = 0.;
  hl();
  REQUIRE(hl.outputs.lines.value.size() == 1);
  {
    const auto& s = hl.outputs.lines.value[0];
    CHECK(s.y1 == Approx(100.14548429f).epsilon(1e-6));
    CHECK(s.y2 == Approx(128.88790873f).epsilon(1e-6));
    CHECK(std::abs(s.y2 - s.y1) > 28.f); // a "horizontal" line, tilted by 28.7 px
  }
}
