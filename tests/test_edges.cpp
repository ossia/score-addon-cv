// Tests for the edge/smoothing objects: Canny (cv.jit.canny) and GaussianBlur (cv.jit.blur).
//
// Every assertion below is a hand-computed expectation, so a few numbers are worth spelling
// out once here.
//
// CANNY GRADIENT SCALE (default path, i.e. "Pre-smooth" OFF -- what ships, and what matches
// cv.jit). The pipeline is: Rec.601 luma -> 3x3 Sobel -> |gx| + |gy|. There is deliberately
// NO pre-smoothing stage: cv::Canny, which is what cv.jit.canny calls, does not smooth
// either -- Sobel's own [1 2 1] tap is the only smoothing in the chain. Sobel is
// unnormalised (no /8), exactly as in OpenCV.
//
// Take a vertical step of amplitude A: columns [0,c) are 0, columns [c,W) are A, so the luma
// profile is s(x) = 0 for x < c and s(x) = A for x >= c, constant down each column. On a
// column-constant image the vertical [1 2 1] tap of the gx kernel just sums to 4, and the
// gy kernel differences two identical rows, so
//     gx(x) = 4 * (s(x+1) - s(x-1))        gy(x) = 0
// which is non-zero on exactly the two columns straddling the transition:
//     gx(c-2) = 4 * (s(c-1) - s(c-3)) = 0
//     gx(c-1) = 4 * (s(c)   - s(c-2)) = 4 * (A - 0) = 4A
//     gx(c)   = 4 * (s(c+1) - s(c-1)) = 4 * (A - 0) = 4A
//     gx(c+1) = 4 * (s(c+2) - s(c))   = 0
// so **mag = 4 * A** on columns c-1 and c, exactly tied, and 0 everywhere else.
//     A = 255 -> 1020        A = 60 -> 240
// (Confirmed against the object itself on a 16x10 step: 1020 on columns 7 and 8 for A = 255,
// 240 for A = 60.)
//
// PRE-SMOOTH TOGGLE. With the optional separable [1 2 1]/4 pass ON, the step first becomes
//     s(c-2) = 0, s(c-1) = A/4, s(c) = 3A/4, s(c+1) = A
// (the vertical half of the separable pass is a no-op on a column-constant image), hence
//     mag(c-1) = 4 * (s(c) - s(c-2))   = 4 * 3A/4       = 3A
//     mag(c)   = 4 * (s(c+1) - s(c-1)) = 4 * (A - A/4)  = 3A
// i.e. **mag = 3 * A**: the toggle costs exactly a quarter of the magnitude of an ideal step,
// while leaving the tie -- and therefore the edge location -- alone. 255 -> 765, 60 -> 180.
//
// THRESHOLD CEILING. cv.jit clips `threshold` and `range` to [0,255] and derives
//     low = floor(clamp(threshold - range, 0, 255))  high = floor(clamp(threshold + range, ...))
// so `high` can never exceed 255. Any step with 4A > 255 -- that is, A >= 64 -- is therefore
// *unconditionally* strong on the default path, whatever the attributes. Cases below that
// need a merely-weak ridge use A = 60 (mag = 240 <= 255) rather than A = 64.
// The floors are cv::Canny's (`int low = cvFloor(low_thresh)`); with a gray input every
// magnitude is an integer so they are invisible, but with a coloured input the Rec.601 luma
// -- and hence the magnitude -- is fractional and they bite. See the dedicated test.
//
// TIE-BREAK. The two axis-aligned NMS sectors use `m > previous && m >= next`, so the tied
// plateau {c-1, c} keeps exactly the lower index, c-1. With a transition at column 8 the
// ridge therefore lands on column 7 -- one pixel, deterministically. (Break the tie, as the
// ramped image in the hysteresis test does, and the winner moves accordingly.)
// The DIAGONAL sector is different, and deliberately so: OpenCV writes it with TWO strict
// comparisons, `m > _mag_p[j-s] && m > _mag_n[j+s]`, so a tie along a diagonal kills BOTH
// pixels instead of keeping the first. The 2x2 checkerboard in the degenerate-sizes test is
// exactly that configuration and is the regression guard for it.
//
// BORDER. There is no dead ring. Sobel uses BORDER_REPLICATE -- what cv::Canny passes to it
// -- so on the frame column gx(0) = 4*(s(1) - s(0)) rather than the identically-zero value
// reflect-101 would give, and NMS then runs over every pixel with a virtual 1-pixel ring of
// ZERO magnitude standing in for the missing neighbours. That is OpenCV's own arrangement
// (its magnitude buffer is (rows+2)x(cols+2), ring zeroed, and its NMS loop covers
// j = 0..cols-1 over rows 0..rows-1), and it means:
//   * a step at column c lights column c-1 for every row 0..H-1, borders included;
//   * a step at column 1 lights column ZERO, because mag(0) = mag(1) = 4A ties and the
//     axis-aligned rule keeps the lower index. Excluding the frame would report column 1 --
//     a one-pixel position error, not just a missing pixel.
// Every Canny expectation below was cross-checked against the real cv::Canny from OpenCV
// 4.6 on the identical image and band; all of them match it exactly.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/Canny.hpp>
#include <CV/Cpu/GaussianBlur.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using Catch::Approx;
using namespace cvtest;

namespace
{
// ---- Canny helpers (output is r8: one byte per pixel) --------------------------------------
int edgeCount(const cv::Canny& obj)
{
  auto& o = obj.outputs.image.texture;
  int n = 0;
  for(int i = 0; i < o.width * o.height; ++i)
    n += (o.bytes[i] != 0) ? 1 : 0;
  return n;
}

// Columns of the edge pixels on row y.
std::vector<int> edgeColumns(const cv::Canny& obj, int y)
{
  auto& o = obj.outputs.image.texture;
  std::vector<int> cols;
  for(int x = 0; x < o.width; ++x)
    if(o.bytes[y * o.width + x] != 0)
      cols.push_back(x);
  return cols;
}

// A vertical step: columns [0, c) are 0, columns [c, W) are `amp`.
Image verticalStep(int W, int H, int c, std::uint8_t amp)
{
  Image img(W, H, 0);
  img.fillRect(c, 0, W - c, H, amp);
  return img;
}

// Run Canny on `img` with the given attributes. `presmooth` defaults to false, which is both
// the object's own default and the cv.jit-faithful path; the cases that exercise the toggle
// pass it explicitly.
void runCanny(cv::Canny& obj, Image& img, float threshold, float range, bool presmooth = false)
{
  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = threshold;
  obj.inputs.range.value = range;
  obj.inputs.presmooth.value = presmooth;
  obj();
}

// ---- Blur helpers (output is rgba8) ---------------------------------------------------------
std::uint8_t red(const cv::GaussianBlur& obj, int x, int y)
{
  auto& o = obj.outputs.image.texture;
  return o.bytes[(static_cast<std::size_t>(y) * o.width + x) * 4];
}
std::uint8_t alpha(const cv::GaussianBlur& obj, int x, int y)
{
  auto& o = obj.outputs.image.texture;
  return o.bytes[(static_cast<std::size_t>(y) * o.width + x) * 4 + 3];
}
}

// ==================================================================================== Canny

TEST_CASE("Canny locates a vertical step edge on the expected column", "[canny]")
{
  // Transition at column 8, A = 255 -> mag = 4*255 = 1020 on columns 7 and 8, far above the
  // default band [140,160]; the tied plateau {7,8} resolves to column 7.
  //
  // Rows 0 and 9 are edges too. Replicate borders make the top row's vertical [1 2 1] tap
  // read (row 0, row 0, row 1) instead of (row 1, row 0, row 1), but the image is constant
  // down every column, so both spellings give the same gx = 4*(s(x+1) - s(x-1)) and gy = 0.
  // The frame therefore carries the identical ridge, and NMS -- which now runs over it --
  // keeps it. ALL TEN rows, not eight.
  cv::Canny obj;
  Image img = verticalStep(16, 10, 8, 255);
  runCanny(obj, img, 150.f, 10.f);

  auto& o = obj.outputs.image.texture;
  REQUIRE(o.bytes != nullptr);
  REQUIRE(o.width == 16);
  REQUIRE(o.height == 10);

  CHECK(edgeCount(obj) == 10);
  for(int y = 0; y <= 9; ++y)
  {
    INFO("row " << y);
    auto cols = edgeColumns(obj, y);
    REQUIRE(cols.size() == 1u);
    CHECK(cols[0] == 7);
  }
}

TEST_CASE("Canny puts a near-frame edge on the frame itself", "[canny]")
{
  // The probe that pins BOTH the replicate Sobel border and the full-image NMS. 8x8, column
  // 0 black, columns 1..7 white: s(0) = 0, s(x >= 1) = 255.
  //
  // Under BORDER_REPLICATE, gx(x) = 4*(s(x+1) - s(x-1)) with s(-1) := s(0):
  //     gx(0) = 4*(s(1) - s(0))  = 4*255 = 1020
  //     gx(1) = 4*(s(2) - s(0))  = 4*255 = 1020
  //     gx(2) = 4*(s(3) - s(1))  = 0
  // gy = 0 everywhere (all rows identical), so mag = 1020 on columns 0 AND 1, tied, and 0
  // elsewhere. The horizontal NMS rule `m > left && m >= right` reads the zero ring at
  // column -1, so column 0 wins (1020 > 0 and 1020 >= 1020) and column 1 loses
  // (1020 > 1020 is false). Answer: column 0, all 8 rows.
  //
  // Both of the old behaviours got this wrong, and wrong in the same direction: reflect-101
  // makes gx(0) = 4*(s(1) - s(1)) = 0 so the frame is blank and column 1 wins by default,
  // and force-zeroing the frame after NMS also leaves column 1. Either way the edge is
  // reported one pixel too far inside the image. Verified against cv::Canny (OpenCV 4.6):
  // it lights column 0 on all 8 rows.
  cv::Canny obj;
  Image img(8, 8, 255);
  for(int y = 0; y < 8; ++y)
    img.setGray(0, y, 0);
  runCanny(obj, img, 150.f, 10.f);

  CHECK(edgeCount(obj) == 8);
  for(int y = 0; y < 8; ++y)
  {
    INFO("row " << y);
    CHECK(edgeColumns(obj, y) == std::vector<int>{0});
  }
}

TEST_CASE("Canny floors the hysteresis band, as cvFloor does", "[canny]")
{
  // cv::Canny does `int low = cvFloor(low_thresh); int high = cvFloor(high_thresh);` before
  // it compares anything. On a gray image that is unobservable -- every magnitude is an
  // integer, and for integral m the predicates `m > 239.5` and `m > 239` coincide -- so this
  // case uses a COLOURED step, where the Rec.601 luma is fractional and so is the magnitude.
  //
  // Step from black to pure red (200, 0, 0): luma = 0.299 * 200 = 59.8, so on the two
  // columns straddling the transition mag = 4 * 59.8 = 239.2 exactly (gy = 0, the image is
  // column-constant), and the tie resolves to column 7 as usual.
  //
  //   threshold 239.5, range 0 -> band floors to [239, 239]: 239.2 > 239 -> strong -> edge
  //   threshold 240.0, range 0 -> band is [240, 240]:        239.2 < 240 -> not a candidate
  //
  // Without the floor the first case would compare 239.2 > 239.5 and find nothing at all, so
  // this pair brackets the behaviour from both sides: it fails if the floor is missing, and
  // it also fails if the floor is replaced by a round or a ceil.
  Image img(16, 6, 0);
  for(int y = 0; y < 6; ++y)
    for(int x = 8; x < 16; ++x)
      img.set(x, y, 200, 0, 0);

  {
    cv::Canny obj;
    runCanny(obj, img, 239.5f, 0.f);
    CHECK(edgeCount(obj) == 6); // 6 rows, frame included
    for(int y = 0; y < 6; ++y)
      CHECK(edgeColumns(obj, y) == std::vector<int>{7});
  }
  {
    cv::Canny obj;
    runCanny(obj, img, 240.f, 0.f);
    CHECK(edgeCount(obj) == 0);
  }
  {
    // ...and the same band one integer lower does see it, so the case above is a genuine
    // threshold crossing rather than an image with no gradient in it.
    cv::Canny obj;
    runCanny(obj, img, 239.f, 0.f);
    CHECK(edgeCount(obj) == 6);
  }
}

TEST_CASE("Canny NMS thins the ridge to exactly one pixel", "[canny]")
{
  // Explicitly asymmetric profile: the transition pixel itself is at half value, so the
  // gradient has a unique (not tied) maximum. With s = (..., 0, 128, 255, 255, ...) and
  // gx(x) = 4*(s(x+1) - s(x-1)):
  //     mag(6) = 4*(128 -   0) = 512
  //     mag(7) = 4*(255 -   0) = 1020
  //     mag(8) = 4*(255 - 128) = 508
  // NMS must therefore keep column 7 and drop its immediate neighbours. The image is
  // column-constant, so replicate borders change nothing and all ten rows behave alike.
  cv::Canny obj;
  Image img = verticalStep(16, 10, 8, 255);
  for(int y = 0; y < 10; ++y)
    img.setGray(7, y, 128);
  runCanny(obj, img, 150.f, 10.f);

  auto& o = obj.outputs.image.texture;
  CHECK(edgeCount(obj) == 10);
  for(int y = 0; y <= 9; ++y)
  {
    auto cols = edgeColumns(obj, y);
    REQUIRE(cols.size() == 1u); // one pixel wide, not a 2- or 3-px band
    CHECK(cols[0] == 7);
    CHECK(o.bytes[y * 16 + 6] == 0);
    CHECK(o.bytes[y * 16 + 8] == 0);
  }
}

TEST_CASE("Canny finds nothing in a flat image", "[canny]")
{
  for(std::uint8_t v : {std::uint8_t{0}, std::uint8_t{60}, std::uint8_t{255}})
  {
    cv::Canny obj;
    Image img(12, 12, v);
    runCanny(obj, img, 150.f, 10.f);
    INFO("flat value " << int(v));
    CHECK(edgeCount(obj) == 0);
  }
}

TEST_CASE("Canny output is strictly binary 0 or 255", "[canny]")
{
  cv::Canny obj;
  Image img(24, 24, 0);
  img.fillRect(4, 4, 9, 9, 255);
  img.fillRect(14, 14, 6, 6, 128);
  runCanny(obj, img, 150.f, 10.f);

  auto& o = obj.outputs.image.texture;
  REQUIRE(o.bytes);
  int edges = 0;
  for(int i = 0; i < o.width * o.height; ++i)
  {
    const std::uint8_t v = o.bytes[i];
    REQUIRE((v == 0 || v == 255));
    edges += (v == 255);
  }
  CHECK(edges > 0); // and the image does contain edges
}

TEST_CASE("Canny threshold gates weak edges", "[canny]")
{
  // A = 60 -> mag = 4*60 = 240; A = 255 -> mag = 4*255 = 1020. The image is column-constant,
  // so all 12 rows -- frame included -- carry the same ridge.
  {
    cv::Canny obj; // default band [140,160]: 240 > 160 -> strong
    Image img = verticalStep(16, 12, 8, 60);
    runCanny(obj, img, 150.f, 10.f);
    CHECK(edgeCount(obj) == 12); // all 12 rows
  }
  {
    cv::Canny obj; // band [245,255]: 240 < 245 -> below low, nothing survives
    Image img = verticalStep(16, 12, 8, 60);
    runCanny(obj, img, 250.f, 5.f);
    CHECK(edgeCount(obj) == 0);
  }
  {
    cv::Canny obj; // same high threshold, but 1020 > 255 -> still detected
    Image img = verticalStep(16, 12, 8, 255);
    runCanny(obj, img, 250.f, 5.f);
    CHECK(edgeCount(obj) == 12);
  }
}

TEST_CASE("Canny range widens the hysteresis band", "[canny]")
{
  // A = 60 -> mag = 240. high = threshold + range, so growing `range` with `threshold` fixed
  // raises the bar an *unseeded* edge must clear.
  {
    cv::Canny obj; // band [180,190]: 240 > 190 -> strong
    Image img = verticalStep(16, 12, 8, 60);
    runCanny(obj, img, 185.f, 5.f);
    CHECK(edgeCount(obj) == 12);
  }
  {
    cv::Canny obj; // band [115,255]: 240 is now only *weak*, and has no seed -> dropped
    Image img = verticalStep(16, 12, 8, 60);
    runCanny(obj, img, 185.f, 70.f);
    CHECK(edgeCount(obj) == 0);
  }
  {
    cv::Canny obj; // degenerate band [250,250]: 240 < 250 -> not even a candidate
    Image img = verticalStep(16, 12, 8, 60);
    runCanny(obj, img, 250.f, 0.f);
    CHECK(edgeCount(obj) == 0);
  }
}

TEST_CASE("Canny hysteresis links a weak edge to a strong seed", "[canny]")
{
  // A vertical step at column 8 (left half 0, right half v) whose amplitude ramps *exactly*
  // linearly down a 16x30 image: v(y) = 255 - 8y, so v(0) = 255 and v(29) = 23.
  //
  // Exact linearity is what makes this hand-computable: v(y-1) + 2v(y) + v(y+1) = 4v(y) and
  // v(y+1) - v(y-1) = -16 for every interior row, with no rounding wobble. Writing the 3x3
  // taps out per column (columns 0..6 are 0, columns 8..15 carry v(y)):
  //   col 6:  gx = (col 7) - (col 5) = 0,      gy = 0                     -> mag = 0
  //   col 7:  gx = (col 8) - (col 6) = 4v(y),  gy = v(y+1) - v(y-1) = -16 -> mag = 4v(y) + 16
  //   col 8:  gx = (col 9) - (col 7) = 4v(y),  gy = 3(v(y+1)-v(y-1)) = -48
  //                                                                      -> mag = 4v(y) + 48
  //   cols 9..14: gx = 0 (both flanking columns carry v), gy = 4(v(y+1)-v(y-1)) = -64
  //                                                                      -> mag = 64
  //
  // Two consequences, both deliberate:
  //  * mag(8) = mag(7) + 32 > mag(7) for every row, so the ramp *breaks* the tie a constant
  //    step would produce: NMS keeps column 8 and drops column 7. The ridge stays 1px wide.
  //  * The right half is flat horizontally but ramping vertically, so it carries a genuine
  //    vertical gradient of magnitude 64. That is the trap: with low = 0 those pixels become
  //    weak candidates and hysteresis floods them from the ridge, smearing whole rows. So
  //    `low` is set to 100 here, which puts the entire ramp plateau (64) below it -- it never
  //    becomes a candidate at all -- while leaving the ridge untouched.
  //
  // THE TWO FRAME ROWS. Replicate borders make row -1 read as row 0 and row 30 as row 29, so
  // the vertical difference there is halved and the [1 2 1] column sum is no longer 4v:
  //   y = 0 : col 8 gx = v(-1) + 2v(0) + v(1) = 255 + 510 + 247 = 1012, gy = 3(v(1) - v(0))
  //           = -24            -> mag(8,0) = 1036;  col 7 gy = v(1) - v(0) = -8, gx = 1012
  //                                                          -> mag(7,0) = 1020
  //   y = 29: v(29) = 23, v(30) := v(29) = 23, v(28) = 31
  //           col 8 gx = 31 + 46 + 23 = 100, gy = 3(23 - 31) = -24 -> mag(8,29) = 124
  //           col 7 gx = 100,                gy = 23 - 31   = -8  -> mag(7,29) = 108
  // In both frame rows mag(8) still exceeds mag(7), so NMS keeps column 8 there as well, and
  // the ramp plateau on columns 9..14 drops to 4*(v(y+1) - v(y)) = 32 -- further below
  // low = 100 than the interior's 64. So the frame adds two more ridge pixels and no noise.
  //
  // Ridge magnitude: mag(8,y) = 4(255 - 8y) + 48 = 1068 - 32y for y in 1..28, i.e. 1036 at
  // y=1 down to 172 at y=28, plus 1036 at y=0 and 124 at y=29 from the derivation above.
  // It crosses high = 255 at 1068 - 32y > 255 <=> y < 25.4, so rows 0..25 are strong seeds
  // and rows 26, 27, 28, 29 (mag 236, 204, 172, 124) are merely weak -- above low = 100,
  // below high = 255 -- and can only appear via 8-connected propagation from the chain above.
  Image img(16, 30, 0);
  for(int y = 0; y < 30; ++y)
    for(int x = 8; x < 16; ++x)
      img.setGray(x, y, static_cast<std::uint8_t>(255 - 8 * y));

  {
    cv::Canny obj;
    runCanny(obj, img, 177.5f, 77.5f); // low = 100, high = 255
    CHECK(edgeCount(obj) == 30);       // all 30 rows, nothing else
    for(int y = 0; y <= 29; ++y)
    {
      INFO("row " << y);
      auto cols = edgeColumns(obj, y);
      REQUIRE(cols.size() == 1u); // one pixel wide -- the ramp plateau must not leak in
      CHECK(cols[0] == 8);
    }
  }
  {
    // The control: same image, same high = 255, but low raised to 240. Rows 26..29 (236, 204,
    // 172, 124) now fall below `low`, so they are not candidates and cannot be promoted; the
    // 26 strong rows are unaffected. That the answer drops from 30 to exactly 26 is what
    // proves those four rows came from hysteresis rather than from clearing `high` on their
    // own.
    cv::Canny obj;
    runCanny(obj, img, 247.5f, 7.5f); // low = 240, high = 255
    CHECK(edgeCount(obj) == 26);
    for(int y = 0; y <= 25; ++y)
      CHECK(edgeColumns(obj, y) == std::vector<int>{8});
    for(int y = 26; y <= 29; ++y)
      CHECK(edgeColumns(obj, y).empty());
  }
}

TEST_CASE("Canny drops an isolated weak edge", "[canny]")
{
  // Same image, two bands. A = 60 -> mag = 240 throughout, which is <= 255 and can therefore
  // still be made merely-weak (an A >= 64 step could not: 4A > 255 = the ceiling on `high`).
  Image img = verticalStep(16, 12, 8, 60);
  {
    cv::Canny obj; // band [0,255]: 240 is weak everywhere, no seed anywhere -> nothing
    runCanny(obj, img, 127.5f, 127.5f);
    CHECK(edgeCount(obj) == 0);
  }
  {
    cv::Canny obj; // band [0,0]: the very same ridge is now strong -> fully detected.
    runCanny(obj, img, 0.f, 0.f);
    CHECK(edgeCount(obj) == 12); // proves the ridge existed; only the seeding differed
  }
}

TEST_CASE("Canny pre-smooth attenuates the gradient by exactly a quarter", "[canny]")
{
  // Same image, same attributes, toggle flipped: mag = 4A off, 3A on. With A = 60 that is
  // 240 vs 180, so a band placed strictly between the two must see the edge with the toggle
  // off and nothing at all with it on. This is the defining property of the toggle: it is
  // the only reason it is not cv.jit-faithful, so it is worth pinning numerically.
  Image img = verticalStep(16, 12, 8, 60);
  {
    cv::Canny obj; // degenerate band [210,210]: 240 > 210 -> strong
    runCanny(obj, img, 210.f, 0.f, /* presmooth */ false);
    CHECK(edgeCount(obj) == 12); // all 12 rows
    for(int y = 0; y <= 11; ++y)
      CHECK(edgeColumns(obj, y) == std::vector<int>{7});
  }
  {
    cv::Canny obj; // 180 < 210 -> not even a weak candidate; the edge disappears entirely
    runCanny(obj, img, 210.f, 0.f, /* presmooth */ true);
    CHECK(edgeCount(obj) == 0);
  }
  {
    // ...and that is attenuation, not destruction: drop the band below 180 and the smoothed
    // ridge comes back in full, on the same rows and the same column.
    cv::Canny obj; // band [150,160]: 180 > 160 -> strong
    runCanny(obj, img, 155.f, 5.f, /* presmooth */ true);
    CHECK(edgeCount(obj) == 12);
    for(int y = 0; y <= 11; ++y)
      CHECK(edgeColumns(obj, y) == std::vector<int>{7});
  }
}

TEST_CASE("Canny pre-smooth does not move a clean strong edge", "[canny]")
{
  // A full 0->255 step clears `high` either way (1020 off, 765 on), and the [1 2 1]/4 pass is
  // symmetric about the transition, so columns 7 and 8 stay exactly tied in both cases and
  // NMS keeps the lower index in both. The two edge maps must be byte-for-byte identical.
  //
  // Pre-smoothing also spills a *flanking* pair the unsmoothed image does not have: with
  // s' = (..., 0, 63.75, 191.25, 255, ...) around the transition, mag(6) = 4*63.75 = 255 and
  // mag(9) = 4*(255 - 191.25) = 255. Those clear high = 160, but NMS kills them -- each has
  // the 765-magnitude ridge as its "next" neighbour -- so they never reach the output and
  // the equality below is not an accident of thresholding.
  Image img = verticalStep(16, 10, 8, 255);
  auto run = [&](bool presmooth) {
    cv::Canny obj;
    runCanny(obj, img, 150.f, 10.f, presmooth);
    auto& o = obj.outputs.image.texture;
    return std::vector<std::uint8_t>(o.bytes, o.bytes + o.width * o.height);
  };
  const auto off = run(false);
  const auto on = run(true);
  CHECK(off == on);

  // ...and the answer they agree on is the expected one: column 7, on all ten rows.
  int n = 0;
  for(int y = 0; y < 10; ++y)
    for(int x = 0; x < 16; ++x)
      if(off[y * 16 + x] != 0)
      {
        ++n;
        INFO("edge at " << x << "," << y);
        CHECK(x == 7);
      }
  CHECK(n == 10);
}

TEST_CASE("Canny ignores an unchanged frame", "[canny]")
{
  {
    cv::Canny obj;
    Image img = verticalStep(16, 10, 8, 255);
    feed(obj.inputs.image, img);
    obj.inputs.image.texture.changed = false;
    obj();
    CHECK(obj.outputs.image.texture.width == 0);
    CHECK(obj.outputs.image.texture.bytes == nullptr);
  }
  {
    // A previously produced output must survive an unchanged frame untouched.
    cv::Canny obj;
    Image img = verticalStep(16, 10, 8, 255);
    runCanny(obj, img, 150.f, 10.f);
    const int before = edgeCount(obj);
    REQUIRE(before == 10);

    Image other(16, 10, 0); // completely different content...
    other.fillRect(0, 0, 16, 5, 255);
    feed(obj.inputs.image, other);
    obj.inputs.image.texture.changed = false; // ...but not flagged as changed
    obj();
    CHECK(edgeCount(obj) == before);
    for(int y = 0; y <= 9; ++y)
      CHECK(edgeColumns(obj, y) == std::vector<int>{7});
  }
}

TEST_CASE("Canny handles degenerate image sizes", "[canny]")
{
  // Checkerboard, I(x,y) = 255 when (x + y) is odd. With replicate borders and a zero
  // magnitude ring these are all fully hand-computable, and two of them are the sharpest
  // regression guards in the suite. Every map below was confirmed against cv::Canny 4.6.
  //
  //   1x1  ->  "."      the only pixel is 0 and every neighbour replicates to it, so
  //                     gx = gy = 0.
  //
  //   1x8  ->  rows 0 and 7 only. W = 1 collapses gx to 0 for every pixel, and
  //            gy = 4*(v(y+1) - v(y-1)) with v(-1) := v(0) and v(8) := v(7):
  //              y=0: 4*(255 - 0) = 1020    y=7: 4*(255 - 0) = 1020
  //              1 <= y <= 6: v(y+1) = v(y-1) (same parity) -> 0
  //            The vertical NMS rule then keeps both, reading zeros from the ring above row
  //            0 and below row 7.  8x1 is the same by transposition.
  //
  //   2x2  ->  EMPTY, and this is the diagonal-strictness guard. All four pixels come out
  //            with |gx| = |gy| = 510, i.e. mag = 1020 everywhere, a perfect DIAGONAL
  //            plateau. Worked for (0,0): gx = (I(1,0) + 2I(1,0) + I(1,1)) - (I(0,0) +
  //            2I(0,0) + I(0,1)) = (255 + 510 + 0) - (0 + 0 + 255) = 510, and gy = 510 by
  //            symmetry. |gy| is between tan(22.5)|gx| and tan(67.5)|gx|, so the diagonal
  //            branch applies, and it is the one OpenCV writes with TWO strict comparisons.
  //            Each pixel therefore needs to beat BOTH diagonal neighbours strictly; each
  //            has one neighbour in the zero ring (fine) and one that is another 1020 (not
  //            fine). Nothing survives.
  //              *** A "symmetrised" `m > a && m >= b` diagonal -- the textbook spelling --
  //              *** lights row 0 here (##/..). That is precisely the bug this guards.
  //
  //   2x3  ->  rows 0 and 2, both columns. Row 1's two pixels have gx = gy = 0 (the
  //            [1 2 1] taps cancel across the replicate-doubled columns), so the four
  //            corner-ish pixels each face the zero ring on one diagonal and a 0-magnitude
  //            pixel on the other, and all four survive.
  //
  //   3x3  ->  the four corners only. The centre and the four edge midpoints all have
  //            gx = gy = 0 (opposite taps are equal), leaving mag = 1020 at the corners
  //            with a zero on both diagonal neighbours -- kept.
  const std::vector<std::pair<std::pair<int, int>, std::string>> cases{
      {{1, 1}, "."},
      {{2, 2}, ".."
               ".."},
      {{1, 8}, "#......#"},   // column image, written transposed
      {{8, 1}, "#......#"},
      {{2, 3}, "##"
               ".."
               "##"},
      {{3, 3}, "#.#"
               "..."
               "#.#"}};

  for(const auto& [wh, expect] : cases)
  {
    const auto [w, h] = wh;
    cv::Canny obj;
    Image img(w, h, 0);
    for(int y = 0; y < h; ++y)
      for(int x = 0; x < w; ++x)
        img.setGray(x, y, static_cast<std::uint8_t>((x + y) % 2 ? 255 : 0));
    runCanny(obj, img, 150.f, 10.f);

    INFO(w << "x" << h);
    auto& o = obj.outputs.image.texture;
    REQUIRE(o.width == w);
    REQUIRE(o.height == h);
    REQUIRE(o.bytes != nullptr);
    REQUIRE(expect.size() == static_cast<std::size_t>(w * h));

    std::string got;
    for(int i = 0; i < w * h; ++i)
    {
      REQUIRE((o.bytes[i] == 0 || o.bytes[i] == 255));
      got += (o.bytes[i] ? '#' : '.');
    }
    CHECK(got == expect);
  }
}

// ============================================================================= GaussianBlur

TEST_CASE("GaussianBlur kernel is normalised: a uniform image is unchanged", "[blur]")
{
  // The strongest border-handling test there is: if reflect-101 indexing or the kernel
  // normalisation is wrong, the edges of a constant image drift away from the constant.
  for(int r : {1, 2, 5, 9})
  {
    for(int n : {1, 2, 3, 6, 17})
    {
      cv::GaussianBlur obj;
      Image img(n, n, 137);
      feed(obj.inputs.image, img);
      obj.inputs.radius.value = r;
      obj.inputs.sigma.value = 0.f;
      obj();

      INFO("radius " << r << ", size " << n);
      auto& o = obj.outputs.image.texture;
      REQUIRE(o.width == n);
      REQUIRE(o.height == n);
      for(int y = 0; y < n; ++y)
        for(int x = 0; x < n; ++x)
        {
          REQUIRE(red(obj, x, y) == 137);
          REQUIRE(alpha(obj, x, y) == 255);
        }
    }
  }
}

TEST_CASE("GaussianBlur spreads a single bright pixel symmetrically", "[blur]")
{
  cv::GaussianBlur obj;
  Image img(11, 11, 0);
  img.set(5, 5, 255, 255, 255);
  feed(obj.inputs.image, img);
  obj.inputs.radius.value = 2; // k = 5, sigma = 1.1
  obj();

  // The centre stays the maximum and is strictly darker than the original impulse.
  const int c = red(obj, 5, 5);
  CHECK(c > 0);
  CHECK(c < 255);
  for(int y = 0; y < 11; ++y)
    for(int x = 0; x < 11; ++x)
      REQUIRE(red(obj, x, y) <= c);

  // Four-fold symmetry about the impulse.
  for(int dy = 0; dy <= 5; ++dy)
    for(int dx = 0; dx <= 5; ++dx)
    {
      const int v = red(obj, 5 + dx, 5 + dy);
      REQUIRE(red(obj, 5 - dx, 5 + dy) == v);
      REQUIRE(red(obj, 5 + dx, 5 - dy) == v);
      REQUIRE(red(obj, 5 - dx, 5 - dy) == v);
      REQUIRE(red(obj, 5 + dy, 5 + dx) == v); // separable -> also transpose-symmetric
    }

  // Monotonically decreasing away from the centre, and zero outside the kernel support
  // (radius 2 in each axis).
  CHECK(red(obj, 5, 5) > red(obj, 6, 5));
  CHECK(red(obj, 6, 5) > red(obj, 7, 5));
  CHECK(red(obj, 8, 5) == 0);
  CHECK(red(obj, 5, 8) == 0);

  // Energy is conserved up to per-pixel rounding (25 taps, each losing at most 0.5).
  int sum = 0;
  for(int y = 0; y < 11; ++y)
    for(int x = 0; x < 11; ++x)
      sum += red(obj, x, y);
  CHECK(sum == Approx(255).margin(15));
}

TEST_CASE("GaussianBlur spreads further with a larger radius", "[blur]")
{
  int nonzero[4] = {};
  for(int r = 1; r <= 3; ++r)
  {
    cv::GaussianBlur obj;
    Image img(15, 15, 0);
    img.set(7, 7, 255, 255, 255);
    feed(obj.inputs.image, img);
    obj.inputs.radius.value = r;
    obj();

    int n = 0;
    for(int y = 0; y < 15; ++y)
      for(int x = 0; x < 15; ++x)
        n += (red(obj, x, y) != 0) ? 1 : 0;
    nonzero[r] = n;

    // Support is exactly the (2r+1)^2 box around the impulse.
    CHECK(red(obj, 7 + r, 7) > 0);
    CHECK(red(obj, 7 + r + 1, 7) == 0);
  }
  CHECK(nonzero[1] == 9);  // 3x3
  CHECK(nonzero[2] == 25); // 5x5
  CHECK(nonzero[3] == 45); // 7x7 minus the four corners, which round to 0
  CHECK(nonzero[1] < nonzero[2]);
  CHECK(nonzero[2] < nonzero[3]);
}

TEST_CASE("GaussianBlur uses OpenCV's fixed table for small kernels", "[blur]")
{
  // THE POINT OF THIS TEST. getGaussianKernel() has two branches and consults them in this
  // order (modules/imgproc/src/smooth.dispatch.cpp):
  //     n odd && n <= 7 && sigma <= 0  ->  return small_gaussian_tab[n>>1] verbatim
  //     otherwise                      ->  exp(-x^2 / 2 sigma^2), sigma from the formula
  //                                        sigma = 0.3*((n-1)*0.5 - 1) + 0.8 if sigma <= 0
  // so for k in {1,3,5,7} -- which is cv.jit.blur's DEFAULT, radius 1 -> k = 3 -- the sigma
  // formula is never evaluated. Asserting the formula alone (as this case used to) checks a
  // helper against itself and cannot see the difference; the discriminator has to be the
  // kernel that actually gets convolved.
  //
  // The cleanest read-out of the centre tap w0 is the impulse response at the impulse: the
  // blur is separable, so out(0,0) = round(255 * w0 * w0). Both branches, hand-computed:
  //
  //   k=3   table  w0 = 1/2         -> 255 * 0.25       = 63.75  -> 64
  //         formula sigma = 0.8, taps exp(0), exp(-1/1.28) = 1, 0.4578330;
  //         sum = 1.9156660, w0 = 0.5220108 -> 255 * 0.2724953 = 69.49 -> 69
  //   k=5   table  w0 = 3/8         -> 255 * 0.140625   = 35.86  -> 36
  //         formula sigma = 1.1, taps 1, 0.6615008, 0.1915450; sum = 2.7060916,
  //         w0 = 0.3695366 -> 255 * 0.1365573 = 34.82 -> 35
  //   k=7   table  w0 = 0.28125     -> 255 * 0.0791016  = 20.17  -> 20
  //         formula sigma = 1.4, taps 1, 0.7783125, 0.3669297, 0.1039993;
  //         sum = 3.4984830, w0 = 0.2858385 -> 255 * 0.0817036 = 20.83 -> 21
  //   k=11  NO table (11 > 7) -> formula, sigma = 2.0, taps 1, 0.8824969, 0.6065307,
  //         0.3246525, 0.1353353, 0.0439369; sum = 4.9859046, w0 = 0.2005653
  //                                          -> 255 * 0.0402264 = 10.26 -> 10
  //
  // Every table row is a power-of-two-denominator kernel -- {1}, {1,2,1}/4, {1,4,6,4,1}/16,
  // {1,7,14,18,14,7,1}/64 -- and each sums to exactly 1, so no normalisation runs on them.
  auto impulseCentre = [](int radius, float sigma) {
    cv::GaussianBlur obj;
    Image img(21, 21, 0);
    img.set(10, 10, 255, 255, 255);
    feed(obj.inputs.image, img);
    obj.inputs.radius.value = radius;
    obj.inputs.sigma.value = sigma;
    obj();
    return int(red(obj, 10, 10));
  };

  // sigma == 0, k <= 7: the TABLE, not the formula. If the formula leaked back in these
  // would read 69 / 35 / 21 instead.
  CHECK(impulseCentre(1, 0.f) == 64); // k = 3, the shipped default
  CHECK(impulseCentre(2, 0.f) == 36); // k = 5
  CHECK(impulseCentre(3, 0.f) == 20); // k = 7
  // ...and asking for that same sigma explicitly takes the formula branch and DOES give the
  // other number, which is what proves the two branches are distinguishable at all.
  CHECK(impulseCentre(1, cv::gaussian_auto_sigma(3)) == 69);
  CHECK(impulseCentre(2, cv::gaussian_auto_sigma(5)) == 35);
  CHECK(impulseCentre(3, cv::gaussian_auto_sigma(7)) == 21);

  // k = 11 is past the table, so sigma == 0 falls through to the formula and the two agree.
  CHECK(impulseCentre(5, 0.f) == 10);
  CHECK(impulseCentre(5, cv::gaussian_auto_sigma(11)) == 10);

  // The formula itself, for the sizes where it is actually reached.
  CHECK(cv::gaussian_auto_sigma(3) == Approx(0.8f));
  CHECK(cv::gaussian_auto_sigma(5) == Approx(1.1f));
  CHECK(cv::gaussian_auto_sigma(7) == Approx(1.4f));
  CHECK(cv::gaussian_auto_sigma(11) == Approx(2.0f)); // radius 5: the first size that uses it

  // Same thing on the object's default, seen on a step rather than an impulse -- this is the
  // number a user notices. Columns [0,8) black, [8,16) white; the image is constant down
  // every column so the vertical pass is a no-op and the 2D result equals the 1D one:
  //     out(7) = 0.25*s(6) + 0.5*s(7) + 0.25*s(8) = 0.25*255            = 63.75 -> 64
  //     out(8) = 0.25*s(7) + 0.5*s(8) + 0.25*s(9) = 0.75*255            = 191.25 -> 191
  // With the sigma-0.8 kernel instead: out(8) = (0.5220108 + 0.2389946)*255 = 194.06 -> 194,
  // i.e. 3 levels too bright. cv::GaussianBlur 4.6 returns 64 / 191 on this exact input.
  {
    cv::GaussianBlur obj;
    Image img(16, 4, 0);
    img.fillRect(8, 0, 8, 4, 255);
    feed(obj.inputs.image, img);
    obj.inputs.radius.value = 1; // the default
    obj.inputs.sigma.value = 0.f;
    obj();
    for(int y = 0; y < 4; ++y)
    {
      INFO("row " << y);
      CHECK(int(red(obj, 6, y)) == 0);
      CHECK(int(red(obj, 7, y)) == 64);
      CHECK(int(red(obj, 8, y)) == 191);
      CHECK(int(red(obj, 9, y)) == 255);
    }
  }

  // And sigma == 0 must be "let OpenCV choose", never "no blur": for a kernel size past the
  // table it is byte-identical to passing the derived sigma explicitly, and a different
  // sigma at the same radius gives a different image, so neither check is vacuous.
  Image img(21, 21, 0);
  img.fillRect(6, 6, 7, 7, 200);
  img.set(3, 16, 255, 30, 90);

  auto blur = [&](int radius, float sigma) {
    cv::GaussianBlur obj;
    feed(obj.inputs.image, img);
    obj.inputs.radius.value = radius;
    obj.inputs.sigma.value = sigma;
    obj();
    auto& o = obj.outputs.image.texture;
    return std::vector<std::uint8_t>(o.bytes, o.bytes + o.width * o.height * 4);
  };

  CHECK(blur(5, 0.f) == blur(5, cv::gaussian_auto_sigma(11))); // k = 11: formula both ways
  CHECK(blur(5, 4.f) != blur(5, 0.f));
  // ...whereas at k = 5 they must DIFFER, because sigma == 0 takes the table.
  CHECK(blur(2, 0.f) != blur(2, cv::gaussian_auto_sigma(5)));
}

TEST_CASE("GaussianBlur preserves alpha and treats channels independently", "[blur]")
{
  cv::GaussianBlur obj;
  Image img(9, 9, 0);
  for(int y = 0; y < 9; ++y)
    for(int x = 0; x < 9; ++x)
      img.set(x, y, 0, 0, 0, static_cast<std::uint8_t>(x * 9 + y)); // 0..80, all distinct
  img.set(4, 4, 255, 0, 0, 200);                                    // pure red impulse

  feed(obj.inputs.image, img);
  obj.inputs.radius.value = 2;
  obj();

  auto& o = obj.outputs.image.texture;
  for(int y = 0; y < 9; ++y)
    for(int x = 0; x < 9; ++x)
    {
      // Alpha passes through byte-for-byte -- it is never blurred.
      REQUIRE(alpha(obj, x, y) == img.px[(y * 9 + x) * 4 + 3]);
      // Green and blue had no signal at all, so they must stay exactly 0.
      const auto* p = o.bytes + (static_cast<std::size_t>(y) * 9 + x) * 4;
      REQUIRE(p[1] == 0);
      REQUIRE(p[2] == 0);
    }
  CHECK(red(obj, 4, 4) > 0);
  CHECK(red(obj, 5, 4) > 0);
}

TEST_CASE("GaussianBlur ignores an unchanged frame", "[blur]")
{
  {
    cv::GaussianBlur obj;
    Image img(8, 8, 200);
    feed(obj.inputs.image, img);
    obj.inputs.image.texture.changed = false;
    obj();
    CHECK(obj.outputs.image.texture.width == 0);
    CHECK(obj.outputs.image.texture.bytes == nullptr);
  }
  {
    cv::GaussianBlur obj;
    Image img(8, 8, 0);
    img.fillRect(2, 2, 4, 4, 240);
    feed(obj.inputs.image, img);
    obj.inputs.radius.value = 1;
    obj();
    const int before = red(obj, 4, 4);
    REQUIRE(before > 0);

    Image other(8, 8, 10); // different content, not flagged changed
    feed(obj.inputs.image, other);
    obj.inputs.image.texture.changed = false;
    obj();
    CHECK(red(obj, 4, 4) == before);
  }
}

TEST_CASE("GaussianBlur handles degenerate image sizes", "[blur]")
{
  for(auto [w, h] : {std::pair{1, 1}, std::pair{2, 2}, std::pair{1, 5}, std::pair{5, 1},
                     std::pair{2, 3}})
  {
    for(int r : {1, 4})
    {
      cv::GaussianBlur obj;
      Image img(w, h, 0);
      for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
          img.set(x, y, static_cast<std::uint8_t>(40 + 20 * x), 0, 0, 128);
      feed(obj.inputs.image, img);
      obj.inputs.radius.value = r;
      obj();

      INFO(w << "x" << h << " radius " << r);
      auto& o = obj.outputs.image.texture;
      REQUIRE(o.width == w);
      REQUIRE(o.height == h);
      REQUIRE(o.bytes != nullptr);
      for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
          REQUIRE(alpha(obj, x, y) == 128);
    }
  }

  // A 1x1 image is its own blur, whatever the radius.
  cv::GaussianBlur obj;
  Image one(1, 1, 0);
  one.set(0, 0, 12, 34, 56, 78);
  feed(obj.inputs.image, one);
  obj.inputs.radius.value = 7;
  obj();
  auto& o = obj.outputs.image.texture;
  CHECK(o.bytes[0] == 12);
  CHECK(o.bytes[1] == 34);
  CHECK(o.bytes[2] == 56);
  CHECK(o.bytes[3] == 78);
}
