// Tests for the Hough post-processing chain: cv::Extrema (accumulator peak picker,
// cv.jit.extrema) and cv::HoughLines (peaks -> segments, the cv.jit.hough2lines
// abstraction), plus the two wired together end-to-end.
//
// All expectations are hand-computed from the documented formulas, not from the code.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/Extrema.hpp>
#include <CV/Cpu/HoughLines.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

using Catch::Approx;
using namespace cvtest;

namespace
{
inline constexpr double pi = std::numbers::pi_v<double>;

// Build a synthetic accumulator image: gray value == accumulator count (0..255).
Image accumulator(int w, int h, std::uint8_t fill = 0)
{
  return Image{w, h, fill}; // fill applies to r,g,b; alpha is forced to 255
}

// Reset an Extrema object's inputs to the cv.jit defaults.
void defaults(cv::Extrema& obj)
{
  obj.inputs.threshold.value = 20;
  obj.inputs.maxpoints.value = 64;
  obj.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours8;
}
}

// ------------------------------------------------------------------------------ Extrema

TEST_CASE("Extrema finds a single planted peak at the exact bin", "[extrema]")
{
  cv::Extrema obj;
  Image img = accumulator(16, 12);
  img.setGray(5, 7, 200);

  feed(obj.inputs.image, img);
  defaults(obj);
  obj();

  REQUIRE(obj.outputs.peaks.value.size() == 1);
  CHECK(obj.outputs.peaks.value[0].x == 5); // column = rho bin
  CHECK(obj.outputs.peaks.value[0].y == 7); // row    = theta bin
  CHECK(obj.outputs.peaks.value[0].value == Approx(200.f).margin(1.f));
  CHECK(obj.outputs.count.value == 1);
  CHECK(obj.outputs.acc_width.value == 16);
  CHECK(obj.outputs.acc_height.value == 12);
}

TEST_CASE("Extrema threshold is strict: value == threshold is rejected", "[extrema]")
{
  Image img = accumulator(16, 16);
  img.setGray(8, 8, 20);

  {
    cv::Extrema obj;
    feed(obj.inputs.image, img);
    defaults(obj); // threshold == 20
    obj();
    CHECK(obj.outputs.peaks.value.empty()); // 20 > 20 is false
    CHECK(obj.outputs.count.value == 0);
  }

  Image img2 = accumulator(16, 16);
  img2.setGray(8, 8, 21);
  {
    cv::Extrema obj;
    feed(obj.inputs.image, img2);
    defaults(obj);
    obj();
    REQUIRE(obj.outputs.peaks.value.size() == 1); // 21 > 20
    CHECK(obj.outputs.peaks.value[0].x == 8);
    CHECK(obj.outputs.peaks.value[0].y == 8);
  }

  // A peak below the threshold is dropped even though it is a genuine local max.
  Image img3 = accumulator(16, 16);
  img3.setGray(8, 8, 5);
  {
    cv::Extrema obj;
    feed(obj.inputs.image, img3);
    defaults(obj);
    obj();
    CHECK(obj.outputs.peaks.value.empty());
    // ... but it is found once the threshold drops below it.
    feed(obj.inputs.image, img3);
    obj.inputs.threshold.value = 4;
    obj();
    REQUIRE(obj.outputs.peaks.value.size() == 1);
    CHECK(obj.outputs.peaks.value[0].value == Approx(5.f).margin(1.f));
  }
}

TEST_CASE("Extrema maxpoints caps the count and keeps raster order", "[extrema]")
{
  cv::Extrema obj;
  Image img = accumulator(32, 32);
  // Five isolated peaks. Raster order (row-major) is: (4,2) (20,2) (8,10) (24,18) (12,26).
  img.setGray(20, 2, 100);
  img.setGray(4, 2, 110);
  img.setGray(8, 10, 250); // strongest, but NOT first in raster order
  img.setGray(24, 18, 90);
  img.setGray(12, 26, 200);

  feed(obj.inputs.image, img);
  defaults(obj);
  obj();
  REQUIRE(obj.outputs.peaks.value.size() == 5);

  feed(obj.inputs.image, img);
  obj.inputs.maxpoints.value = 3;
  obj();
  REQUIRE(obj.outputs.peaks.value.size() == 3);
  CHECK(obj.outputs.count.value == 3);
  // First-found in raster order, NOT strongest-first (cv.jit aborts the scan).
  CHECK(obj.outputs.peaks.value[0].x == 4);
  CHECK(obj.outputs.peaks.value[0].y == 2);
  CHECK(obj.outputs.peaks.value[1].x == 20);
  CHECK(obj.outputs.peaks.value[1].y == 2);
  CHECK(obj.outputs.peaks.value[2].x == 8);
  CHECK(obj.outputs.peaks.value[2].y == 10);
}

TEST_CASE("Extrema mode 0 (8-neighbour) rejects what mode 1 (4-neighbour) accepts",
          "[extrema]")
{
  // (5,5) = 200 with a LARGER diagonal (SE) neighbour at (6,6) = 250.
  // 4-neighbourhood of (5,5) is E(6,5)=W(4,5)=N(5,4)=S(5,6)=0  -> local max.
  // 8-neighbourhood additionally sees SE(6,6)=250 > 200         -> not a local max.
  // (6,6) is a local max under BOTH tests (its NW neighbour 200 < 250).
  Image img = accumulator(16, 16);
  img.setGray(5, 5, 200);
  img.setGray(6, 6, 250);

  {
    cv::Extrema obj;
    feed(obj.inputs.image, img);
    defaults(obj);
    obj.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours4;
    obj();
    REQUIRE(obj.outputs.peaks.value.size() == 2);
    CHECK(obj.outputs.peaks.value[0].x == 5);
    CHECK(obj.outputs.peaks.value[0].y == 5);
    CHECK(obj.outputs.peaks.value[1].x == 6);
    CHECK(obj.outputs.peaks.value[1].y == 6);
  }
  {
    cv::Extrema obj;
    feed(obj.inputs.image, img);
    defaults(obj); // Neighbours8 == cv.jit mode 0 == the default
    obj();
    REQUIRE(obj.outputs.peaks.value.size() == 1);
    CHECK(obj.outputs.peaks.value[0].x == 6);
    CHECK(obj.outputs.peaks.value[0].y == 6);
  }
}

TEST_CASE("Extrema never reports border pixels", "[extrema]")
{
  cv::Extrema obj;
  Image img = accumulator(16, 16);
  img.setGray(0, 0, 255);
  img.setGray(15, 15, 255);
  img.setGray(0, 8, 255);
  img.setGray(8, 0, 255);
  img.setGray(15, 7, 255);
  img.setGray(7, 15, 255);

  feed(obj.inputs.image, img);
  defaults(obj);
  obj();
  CHECK(obj.outputs.peaks.value.empty());

  // The very same value one cell inside IS reported.
  Image img2 = accumulator(16, 16);
  img2.setGray(1, 1, 255);
  img2.setGray(14, 14, 255);
  feed(obj.inputs.image, img2);
  obj();
  REQUIRE(obj.outputs.peaks.value.size() == 2);
  CHECK(obj.outputs.peaks.value[0].x == 1);
  CHECK(obj.outputs.peaks.value[0].y == 1);
  CHECK(obj.outputs.peaks.value[1].x == 14);
  CHECK(obj.outputs.peaks.value[1].y == 14);
}

TEST_CASE("Extrema yields nothing on a flat image or a plateau", "[extrema]")
{
  {
    cv::Extrema obj;
    Image img = accumulator(16, 16, 0);
    feed(obj.inputs.image, img);
    defaults(obj);
    obj();
    CHECK(obj.outputs.peaks.value.empty());
  }
  {
    // Flat but well above the threshold: strict '>' still rejects everything.
    cv::Extrema obj;
    Image img = accumulator(16, 16, 200);
    feed(obj.inputs.image, img);
    defaults(obj);
    obj();
    CHECK(obj.outputs.peaks.value.empty());
    CHECK(obj.outputs.count.value == 0);
  }
  {
    // Two-cell horizontal plateau: neither cell is strictly greater than the other.
    cv::Extrema obj;
    Image img = accumulator(16, 16);
    img.setGray(7, 7, 200);
    img.setGray(8, 7, 200);
    feed(obj.inputs.image, img);
    defaults(obj);
    obj();
    CHECK(obj.outputs.peaks.value.empty());
  }
  {
    // Same, diagonally: rejected by mode 0 only -- mode 1 does not look at diagonals,
    // so both cells come out. This pins the plateau rule per-neighbourhood.
    cv::Extrema obj;
    Image img = accumulator(16, 16);
    img.setGray(7, 7, 200);
    img.setGray(8, 8, 200);
    feed(obj.inputs.image, img);
    defaults(obj);
    obj();
    CHECK(obj.outputs.peaks.value.empty());

    feed(obj.inputs.image, img);
    obj.inputs.mode.value = cv::ExtremaNeighbourhood::Neighbours4;
    obj();
    CHECK(obj.outputs.peaks.value.size() == 2);
  }
}

TEST_CASE("Extrema handles degenerate accumulator sizes", "[extrema]")
{
  cv::Extrema obj;
  Image img = accumulator(2, 2, 255);
  feed(obj.inputs.image, img);
  defaults(obj);
  obj();
  CHECK(obj.outputs.peaks.value.empty());
  CHECK(obj.outputs.acc_width.value == 2);

  // Non-square accumulator: the cv.jit stride bug would sample wrong cells here.
  // 64 wide x 8 high, single peak; must be found at exactly (33, 4).
  cv::Extrema obj2;
  Image wide = accumulator(64, 8);
  wide.setGray(33, 4, 180);
  feed(obj2.inputs.image, wide);
  defaults(obj2);
  obj2();
  REQUIRE(obj2.outputs.peaks.value.size() == 1);
  CHECK(obj2.outputs.peaks.value[0].x == 33);
  CHECK(obj2.outputs.peaks.value[0].y == 4);
}

TEST_CASE("Extrema forwards the accumulator geometry to HoughLines", "[extrema]")
{
  // cv::Extrema sits between the producer of the accumulator and cv::HoughLines, which
  // needs the producer's exact bin steps and the SOURCE frame size to turn a bin index back
  // into a line. Those four values are pass-through: unchanged by the peak search, and
  // published even on a tick that brings no new frame (a patch that only moves `Rho` must
  // not stall them downstream while the peaks stay valid).
  cv::Extrema obj;
  Image img = accumulator(16, 12);
  img.setGray(5, 7, 200);

  feed(obj.inputs.image, img);
  defaults(obj);
  obj.inputs.theta_step.value = 0.0872664626;
  obj.inputs.rho_step.value = 4.0;
  obj.inputs.src_width.value = 640;
  obj.inputs.src_height.value = 480;
  obj();

  REQUIRE(obj.outputs.peaks.value.size() == 1);
  CHECK(obj.outputs.theta_step.value == Approx(0.0872664626).epsilon(1e-12));
  CHECK(obj.outputs.rho_step.value == Approx(4.0));
  CHECK(obj.outputs.src_width.value == 640);
  CHECK(obj.outputs.src_height.value == 480);
  // The accumulator size still comes from the texture itself, not from an input.
  CHECK(obj.outputs.acc_width.value == 16);
  CHECK(obj.outputs.acc_height.value == 12);

  // No new frame (`changed` is false after the previous run): the peaks are untouched but
  // the geometry keeps flowing.
  obj.inputs.rho_step.value = 2.5;
  obj();
  CHECK(obj.outputs.rho_step.value == Approx(2.5));
  CHECK(obj.outputs.peaks.value.size() == 1);

  // Unset (0) stays 0, which is HoughLines' "derive it yourself" signal.
  cv::Extrema fresh;
  Image img2 = accumulator(16, 12);
  img2.setGray(5, 7, 200);
  feed(fresh.inputs.image, img2);
  defaults(fresh);
  fresh();
  CHECK(fresh.outputs.theta_step.value == 0.);
  CHECK(fresh.outputs.rho_step.value == 0.);
  CHECK(fresh.outputs.src_width.value == 0);
  CHECK(fresh.outputs.src_height.value == 0);
}

TEST_CASE("Extrema's threshold is 8-bit, because the accumulator always is", "[extrema]")
{
  // The accumulator reaches this object as an RGBA8 texture whatever produced it, so a
  // decoded count can never exceed 255 and cv.jit's [0, 4096] range was dead travel.
  static_assert(
      decltype(std::declval<cv::Extrema&>().inputs.threshold)::range{}.max == 255,
      "an 8-bit input cannot be thresholded above 255");

  // A value pushed in programmatically above the range is still clamped at calc time.
  cv::Extrema obj;
  Image img = accumulator(16, 16);
  img.setGray(8, 8, 255); // the strongest a texture bin can possibly be
  feed(obj.inputs.image, img);
  defaults(obj);
  obj.inputs.threshold.value = 4096;
  obj();
  CHECK(obj.outputs.peaks.value.empty()); // 255 > 255 is false

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 254;
  obj();
  REQUIRE(obj.outputs.peaks.value.size() == 1);
  CHECK(obj.outputs.peaks.value[0].value == Approx(255.f));
}

// --------------------------------------------------------------------------- HoughLines

namespace
{
void houghDefaults(cv::HoughLines& obj, int w, int h, int accW, int accH)
{
  obj.inputs.width.value = w;
  obj.inputs.height.value = h;
  obj.inputs.acc_width.value = accW;
  obj.inputs.acc_height.value = accH;
  // 0 == unset: fall back to deriving the bin steps from the accumulator size.
  obj.inputs.theta_step.value = 0.;
  obj.inputs.rho_step.value = 0.;
  obj.inputs.rho_origin.value = cv::HoughRhoOrigin::ImageCorner;
  obj.inputs.threshold.value = 0;
  obj.inputs.maxlines.value = 64;
}
}

TEST_CASE("HoughLines: explicit Theta step / Rho step beat the derived ones", "[hough2lines]")
{
  // The whole of BUG 2 in one object, with no producer involved.
  //
  // A cv::HoughTransform accumulator built with the cv.jit defaults on a 640 x 480 frame is
  // 560 x 35 -- but you CANNOT recover its bin steps from those two numbers, because both
  // were produced by a truncation:
  //     numangle = (int)(pi / 0.0872664626)      = (int)35.999999996 = 35
  //     numrho   = (int)((2*(640+480) + 1) / 4)  = (int)560.25       = 560
  //
  // Peak (304, 18) is the horizontal line y = 100 (see tests/test_lines.cpp for the full
  // derivation). With the real steps wired in:
  //     theta = 18 * 0.0872664626 = 1.5707963268   -> sin = 1, cos = -5.1034e-12
  //     rho   = (304 - (560-1)/2) * 4 = (304 - 279) * 4 = 100      [INTEGER offset]
  //     y1 = 100, y2 = (100 - 640*cos)/sin = 100.000000003
  cv::HoughLines obj;
  houghDefaults(obj, 640, 480, 560, 35);
  obj.inputs.peaks.value = {cv::hough_peak{.x = 304, .y = 18, .value = 200.f}};

  SECTION("with the producer's steps: exactly the line that was drawn")
  {
    obj.inputs.theta_step.value = 5.0 * 0.01745329252;
    obj.inputs.rho_step.value = 4.0;
    obj();
    REQUIRE(obj.outputs.lines.value.size() == 1);
    const auto& l = obj.outputs.lines.value[0];
    CHECK(l.y1 == Approx(100.f).margin(1e-4));
    CHECK(l.y2 == Approx(100.f).margin(1e-4));
  }

  SECTION("without them: pi/accH and (2*(w+h)+1)/accW, and the line is tilted")
  {
    // dTheta = pi/35 = 0.08975979010256552 (2.857% too large), theta = 1.6156762218461793
    //   sin = 0.9989930665413147, cos = -0.04486483035051486
    // rho_scale = 2241/560 = 4.001785714285714, rho = 25 * that = 100.04464285714286
    //   y1 = 100.14548429, y2 = 128.88790873
    obj(); // theta_step / rho_step left at 0
    REQUIRE(obj.outputs.lines.value.size() == 1);
    const auto& l = obj.outputs.lines.value[0];
    CHECK(l.y1 == Approx(100.14548429f).epsilon(1e-6));
    CHECK(l.y2 == Approx(128.88790873f).epsilon(1e-6));
    CHECK(std::abs(l.y2 - l.y1) > 28.f);
  }
}

TEST_CASE("HoughLines: the rho offset is an INTEGER (accW-1)/2, like the accumulator's",
          "[hough2lines]")
{
  // cv::HoughTransform / cv.jit.hough place rho = 0 at bin (numrho - 1) / 2 computed in
  // INTEGER arithmetic. For any EVEN accumulator width that is half a bin below the
  // fractional centre, and using (accW - 1) / 2.0 instead biases every reported rho.
  //
  // w = h = 100, accW = 256 (even), accH = 180, peak (140, 90):
  //   rho_scale = (2*(100+100) + 1) / 256 = 401/256 = 1.56640625
  //   integer offset (256-1)/2 = 127   -> rho = 13   * 1.56640625 = 20.36328125
  //   float   offset      127.5        -> rho = 12.5 * 1.56640625 = 19.580078125
  // theta bin 90 = pi/2 -> sin = 1, so y1 == y2 == rho and the two are 0.783 px apart.
  cv::HoughLines obj;
  houghDefaults(obj, 100, 100, 256, 180);
  obj.inputs.peaks.value = {cv::hough_peak{.x = 140, .y = 90, .value = 200.f}};
  obj();

  REQUIRE(obj.outputs.lines.value.size() == 1);
  const auto& l = obj.outputs.lines.value[0];
  CHECK(l.y1 == Approx(20.36328125f).epsilon(1e-6));
  CHECK(l.y2 == Approx(20.36328125f).epsilon(1e-6));
  CHECK(l.y1 != Approx(19.580078125f).epsilon(1e-6)); // the float-offset answer

  // On an ODD accW the two conventions coincide, which is exactly why the old chain test
  // (accW = 257) never noticed: 256/2 == 256/2.0 == 128.
  cv::HoughLines odd;
  houghDefaults(odd, 100, 100, 257, 180);
  odd.inputs.peaks.value = {cv::hough_peak{.x = 140, .y = 90, .value = 200.f}};
  odd();
  REQUIRE(odd.outputs.lines.value.size() == 1);
  // rho_scale = 401/257 = 1.560311284046693, rho = (140 - 128) * that = 18.72373540856
  CHECK(odd.outputs.lines.value[0].y1 == Approx(18.72373540856f).epsilon(1e-6));
}

TEST_CASE("HoughLines: `Rho origin` = Image centre decodes a Hough.cs accumulator",
          "[hough2lines][shader]")
{
  // BUG 3. HoughLines' defaults (Acc width 256, Acc height 180) are the SIZE of the
  // accumulator CV/Shaders/Analysis/Hough.cs produces, but size is not the convention: the
  // shader measures rho from the image CENTRE in NORMALISED units over +/- sqrt(1/2)
  // (Hough.cs:60-77), while this object measures it from the image ORIGIN in pixels. The
  // two differ by a scale factor of ~2*sqrt(2) AND an origin shift, which the old code
  // silently papered over.
  //
  // The lambda below is Hough.cs's voting arithmetic transcribed line for line, so the
  // expectations are derived from the shader, not from HoughLines.
  constexpr int S = 256, RHO_BINS = 256, THETA_BINS = 180;
  auto shaderRhoBin = [&](int px, int py, int t) {
    const double th = t * pi / THETA_BINS;
    const double ux = (px + 0.5) / S, uy = (py + 0.5) / S;
    const double cx = ux - 0.5, cy = uy - 0.5;
    const double rho = cx * std::cos(th) + cy * std::sin(th);
    const double RHO_MAX = 0.70710678;
    int rbin = static_cast<int>((rho / RHO_MAX * 0.5 + 0.5) * (RHO_BINS - 1) + 0.5);
    return std::clamp(rbin, 0, RHO_BINS - 1);
  };

  // Auto rho scale in centre mode is sqrt(2)*Width/(accW-1) = 1.419759497911813 px/bin, so
  // the quantisation error can be at most half a bin = 0.71 px.
  SECTION("a horizontal line at y = 192 comes back at y = 192")
  {
    const int rbin = shaderRhoBin(0, 192, 90); // theta bin 90 = pi/2
    REQUIRE(rbin == 173);

    cv::HoughLines obj;
    houghDefaults(obj, S, S, RHO_BINS, THETA_BINS);
    obj.inputs.rho_origin.value = cv::HoughRhoOrigin::ImageCentre;
    obj.inputs.peaks.value = {cv::hough_peak{.x = rbin, .y = 90, .value = 200.f}};
    obj();

    REQUIRE(obj.outputs.lines.value.size() == 1);
    const auto& l = obj.outputs.lines.value[0];
    // rho_centre = (173 - 127.5) * 1.419759497911813 = 64.5990571550
    // rho_corner = rho_centre + ((S-1)/2)*cos + ((S-1)/2)*sin = + 127.5 = 192.0990571550
    CHECK(l.y1 == Approx(192.09905715f).epsilon(1e-6));
    CHECK(l.y2 == Approx(192.09905715f).epsilon(1e-6));
    CHECK(std::abs(l.y1 - 192.f) < 0.71f); // within half a rho bin of the truth

    // The corner convention on the very same bin is off by ~8 px:
    //   (173 - (256-1)/2) * ((2*(256+256)+1)/256) = 46 * 4.00390625 = 184.1796875
    obj.inputs.rho_origin.value = cv::HoughRhoOrigin::ImageCorner;
    obj();
    REQUIRE(obj.outputs.lines.value.size() == 1);
    CHECK(obj.outputs.lines.value[0].y1 == Approx(184.1796875f).epsilon(1e-6));
    CHECK(std::abs(obj.outputs.lines.value[0].y1 - 192.f) > 7.f);
  }

  SECTION("a vertical line at x = 200 comes back at x = 200")
  {
    const int rbin = shaderRhoBin(200, 0, 0); // theta bin 0 -> cos = 1, sin = 0
    REQUIRE(rbin == 179);

    cv::HoughLines obj;
    houghDefaults(obj, S, S, RHO_BINS, THETA_BINS);
    obj.inputs.rho_origin.value = cv::HoughRhoOrigin::ImageCentre;
    obj.inputs.peaks.value = {cv::hough_peak{.x = rbin, .y = 0, .value = 200.f}};
    obj();

    REQUIRE(obj.outputs.lines.value.size() == 1);
    const auto& l = obj.outputs.lines.value[0];
    // rho_corner = (179 - 127.5) * 1.419759497911813 + 127.5 = 200.6176141425
    CHECK(l.x1 == Approx(200.61761414f).epsilon(1e-6));
    CHECK(l.x2 == Approx(200.61761414f).epsilon(1e-6));
    CHECK(l.y1 == Approx(0.f));
    CHECK(l.y2 == Approx(static_cast<float>(S)));
    CHECK(std::abs(l.x1 - 200.f) < 0.71f);

    // Corner convention on the same bin: (179 - 127) * 4.00390625 = 208.203125, 8 px out.
    obj.inputs.rho_origin.value = cv::HoughRhoOrigin::ImageCorner;
    obj();
    CHECK(obj.outputs.lines.value[0].x1 == Approx(208.203125f).epsilon(1e-6));
    CHECK(std::abs(obj.outputs.lines.value[0].x1 - 200.f) > 7.f);
  }
}

TEST_CASE("cv_round is OpenCV's cvRound (ties to EVEN), not std::lround", "[hough2lines]")
{
  // OpenCV's cvRound compiles to cvtsd2si / lrint and therefore rounds half to EVEN under
  // the default FE_TONEAREST; std::lround rounds half AWAY FROM ZERO. Ported OpenCV code
  // that says cvRound must use cv::cv_round, or it will disagree on every exact .5 -- which
  // in a Hough accumulator is precisely a bin boundary.
  CHECK(cv::cv_round(0.5) == 0);
  CHECK(cv::cv_round(1.5) == 2);
  CHECK(cv::cv_round(2.5) == 2);
  CHECK(cv::cv_round(-0.5) == 0);
  CHECK(cv::cv_round(-1.5) == -2);
  CHECK(cv::cv_round(-2.5) == -2);
  // ... where std::lround gives 1, 2, 3, -1, -2, -3.
  CHECK(std::lround(0.5) == 1);
  CHECK(std::lround(2.5) == 3);
  CHECK(std::lround(-0.5) == -1);
  // Away from ties they agree.
  CHECK(cv::cv_round(0.4) == 0);
  CHECK(cv::cv_round(0.6) == 1);
  CHECK(cv::cv_round(-3.7) == -4);
}

TEST_CASE("HoughLines matches the hand-computed formula", "[hough2lines]")
{
  // w = h = 100, accW = 201, accH = 180.
  //   dTheta    = pi/180 (one degree per theta bin)
  //   rho_scale = (2*(100+100)+1)/201 = 401/201 = 1.99502487562189
  //   rho(120)  = (120 - (201-1)/2) * rho_scale = 20 * 1.99502487562189
  //             = 39.9004975124378
  cv::HoughLines obj;
  houghDefaults(obj, 100, 100, 201, 180);

  SECTION("theta bin 90 -> horizontal line at y = rho")
  {
    // theta = 90 deg: sin = 1, cos = 0 -> y1 = y2 = rho.
    obj.inputs.peaks.value = {cv::hough_peak{.x = 120, .y = 90, .value = 200.f}};
    obj();
    REQUIRE(obj.outputs.lines.value.size() == 1);
    const auto& l = obj.outputs.lines.value[0];
    CHECK(l.x1 == Approx(0.f));
    CHECK(l.y1 == Approx(39.9004975f).epsilon(1e-5));
    CHECK(l.x2 == Approx(100.f));
    CHECK(l.y2 == Approx(39.9004975f).epsilon(1e-5));
    CHECK(obj.outputs.count.value == 1);
  }

  SECTION("theta bin 45 -> diagonal line")
  {
    // theta = 45 deg: sin = cos = 0.7071067811865476.
    //   y1 = rho / sin           = 56.42782472752349
    //   y2 = (rho - 100*cos)/sin = -43.572175272476514
    obj.inputs.peaks.value = {cv::hough_peak{.x = 120, .y = 45, .value = 200.f}};
    obj();
    REQUIRE(obj.outputs.lines.value.size() == 1);
    const auto& l = obj.outputs.lines.value[0];
    CHECK(l.x1 == Approx(0.f));
    CHECK(l.y1 == Approx(56.4278247f).epsilon(1e-5));
    CHECK(l.x2 == Approx(100.f));
    CHECK(l.y2 == Approx(-43.5721753f).epsilon(1e-5));
  }
}

TEST_CASE("HoughLines emits a finite vertical segment when sin(theta) == 0",
          "[hough2lines]")
{
  cv::HoughLines obj;
  houghDefaults(obj, 100, 100, 201, 180);
  obj.inputs.peaks.value = {cv::hough_peak{.x = 120, .y = 0, .value = 200.f}};
  obj();

  REQUIRE(obj.outputs.lines.value.size() == 1);
  const auto& l = obj.outputs.lines.value[0];
  CHECK(std::isfinite(l.x1));
  CHECK(std::isfinite(l.y1));
  CHECK(std::isfinite(l.x2));
  CHECK(std::isfinite(l.y2));
  // theta = 0 -> cos = 1 -> the line is x = rho, spanning the image height.
  CHECK(l.x1 == Approx(39.9004975f).epsilon(1e-5));
  CHECK(l.x2 == Approx(39.9004975f).epsilon(1e-5));
  CHECK(l.y1 == Approx(0.f));
  CHECK(l.y2 == Approx(100.f));
}

TEST_CASE("HoughLines: empty peak list yields no lines", "[hough2lines]")
{
  cv::HoughLines obj;
  houghDefaults(obj, 100, 100, 201, 180);
  obj.inputs.peaks.value.clear();
  obj();
  CHECK(obj.outputs.lines.value.empty());
  CHECK(obj.outputs.count.value == 0);

  // A non-empty run followed by an empty one must clear the previous result.
  obj.inputs.peaks.value = {cv::hough_peak{.x = 120, .y = 90, .value = 200.f}};
  obj();
  REQUIRE(obj.outputs.lines.value.size() == 1);
  obj.inputs.peaks.value.clear();
  obj();
  CHECK(obj.outputs.lines.value.empty());
  CHECK(obj.outputs.count.value == 0);
}

TEST_CASE("HoughLines maxlines caps the output and threshold filters peaks",
          "[hough2lines]")
{
  cv::HoughLines obj;
  houghDefaults(obj, 100, 100, 201, 180);
  obj.inputs.peaks.value = {
      cv::hough_peak{.x = 120, .y = 90, .value = 200.f},
      cv::hough_peak{.x = 121, .y = 91, .value = 150.f},
      cv::hough_peak{.x = 122, .y = 92, .value = 30.f},
      cv::hough_peak{.x = 123, .y = 93, .value = 25.f},
  };

  obj();
  CHECK(obj.outputs.lines.value.size() == 4);

  obj.inputs.maxlines.value = 2;
  obj();
  REQUIRE(obj.outputs.lines.value.size() == 2);
  CHECK(obj.outputs.count.value == 2);
  // Kept in list order: the first entry is still the (120, 90) line.
  CHECK(obj.outputs.lines.value[0].y1 == Approx(39.9004975f).epsilon(1e-5));

  obj.inputs.maxlines.value = 64;
  obj.inputs.threshold.value = 100; // strict '>' -> only the 200 and 150 peaks survive
  obj();
  CHECK(obj.outputs.lines.value.size() == 2);

  obj.inputs.threshold.value = 200; // strict '>': 200 > 200 is false -> nothing survives
  obj();
  CHECK(obj.outputs.lines.value.empty());
  CHECK(obj.outputs.count.value == 0);
}

// ---------------------------------------------------------------------------- The chain

TEST_CASE("Extrema -> HoughLines end to end", "[hough2lines][extrema][chain]")
{
  // 32-wide (rho) x 16-high (theta) accumulator with one peak at bin (20, 8).
  cv::Extrema ex;
  Image acc = accumulator(32, 16);
  acc.setGray(20, 8, 200);
  feed(ex.inputs.image, acc);
  defaults(ex);
  ex();

  REQUIRE(ex.outputs.peaks.value.size() == 1);
  CHECK(ex.outputs.peaks.value[0].x == 20);
  CHECK(ex.outputs.peaks.value[0].y == 8);
  CHECK(ex.outputs.acc_width.value == 32);
  CHECK(ex.outputs.acc_height.value == 16);

  // Pipe the peak list straight into HoughLines, taking the accumulator dims from
  // Extrema's own outputs -- the whole point of the list port.
  cv::HoughLines hl;
  houghDefaults(
      hl, 100, 100, ex.outputs.acc_width.value, ex.outputs.acc_height.value);
  hl.inputs.peaks.value = ex.outputs.peaks.value;
  hl();

  // Hand-computed: dTheta = pi/16, theta bin 8 -> theta = pi/2 (sin = 1, cos = 0).
  //   rho_scale = (2*(100+100)+1)/32 = 401/32 = 12.53125
  //   rho       = (20 - (32-1)/2) * 12.53125
  // (32 - 1) / 2 is an INTEGER division, exactly as in the accumulator builders, so the
  // offset is 15, NOT the 15.5 a float division would give -- this accumulator is 32 bins
  // wide, i.e. EVEN, which is where the two conventions part company:
  //   rho     = 5 * 12.53125 = 62.65625
  //   y1 = y2 = 62.65625
  REQUIRE(hl.outputs.lines.value.size() == 1);
  const auto& l = hl.outputs.lines.value[0];
  CHECK(l.x1 == Approx(0.f));
  CHECK(l.y1 == Approx(62.65625f).epsilon(1e-5));
  CHECK(l.x2 == Approx(100.f));
  CHECK(l.y2 == Approx(62.65625f).epsilon(1e-5));
  CHECK(hl.outputs.count.value == 1);
}

TEST_CASE("Extrema -> HoughLines: two peaks give two distinct lines",
          "[hough2lines][extrema][chain]")
{
  cv::Extrema ex;
  Image acc = accumulator(32, 16);
  acc.setGray(20, 8, 200); // theta = pi/2  -> horizontal
  acc.setGray(10, 4, 180); // theta = pi/4  -> diagonal
  feed(ex.inputs.image, acc);
  defaults(ex);
  ex();
  REQUIRE(ex.outputs.peaks.value.size() == 2);
  // Raster order: row 4 before row 8.
  CHECK(ex.outputs.peaks.value[0].y == 4);
  CHECK(ex.outputs.peaks.value[1].y == 8);

  cv::HoughLines hl;
  houghDefaults(hl, 100, 100, 32, 16);
  hl.inputs.peaks.value = ex.outputs.peaks.value;
  hl();
  REQUIRE(hl.outputs.lines.value.size() == 2);

  // Peak (10, 4): theta = 4*pi/16 = pi/4, rho = (10 - 15)*12.53125 = -62.65625
  // ((32-1)/2 is integer division: 15, not 15.5 -- see the previous test case.)
  //   y1 = rho/sin = -62.65625 * sqrt(2)           = -88.60931851743925
  //   y2 = (rho - 100*cos)/sin = y1 - 100          = -188.60931851743925
  const auto& d = hl.outputs.lines.value[0];
  CHECK(d.y1 == Approx(-88.60931852f).epsilon(1e-5));
  CHECK(d.y2 == Approx(-188.60931852f).epsilon(1e-5));

  const auto& hz = hl.outputs.lines.value[1];
  CHECK(hz.y1 == Approx(62.65625f).epsilon(1e-5));
  CHECK(hz.y2 == Approx(62.65625f).epsilon(1e-5));
}
