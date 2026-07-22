// Tests for blob-analysis objects: Label, Contours, BlobStats, BlobSort.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/Label.hpp>
#include <CV/Cpu/Contours.hpp>
#include <CV/Cpu/BlobStats.hpp>
#include <CV/Cpu/BlobSort.hpp>

#include <cmath>

using Catch::Approx;
using namespace cvtest;

// -------------------------------------------------------------------------------- Label
TEST_CASE("Label counts two separate blobs", "[label]")
{
  cv::Label obj;
  Image img(32, 32, 0);
  img.fillRect(2, 2, 6, 6, 255);    // blob A
  img.fillRect(20, 20, 6, 6, 255);  // blob B

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 0;
  obj();

  CHECK(obj.outputs.count.value == 2);
}

TEST_CASE("Label merges an 8-connected diagonal blob", "[label]")
{
  cv::Label obj;
  Image img(8, 8, 0);
  img.setGray(1, 1, 255);
  img.setGray(2, 2, 255); // diagonal touch -> same blob (8-connectivity)
  img.setGray(3, 3, 255);
  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj();
  CHECK(obj.outputs.count.value == 1);
}

TEST_CASE("Label min-size filter drops small blobs", "[label]")
{
  cv::Label obj;
  Image img(32, 32, 0);
  img.fillRect(2, 2, 6, 6, 255);  // 36 px
  img.setGray(20, 20, 255);       // 1 px

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 10;
  obj();
  CHECK(obj.outputs.count.value == 1); // only the big one survives
}

TEST_CASE("Label on empty image yields zero blobs", "[label]")
{
  cv::Label obj;
  Image img(16, 16, 0);
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.count.value == 0);
}

TEST_CASE("Label exposes a numeric label field with distinct ids per blob", "[label]")
{
  cv::Label obj;
  Image img(32, 32, 0);
  img.fillRect(2, 2, 6, 6, 255);    // blob A
  img.fillRect(20, 20, 6, 6, 255);  // blob B

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 2);

  auto& lbl = obj.outputs.labels.texture;
  REQUIRE(lbl.bytes != nullptr);
  REQUIRE(lbl.width == 32);
  REQUIRE(lbl.height == 32);
  const float* f = reinterpret_cast<const float*>(lbl.bytes);

  auto label_at = [&](int x, int y) { return f[static_cast<std::size_t>(y) * 32 + x]; };

  // Background is 0.
  CHECK(label_at(0, 0) == Approx(0.f));
  // Each blob interior carries a non-zero id.
  const float ia = label_at(4, 4);
  const float ib = label_at(22, 22);
  CHECK(ia > 0.f);
  CHECK(ib > 0.f);
  // The two blobs must carry different label ids.
  CHECK(ia != ib);
}

// ----------------------------------------------------------------------------- Contours
TEST_CASE("Contours finds one contour for a single rectangle", "[contours]")
{
  cv::Contours obj;
  Image img(32, 32, 0);
  img.fillRect(8, 8, 12, 12, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value >= 1);
  // The largest contour's bbox should roughly cover the rectangle.
  const auto& c = obj.outputs.contours.value.front();
  CHECK(c.area > 0.f);
  CHECK(c.perimeter > 0.f);
  CHECK(c.point_count >= 4);
  // centroid near the rectangle centre (~ (8+6)/32, (8+6)/32 ≈ 0.44)
  CHECK(c.centroid.x == Approx(0.44f).margin(0.1));
  CHECK(c.centroid.y == Approx(0.44f).margin(0.1));
}

TEST_CASE("Contours bbox is an inclusive pixel box", "[contours]")
{
  cv::Contours obj;
  Image img(32, 32, 0);
  // Rectangle spanning columns 8..19 and rows 8..19 -> 12 px each way inclusive.
  img.fillRect(8, 8, 12, 12, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value >= 1);
  const auto& c = obj.outputs.contours.value.front();
  // Inclusive box: width = (maxx-minx+1)/W = 12/32, matching BlobStats convention.
  CHECK(c.bbox.w == Approx(12.f / 32.f).margin(0.001));
  CHECK(c.bbox.h == Approx(12.f / 32.f).margin(0.001));
}

TEST_CASE("Contours respects the min-perimeter filter", "[contours]")
{
  cv::Contours obj;
  Image img(32, 32, 0);
  img.setGray(5, 5, 255); // a 1px speck, tiny perimeter

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 50; // far larger than a speck
  obj();
  CHECK(obj.outputs.count.value == 0);
}

// ----------------------------------------------------------------------------- BlobStats
//
// Reference geometry used repeatedly below, with the moments worked out by hand.
//
// HORIZONTAL BAR  -- Image(48,48), fillRect(8, 22, 30, 4): x in 8..37, y in 22..25.
//   m00 = 120, cx = 22.5, cy = 23.5
//   mu20 = 4 * sum_{x=8..37}(x-22.5)^2 = 4 * 2247.5 = 8990
//   mu02 = 30 * sum_{y=22..25}(y-23.5)^2 = 30 * 5     = 150
//   mu11 = 0 (separable and symmetric on both axes); all 3rd-order mu are 0.
//   nu20 = 8990/120^2 = 8990/14400, nu02 = 150/14400, nu11 = 0
//   cv.jit orientation: nu20 > nu02, nu11 == 0 -> c == 0 -> theta == 0 exactly.
//   cv.jit elongation:  ((nu20-nu02)^2 + 4*nu11^2)/(nu20*nu02)
//                     = 8840^2 / (8990*150) = 78145600/1348500 = 57.9500185...
//
// SQUARE -- Image(40,40), fillRect(10, 14, 8, 8): x in 10..17, y in 14..21.
//   m00 = 64, cx = 13.5, cy = 17.5, mu20 = mu02 = 336 exactly, mu11 = 0.
//   nu20 == nu02 == 336/4096 == 0.08203125 -> cv.jit orientation returns EXACTLY 0,
//   and cv.jit elongation numerator is 0 -> elongation == 0 (NOT 1: cv.jit's
//   "elongation" is 0 for an isotropic blob and grows as the blob thins out).
//
// ASYMMETRIC BLOB -- 6 px: (10,10),(11,10),(12,10),(13,10),(10,11),(11,11).
//   m00 = 6, cx = 67/6, cy = 31/3
//   mu20 = 41/6, mu02 = 4/3, mu11 = -4/3, mu30 = 32/9, mu03 = 4/9,
//   mu21 = -8/9, mu12 = -4/9
//   nu20 = 41/216           = 0.1898148148
//   nu02 = (4/3)/36         = 0.0370370370
//   nu11 = (-4/3)/36        = -0.0370370370
//   nu30 = (32/9)/6^2.5     = 0.0403208188
//   nu03 = (4/9)/6^2.5      = 0.0050401024
//   nu21 = (-8/9)/6^2.5     = -0.0100802047
//   nu12 = (-4/9)/6^2.5     = -0.0050401024
//   cv.jit orientation: d = nu20-nu02 > 0, c = 0.5*atan(2*nu11/d) = -0.2257265 < 0
//                       -> theta = c + pi = 2.9158661078
//   cv.jit direction:   theta > 3pi/4 and nu30 > 0 -> theta + pi = 6.0574587614
//   cv.jit elongation:  4.1006097561

namespace
{
constexpr double kPi = 3.14159265358979323846;

// The 6-px asymmetric reference blob described above.
void drawAsymBlob(Image& img)
{
  img.fillRect(10, 10, 4, 1, 255); // (10..13, 10)
  img.fillRect(10, 11, 2, 1, 255); // (10..11, 11)
}
}

TEST_CASE("BlobStats reports centroid, bbox, area for a square", "[blobstats]")
{
  cv::BlobStats obj;
  Image img(40, 40, 0);
  img.fillRect(10, 14, 8, 8, 255); // square at x10..17, y14..21

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  // centroid ≈ (13.5/40, 17.5/40)
  CHECK(b.centroid.x == Approx(13.5f / 40.f).margin(0.02));
  CHECK(b.centroid.y == Approx(17.5f / 40.f).margin(0.02));
  // area = 64 / 1600
  CHECK(b.area == Approx(64.f / 1600.f).margin(0.005));
  // bbox width/height = 8/40
  CHECK(b.bbox.w == Approx(8.f / 40.f).margin(0.03));
  // cv.jit (default) elongation of an isotropic blob is 0, not 1: the numerator
  // (nu20-nu02)^2 + 4*nu11^2 vanishes. This differs from the pre-fix port, which
  // reported the eigenvalue-ratio 1. See the Normalized-mode test below.
  CHECK(b.elongation == Approx(0.f).margin(1e-6));
}

TEST_CASE("BlobStats square reports the eigenvalue ratio 1 in Normalized mode", "[blobstats]")
{
  cv::BlobStats obj;
  Image img(40, 40, 0);
  img.fillRect(10, 14, 8, 8, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj.inputs.formula.value = cv::BlobFormula::Normalized;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.blobs.value.front().elongation == Approx(1.f).margin(1e-4));
}

TEST_CASE("BlobStats cv.jit orientation of a horizontal bar is exactly 0", "[blobstats]")
{
  cv::BlobStats obj;
  Image img(48, 48, 0);
  img.fillRect(8, 22, 30, 4, 255); // wide, short bar -> elongated, horizontal

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  // Hand-computed above: nu11 == 0 and nu20 > nu02 -> theta == 0.
  CHECK(b.orientation == 0.f);
  // cv.jit orientation is always in [0, pi).
  CHECK(b.orientation >= 0.f);
  CHECK(b.orientation < static_cast<float>(kPi));
  // Hand-computed: 8840^2 / (8990*150).
  const float expected = static_cast<float>(78145600.0 / 1348500.0); // 57.9500185...
  CHECK(b.elongation == Approx(expected).epsilon(1e-5));
  CHECK(b.elongation > 2.f); // still "clearly elongated"
}

TEST_CASE("BlobStats cv.jit orientation of a vertical bar is pi/2 and lies in [0,pi)",
          "[blobstats]")
{
  cv::BlobStats obj;
  Image img(48, 48, 0);
  img.fillRect(22, 8, 4, 30, 255); // tall, narrow bar -> vertical principal axis

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  // nu02 > nu20 -> theta = c + pi/2 with c == 0. The old atan2 formula gave -pi/2 here.
  CHECK(b.orientation == Approx(static_cast<float>(kPi * 0.5)).margin(1e-6));
  CHECK(b.orientation >= 0.f);
  CHECK(b.orientation < static_cast<float>(kPi));
}

TEST_CASE("BlobStats cv.jit orientation is exactly 0 when nu20 == nu02", "[blobstats]")
{
  cv::BlobStats obj;
  Image img(40, 40, 0);
  img.fillRect(10, 14, 8, 8, 255); // square: mu20 == mu02 == 336 exactly

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  // Precondition of the branch under test: the two moments really are bit-equal.
  REQUIRE(b.nu20 == b.nu02);
  // cv.jit short-circuits d == 0 to theta = 0 rather than falling into atan.
  CHECK(b.orientation == 0.f);
  CHECK(b.direction == 0.f);
}

TEST_CASE("BlobStats exposes the cv.jit normalised central moments", "[blobstats]")
{
  SECTION("horizontal bar")
  {
    cv::BlobStats obj;
    Image img(48, 48, 0);
    img.fillRect(8, 22, 30, 4, 255);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 4;
    obj();

    REQUIRE(obj.outputs.count.value == 1);
    const auto& b = obj.outputs.blobs.value.front();
    CHECK(b.nu20 == Approx(8990.f / 14400.f).epsilon(1e-6)); // 0.6243055556
    CHECK(b.nu02 == Approx(150.f / 14400.f).epsilon(1e-6));  // 0.0104166667
    CHECK(b.nu11 == Approx(0.f).margin(1e-9));
    // A doubly-symmetric shape has no odd-order central moments.
    CHECK(b.nu30 == Approx(0.f).margin(1e-9));
    CHECK(b.nu03 == Approx(0.f).margin(1e-9));
    CHECK(b.nu21 == Approx(0.f).margin(1e-9));
    CHECK(b.nu12 == Approx(0.f).margin(1e-9));
    // Hu[0] is by definition nu20 + nu02.
    CHECK(b.hu[0] == Approx(b.nu20 + b.nu02).epsilon(1e-6));
  }

  SECTION("asymmetric 6-px blob")
  {
    cv::BlobStats obj;
    Image img(32, 32, 0);
    drawAsymBlob(img);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 0;
    obj();

    REQUIRE(obj.outputs.count.value == 1);
    const auto& b = obj.outputs.blobs.value.front();
    CHECK(b.mass == Approx(6.f));
    CHECK(b.nu20 == Approx(0.1898148148f).margin(1e-6));
    CHECK(b.nu02 == Approx(0.0370370370f).margin(1e-6));
    CHECK(b.nu11 == Approx(-0.0370370370f).margin(1e-6));
    CHECK(b.nu30 == Approx(0.0403208188f).margin(1e-6));
    CHECK(b.nu03 == Approx(0.0050401024f).margin(1e-6));
    CHECK(b.nu21 == Approx(-0.0100802047f).margin(1e-6));
    CHECK(b.nu12 == Approx(-0.0050401024f).margin(1e-6));
  }
}

TEST_CASE("BlobStats cv.jit orientation/direction/elongation on an asymmetric blob",
          "[blobstats]")
{
  cv::BlobStats obj;
  Image img(32, 32, 0);
  drawAsymBlob(img);
  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  // theta = 0.5*atan(2*nu11/(nu20-nu02)) + pi (c < 0 with nu20 > nu02).
  CHECK(b.orientation == Approx(2.9158661078f).margin(1e-5));
  CHECK(b.orientation >= 0.f);
  CHECK(b.orientation < static_cast<float>(kPi));
  // theta > 3pi/4 band and nu30 > 0 -> direction = theta + pi.
  CHECK(b.direction == Approx(6.0574587614f).margin(1e-5));
  // ((nu20-nu02)^2 + 4*nu11^2)/(nu20*nu02)
  CHECK(b.elongation == Approx(4.1006097561f).margin(1e-4));
}

TEST_CASE("BlobStats Flip shifts the direction by exactly pi", "[blobstats]")
{
  auto direction_for = [](bool flip) {
    cv::BlobStats obj;
    Image img(32, 32, 0);
    drawAsymBlob(img);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 0;
    obj.inputs.flip.value = flip;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    return obj.outputs.blobs.value.front().direction;
  };

  const float d0 = direction_for(false);
  const float d1 = direction_for(true);
  CHECK(std::abs(d1 - d0) == Approx(static_cast<float>(kPi)).margin(1e-5));
  // cv.jit subtracts pi when the direction is positive.
  CHECK(d1 == Approx(d0 - static_cast<float>(kPi)).margin(1e-5));

  // Same invariant in Normalized mode (there the shift wraps into [0, 2pi)).
  auto normalized_direction_for = [](bool flip) {
    cv::BlobStats obj;
    Image img(32, 32, 0);
    drawAsymBlob(img);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 0;
    obj.inputs.formula.value = cv::BlobFormula::Normalized;
    obj.inputs.flip.value = flip;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    return obj.outputs.blobs.value.front().direction;
  };
  const float n0 = normalized_direction_for(false);
  const float n1 = normalized_direction_for(true);
  CHECK(std::abs(n1 - n0) == Approx(static_cast<float>(kPi)).margin(1e-5));
}

TEST_CASE("BlobStats Formula switch actually changes the numbers", "[blobstats]")
{
  auto run = [](cv::BlobFormula f) {
    cv::BlobStats obj;
    Image img(32, 32, 0);
    drawAsymBlob(img);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 0;
    obj.inputs.formula.value = f;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    return obj.outputs.blobs.value.front();
  };

  const auto cvjit = run(cv::BlobFormula::CvJit);
  const auto norm = run(cv::BlobFormula::Normalized);

  // cv.jit orientation is in [0, pi); the Normalized (atan2) one is in [-pi/2, pi/2] and
  // here comes out negative -> the two differ by exactly pi.
  CHECK(cvjit.orientation == Approx(2.9158661078f).margin(1e-5));
  CHECK(norm.orientation == Approx(-0.2257265458f).margin(1e-5));
  CHECK(norm.orientation < 0.f);
  CHECK(cvjit.orientation != norm.orientation);

  // Elongations are numerically unrelated quantities.
  CHECK(cvjit.elongation == Approx(4.1006097561f).margin(1e-4));
  CHECK(norm.elongation >= 1.f); // eigenvalue ratio, always >= 1
  CHECK(cvjit.elongation != norm.elongation);

  // The normalised moments are mode-independent.
  CHECK(cvjit.nu20 == norm.nu20);
  CHECK(cvjit.nu11 == norm.nu11);
  CHECK(cvjit.hu[0] == norm.hu[0]);
}

TEST_CASE("BlobStats degenerate 1-pixel-wide blob is well defined in both modes",
          "[blobstats]")
{
  // A 1x5 vertical line: mu20 == 0 exactly (all x equal), mu02 == 10, mu11 == 0.
  // nu20 == 0 -> cv.jit's elongation denominator nu20*nu02 is 0 while the numerator
  // (nu20-nu02)^2 == nu02^2 > 0, so cv.jit yields +inf. Normalized mode guards the
  // degenerate eigenvalue and reports 1.
  auto run = [](cv::BlobFormula f) {
    cv::BlobStats obj;
    Image img(32, 32, 0);
    img.fillRect(20, 10, 1, 5, 255);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 0;
    obj.inputs.formula.value = f;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    return obj.outputs.blobs.value.front();
  };

  const auto cvjit = run(cv::BlobFormula::CvJit);
  CHECK(cvjit.mass == Approx(5.f));
  CHECK(cvjit.nu20 == 0.f);
  CHECK(cvjit.nu02 == Approx(10.f / 25.f).margin(1e-6));
  CHECK(std::isinf(cvjit.elongation));
  CHECK(cvjit.elongation > 0.f); // +inf, not -inf and not NaN
  // Orientation still resolves: nu02 > nu20 -> theta = pi/2.
  CHECK(cvjit.orientation == Approx(static_cast<float>(kPi * 0.5)).margin(1e-6));
  CHECK(std::isfinite(cvjit.direction));

  const auto norm = run(cv::BlobFormula::Normalized);
  CHECK(std::isfinite(norm.elongation));
  CHECK(norm.elongation == Approx(1.f).margin(1e-6));
}

TEST_CASE("BlobStats degenerate single-pixel blob is NaN-elongation in cv.jit mode",
          "[blobstats]")
{
  // A single pixel has every central moment == 0, so cv.jit's elongation is 0/0 == NaN.
  auto run = [](cv::BlobFormula f) {
    cv::BlobStats obj;
    Image img(16, 16, 0);
    img.setGray(5, 5, 255);
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 0;
    obj.inputs.formula.value = f;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    return obj.outputs.blobs.value.front();
  };

  const auto cvjit = run(cv::BlobFormula::CvJit);
  CHECK(cvjit.mass == Approx(1.f));
  CHECK(cvjit.nu20 == 0.f);
  CHECK(cvjit.nu02 == 0.f);
  CHECK(std::isnan(cvjit.elongation));
  // nu20 == nu02 -> orientation short-circuits to exactly 0 (no atan of 0/0).
  CHECK(cvjit.orientation == 0.f);
  CHECK(cvjit.direction == 0.f);

  const auto norm = run(cv::BlobFormula::Normalized);
  CHECK(std::isfinite(norm.elongation));
  CHECK(norm.elongation == Approx(1.f).margin(1e-6));
  CHECK(std::isfinite(norm.orientation));
  CHECK(std::isfinite(norm.direction));
}

TEST_CASE("BlobStats mass equals the blob pixel count", "[blobstats]")
{
  cv::BlobStats obj;
  Image img(40, 40, 0);
  img.fillRect(10, 14, 8, 8, 255); // 8*8 = 64 px

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  CHECK(b.mass == Approx(64.f));
}

TEST_CASE("BlobStats Hu[0] is positive and translation-invariant", "[blobstats]")
{
  auto hu0_for = [](int rx, int ry) {
    cv::BlobStats obj;
    Image img(64, 64, 0);
    img.fillRect(rx, ry, 10, 6, 255); // same shape, different position
    feed(obj.inputs.image, img);
    obj.inputs.threshold.value = 0.5f;
    obj.inputs.min_size.value = 4;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    return obj.outputs.blobs.value.front().hu;
  };

  auto a = hu0_for(5, 5);
  auto b = hu0_for(40, 30);
  CHECK(a[0] > 0.f);
  // Hu invariants are translation-invariant -> identical (up to fp noise) for a
  // translated copy of the same shape.
  CHECK(a[0] == Approx(b[0]).epsilon(1e-4));
  CHECK(a[1] == Approx(b[1]).epsilon(1e-4));
}

TEST_CASE("BlobStats degrees toggle scales angles by 180/pi", "[blobstats]")
{
  Image img(32, 32, 0);
  // Asymmetric blob: orientation ~2.916 rad and direction ~6.057 rad, so the scaling is
  // observable (a symmetric horizontal bar would give 0 == 0 * k and prove nothing).
  drawAsymBlob(img);

  cv::BlobStats radObj;
  feed(radObj.inputs.image, img);
  radObj.inputs.threshold.value = 0.5f;
  radObj.inputs.min_size.value = 4;
  radObj.inputs.degrees.value = false;
  radObj();
  REQUIRE(radObj.outputs.count.value == 1);
  const auto rad = radObj.outputs.blobs.value.front();

  cv::BlobStats degObj;
  feed(degObj.inputs.image, img);
  degObj.inputs.threshold.value = 0.5f;
  degObj.inputs.min_size.value = 4;
  degObj.inputs.degrees.value = true;
  degObj();
  REQUIRE(degObj.outputs.count.value == 1);
  const auto deg = degObj.outputs.blobs.value.front();

  const float k = 180.f / 3.14159265358979323846f;
  CHECK(deg.orientation == Approx(rad.orientation * k).margin(1e-3));
  CHECK(deg.direction == Approx(rad.direction * k).margin(1e-3));
}

// ------------------------------------------------------------------------------ BlobSort
TEST_CASE("BlobSort keeps a stable ID for a moving blob", "[blobsort]")
{
  cv::BlobSort obj;
  clearTexture(obj.inputs.image);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj.inputs.max_distance.value = 0.3f;

  Image f0(48, 48, 0);
  f0.fillRect(10, 10, 6, 6, 255);
  feed(obj.inputs.image, f0);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  int id0 = obj.outputs.blobs.value.front().id;

  // Move the blob a little; ID must persist.
  Image f1(48, 48, 0);
  f1.fillRect(13, 12, 6, 6, 255);
  feed(obj.inputs.image, f1);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.blobs.value.front().id == id0);
}

TEST_CASE("BlobSort assigns a new ID to a new blob", "[blobsort]")
{
  cv::BlobSort obj;
  clearTexture(obj.inputs.image);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj.inputs.max_distance.value = 0.1f;

  Image f0(48, 48, 0);
  f0.fillRect(5, 5, 5, 5, 255);
  feed(obj.inputs.image, f0);
  obj();
  int id0 = obj.outputs.blobs.value.front().id;

  // Second frame: original blob plus a brand-new far-away one.
  Image f1(48, 48, 0);
  f1.fillRect(5, 5, 5, 5, 255);
  f1.fillRect(38, 38, 5, 5, 255);
  feed(obj.inputs.image, f1);
  obj();
  REQUIRE(obj.outputs.count.value == 2);
  // The two IDs differ, and the original is preserved somewhere.
  int a = obj.outputs.blobs.value[0].id;
  int b = obj.outputs.blobs.value[1].id;
  CHECK(a != b);
  CHECK((a == id0 || b == id0));
}

TEST_CASE("BlobSort age increments for a persistent blob", "[blobsort]")
{
  cv::BlobSort obj;
  clearTexture(obj.inputs.image);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj.inputs.max_distance.value = 0.3f;

  // Frame 0: new blob -> age 0.
  Image f0(48, 48, 0);
  f0.fillRect(10, 10, 6, 6, 255);
  feed(obj.inputs.image, f0);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.blobs.value.front().age == 0);

  // Frame 1: same blob moved slightly -> age 1.
  Image f1(48, 48, 0);
  f1.fillRect(12, 11, 6, 6, 255);
  feed(obj.inputs.image, f1);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.blobs.value.front().age == 1);

  // Frame 2: still the same blob -> age 2.
  Image f2(48, 48, 0);
  f2.fillRect(14, 12, 6, 6, 255);
  feed(obj.inputs.image, f2);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.blobs.value.front().age == 2);
}

TEST_CASE("BlobSort recycles the lowest freed id after a blob vanishes", "[blobsort]")
{
  cv::BlobSort obj;
  clearTexture(obj.inputs.image);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj.inputs.max_distance.value = 0.05f; // small -> no accidental matches

  // Frame 0: two blobs -> ids 1 and 2 (in scan order).
  Image f0(48, 48, 0);
  f0.fillRect(5, 5, 5, 5, 255);
  f0.fillRect(38, 38, 5, 5, 255);
  feed(obj.inputs.image, f0);
  obj();
  REQUIRE(obj.outputs.count.value == 2);
  int idLow = obj.outputs.blobs.value[0].id;
  int idHigh = obj.outputs.blobs.value[1].id;
  CHECK(idLow == 1);
  CHECK(idHigh == 2);

  // Frame 1: the first blob disappears, second stays -> id 1 is freed.
  Image f1(48, 48, 0);
  f1.fillRect(38, 38, 5, 5, 255);
  feed(obj.inputs.image, f1);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.blobs.value.front().id == idHigh);

  // Frame 2: a brand-new blob appears far from the survivor. Without recycling it
  // would get id 3; with recycling it must reuse the lowest freed id (1).
  Image f2(48, 48, 0);
  f2.fillRect(38, 38, 5, 5, 255); // survivor
  f2.fillRect(5, 20, 5, 5, 255);  // newcomer
  feed(obj.inputs.image, f2);
  obj();
  REQUIRE(obj.outputs.count.value == 2);
  bool found_recycled = false;
  for(const auto& bl : obj.outputs.blobs.value)
    if(bl.id == idLow)
      found_recycled = true;
  CHECK(found_recycled); // freed id 1 was reused, not 3
}
