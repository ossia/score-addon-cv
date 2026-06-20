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
  // a square is not elongated
  CHECK(b.elongation == Approx(1.f).margin(0.3));
}

TEST_CASE("BlobStats orientation/elongation for a horizontal bar", "[blobstats]")
{
  cv::BlobStats obj;
  Image img(48, 48, 0);
  img.fillRect(8, 22, 30, 4, 255); // wide, short bar -> elongated, ~horizontal

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 4;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto& b = obj.outputs.blobs.value.front();
  CHECK(b.elongation > 2.f); // clearly elongated
  // principal axis ~ horizontal -> orientation near 0 (or +-pi)
  CHECK(std::abs(std::sin(b.orientation)) < 0.3f);
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
  Image img(48, 48, 0);
  img.fillRect(8, 22, 30, 4, 255); // elongated bar with a well-defined orientation

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
