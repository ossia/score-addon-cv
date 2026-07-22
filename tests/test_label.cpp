// Tests for cv.jit.label's restored modes (connectivity / mode / charmode / threshold /
// blob cap) and for the standalone, chainable cv.jit.blobs.* objects in BlobsChain.hpp.
//
// Everything asserted here is hand-derived from the cv.jit sources:
//   cv.jit/source/projects/cv.jit.label/cv.jit.label.cpp
//   cv.jit/source/projects/cv.jit.blobs.{orientation,elongation,direction,bounds,centroids}
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "ScoreTextureModel.hpp"
#include "TestImage.hpp"

#include <CV/Cpu/BlobStats.hpp>
#include <CV/Cpu/BlobsChain.hpp>
#include <CV/Cpu/Label.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using namespace cvtest;

namespace
{
// Read Label's r32f "Labels" data field at a pixel, in cv.jit output-matrix units.
//
// The texture itself carries [0,1] — an r32f output must, because score converts it to RGBA8
// by interpreting the float as if it already were in that range (see the contract at the top
// of CV/Cpu/CartoPol.hpp; raw ids meant every blob arrived as 255). `Max label` is the
// divisor that was applied, so multiplying by it recovers the id / mass / rank.
float lbl_at(cv::Label& obj, int x, int y)
{
  auto& t = obj.outputs.labels.texture;
  const float* f = reinterpret_cast<const float*>(t.bytes);
  return f[static_cast<std::size_t>(y) * t.width + x] * obj.outputs.max_label.value;
}

// The raw, normalised value actually stored in the texture.
float lbl_raw(cv::Label& obj, int x, int y)
{
  auto& t = obj.outputs.labels.texture;
  const float* f = reinterpret_cast<const float*>(t.bytes);
  return f[static_cast<std::size_t>(y) * t.width + x];
}
}

// =================================================================== connectivity
//
// THE discriminator between the port (8-connected union-find) and cv.jit (a 4-connected
// scanline seed-fill: fillBlobLong() only ever walks left/right along a row and steps one
// row up or down, never diagonally). A staircase of corner-touching pixels is therefore
// ONE blob at 8-connectivity and N blobs at 4.
TEST_CASE("Label connectivity: a diagonal chain is 1 blob at 8 and N at 4", "[label]")
{
  cv::Label obj;
  Image img(8, 8, 0);
  img.setGray(1, 1, 255);
  img.setGray(2, 2, 255);
  img.setGray(3, 3, 255);
  img.setGray(4, 4, 255);
  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;

  SECTION("default is 8-connected (backwards compatible)")
  {
    obj();
    CHECK(obj.outputs.count.value == 1);
    // All four pixels carry the same id.
    CHECK(lbl_at(obj, 1, 1) == Approx(1.f));
    CHECK(lbl_at(obj, 4, 4) == Approx(1.f));
  }

  SECTION("Four splits the chain into one blob per pixel (cv.jit)")
  {
    obj.inputs.connectivity.value = cv::Connectivity::Four;
    obj();
    CHECK(obj.outputs.count.value == 4);
    // Raster order of first pixel -> id.
    CHECK(lbl_at(obj, 1, 1) == Approx(1.f));
    CHECK(lbl_at(obj, 2, 2) == Approx(2.f));
    CHECK(lbl_at(obj, 3, 3) == Approx(3.f));
    CHECK(lbl_at(obj, 4, 4) == Approx(4.f));
  }
}

TEST_CASE("Label connectivity: a 4-connected cross is 1 blob in both modes", "[label]")
{
  cv::Label obj;
  Image img(8, 8, 0);
  img.setGray(3, 2, 255);
  img.setGray(2, 3, 255);
  img.setGray(3, 3, 255);
  img.setGray(4, 3, 255);
  img.setGray(3, 4, 255);
  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;

  obj();
  CHECK(obj.outputs.count.value == 1);

  feed(obj.inputs.image, img);
  obj.inputs.connectivity.value = cv::Connectivity::Four;
  obj();
  CHECK(obj.outputs.count.value == 1);
}

// ========================================================================== mode
//
// cv.jit long output, mode 1: "Label each blob with its mass" -- every pixel of a blob
// carries that blob's pixel count.
TEST_CASE("Label mode Mass writes each blob's area into its pixels", "[label]")
{
  cv::Label obj;
  Image img(16, 16, 0);
  img.fillRect(2, 2, 2, 2, 255); // 4 px, id 1
  img.fillRect(8, 8, 3, 3, 255); // 9 px, id 2

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.mode.value = cv::LabelMode::Mass;
  obj();

  CHECK(obj.outputs.count.value == 2);
  CHECK(lbl_at(obj, 2, 2) == Approx(4.f));
  CHECK(lbl_at(obj, 3, 3) == Approx(4.f));
  CHECK(lbl_at(obj, 8, 8) == Approx(9.f));
  CHECK(lbl_at(obj, 10, 10) == Approx(9.f));
  CHECK(lbl_at(obj, 0, 0) == Approx(0.f)); // background
}

TEST_CASE("Label mode Sequential is unchanged by the new attributes", "[label]")
{
  cv::Label obj;
  Image img(16, 16, 0);
  img.fillRect(2, 2, 2, 2, 255);
  img.fillRect(8, 8, 3, 3, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj(); // all defaults

  CHECK(obj.outputs.count.value == 2);
  CHECK(lbl_at(obj, 2, 2) == Approx(1.f));
  CHECK(lbl_at(obj, 8, 8) == Approx(2.f));
  CHECK(obj.outputs.overflow.value == false);
}

// ====================================================================== charmode
//
// cv.jit char output, mode 1: the value is a SIZE RANK. cv.jit qsorts blobs ascending by
// size and writes ndx-i+1 for the blob at 1-based sorted position i, so the LARGEST blob
// is rank 1 and the smallest is rank ndx. A textbook "rank by size" implementation that
// numbered the smallest 1 would fail this test.
TEST_CASE("Label charmode rank: largest blob is rank 1", "[label]")
{
  cv::Label obj;
  Image img(16, 8, 0);
  img.setGray(1, 1, 255);        // id 1, 1 px
  img.fillRect(4, 1, 2, 2, 255); // id 2, 4 px
  img.fillRect(8, 1, 3, 3, 255); // id 3, 9 px

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.charmode.value = true;
  obj.inputs.mode.value = cv::LabelMode::Mass;
  obj();

  REQUIRE(obj.outputs.count.value == 3);
  CHECK(lbl_at(obj, 8, 1) == Approx(1.f)); // 9 px -> largest -> rank 1
  CHECK(lbl_at(obj, 4, 1) == Approx(2.f)); // 4 px
  CHECK(lbl_at(obj, 1, 1) == Approx(3.f)); // 1 px -> smallest -> rank ndx == 3
}

TEST_CASE("Label charmode rank: ranks beyond 255 clamp to 0", "[label]")
{
  // 300 isolated single pixels on a spacing-2 grid: 20 columns x 15 rows.
  // Raster order gives blob id = 20*j + i + 1 for the pixel at (2i, 2j).
  // All sizes are equal, so the stable sort keeps ascending id order and the blob at
  // sorted position i (0-based) gets rank 300 - i, i.e. blob id k gets rank 301 - k.
  // rank <= 255  <=>  k >= 46, so ids 46..300 (255 blobs) rank 1..255 and ids 1..45 -> 0.
  cv::Label obj;
  Image img(40, 30, 0);
  for(int j = 0; j < 15; ++j)
    for(int i = 0; i < 20; ++i)
      img.setGray(2 * i, 2 * j, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.charmode.value = true;
  obj.inputs.mode.value = cv::LabelMode::Mass;
  obj();

  REQUIRE(obj.outputs.count.value == 300);
  CHECK(obj.outputs.overflow.value == false);

  auto rank_of_id = [&](int k) {
    const int i = (k - 1) % 20, j = (k - 1) / 20;
    return lbl_at(obj, 2 * i, 2 * j);
  };

  CHECK(rank_of_id(300) == Approx(1.f));   // largest position -> rank 1
  CHECK(rank_of_id(46) == Approx(255.f));  // last rank that fits a byte
  CHECK(rank_of_id(45) == Approx(0.f));    // rank 256 -> clamped
  CHECK(rank_of_id(1) == Approx(0.f));     // rank 300 -> clamped

  int nonzero = 0;
  for(int k = 1; k <= 300; ++k)
    if(rank_of_id(k) != 0.f)
      ++nonzero;
  CHECK(nonzero == 255);
}

TEST_CASE("Label charmode Sequential clamps indices beyond 255 to 0", "[label]")
{
  // Same 300-blob grid, mode 0: cv.jit's char branch emits the index but writes 0 when
  // it exceeds 255.
  cv::Label obj;
  Image img(40, 30, 0);
  for(int j = 0; j < 15; ++j)
    for(int i = 0; i < 20; ++i)
      img.setGray(2 * i, 2 * j, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.charmode.value = true;
  obj();

  REQUIRE(obj.outputs.count.value == 300);
  CHECK(lbl_at(obj, 0, 0) == Approx(1.f));       // id 1
  CHECK(lbl_at(obj, 2 * 14, 2 * 12) == Approx(255.f)); // id 20*12+14+1 = 255
  CHECK(lbl_at(obj, 2 * 15, 2 * 12) == Approx(0.f));   // id 256 -> 0
  CHECK(lbl_at(obj, 2 * 19, 2 * 14) == Approx(0.f));   // id 300 -> 0

  // ... while the long output keeps the exact index.
  feed(obj.inputs.image, img);
  obj.inputs.charmode.value = false;
  obj();
  CHECK(lbl_at(obj, 2 * 15, 2 * 12) == Approx(256.f));
  CHECK(lbl_at(obj, 2 * 19, 2 * 14) == Approx(300.f));
}

// ===================================================================== size filter
//
// cv.jit compares `blobs[i].size > thresh` -- STRICTLY greater -- so a blob of exactly
// `threshold` pixels is dropped. The port has always used >=; both are offered, and the
// boundary is where they differ.
TEST_CASE("Label size filter: > vs >= at the boundary size", "[label]")
{
  cv::Label obj;
  Image img(16, 16, 0);
  img.fillRect(2, 2, 3, 3, 255); // exactly 9 px

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 9;

  SECTION("AtLeast (default) keeps a blob of exactly min_size")
  {
    obj();
    CHECK(obj.outputs.count.value == 1);
    CHECK(lbl_at(obj, 2, 2) == Approx(1.f));
  }

  SECTION("GreaterThan (cv.jit) drops a blob of exactly min_size")
  {
    obj.inputs.size_filter.value = cv::SizeFilter::GreaterThan;
    obj();
    CHECK(obj.outputs.count.value == 0);
    CHECK(lbl_at(obj, 2, 2) == Approx(0.f));
  }
}

TEST_CASE("Label size filter: both rules agree at min_size 0", "[label]")
{
  cv::Label obj;
  Image img(16, 16, 0);
  img.setGray(3, 3, 255);
  img.fillRect(8, 8, 2, 2, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_size.value = 0;
  obj();
  CHECK(obj.outputs.count.value == 2);

  feed(obj.inputs.image, img);
  obj.inputs.size_filter.value = cv::SizeFilter::GreaterThan;
  obj();
  CHECK(obj.outputs.count.value == 2);
}

// ======================================================================= blob cap
//
// cv.jit hard-caps at 2048 blobs and silently discards the rest (its scan loop is
// `for(j = 0; j < width && ndx < 2048; j++)`, so once the cap is hit no new seed is ever
// started). Same rule here, but the loss is reported on `Overflow`.
TEST_CASE("Label blob cap overflow is observable", "[label]")
{
  cv::Label obj;
  Image img(10, 3, 0);
  for(int i = 0; i < 5; ++i)
    img.setGray(2 * i, 0, 255); // ids 1..5 in raster order

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;

  SECTION("no overflow under the default cap")
  {
    obj();
    CHECK(obj.outputs.count.value == 5);
    CHECK(obj.outputs.overflow.value == false);
    CHECK(lbl_at(obj, 8, 0) == Approx(5.f));
  }

  SECTION("cap 3 keeps the first three and drops the rest")
  {
    obj.inputs.max_blobs.value = 3;
    obj();
    CHECK(obj.outputs.count.value == 3);
    CHECK(obj.outputs.overflow.value == true);
    CHECK(lbl_at(obj, 0, 0) == Approx(1.f));
    CHECK(lbl_at(obj, 2, 0) == Approx(2.f));
    CHECK(lbl_at(obj, 4, 0) == Approx(3.f));
    // Dropped blobs become background, exactly as in cv.jit.
    CHECK(lbl_at(obj, 6, 0) == Approx(0.f));
    CHECK(lbl_at(obj, 8, 0) == Approx(0.f));
    // ... and their visualization pixels are black.
    auto& vis = obj.outputs.image.texture;
    CHECK(vis.bytes[(0 * 10 + 6) * 4 + 0] == 0);
    CHECK(vis.bytes[(0 * 10 + 8) * 4 + 1] == 0);
  }
}

// ===================================================================== edge cases
TEST_CASE("Label edge cases", "[label]")
{
  SECTION("empty image")
  {
    cv::Label obj;
    Image img(16, 16, 0);
    feed(obj.inputs.image, img);
    obj();
    CHECK(obj.outputs.count.value == 0);
    CHECK(obj.outputs.overflow.value == false);
    CHECK(lbl_at(obj, 8, 8) == Approx(0.f));
  }

  SECTION("single pixel")
  {
    cv::Label obj;
    Image img(8, 8, 0);
    img.setGray(5, 6, 255);
    feed(obj.inputs.image, img);
    obj();
    CHECK(obj.outputs.count.value == 1);
    CHECK(lbl_at(obj, 5, 6) == Approx(1.f));

    // mass mode: the lone pixel carries mass 1; rank mode: it is rank 1.
    feed(obj.inputs.image, img);
    obj.inputs.mode.value = cv::LabelMode::Mass;
    obj();
    CHECK(lbl_at(obj, 5, 6) == Approx(1.f));

    feed(obj.inputs.image, img);
    obj.inputs.charmode.value = true;
    obj();
    CHECK(lbl_at(obj, 5, 6) == Approx(1.f));

    // DELIBERATE DEVIATION: cv.jit special-cases ndx == 1 as `equiv[1] = 1` and skips the
    // size test, so a lone under-sized blob is still ranked 1. Here the filter applies.
    feed(obj.inputs.image, img);
    obj.inputs.min_size.value = 2;
    obj();
    CHECK(obj.outputs.count.value == 0);
    CHECK(lbl_at(obj, 5, 6) == Approx(0.f));
  }

  SECTION("all-white image is a single blob of W*H pixels")
  {
    cv::Label obj;
    Image img(4, 4, 255);
    feed(obj.inputs.image, img);
    obj.inputs.mode.value = cv::LabelMode::Mass;
    obj();
    CHECK(obj.outputs.count.value == 1);
    CHECK(lbl_at(obj, 0, 0) == Approx(16.f));
    CHECK(lbl_at(obj, 3, 3) == Approx(16.f));

    // Same at 4-connectivity.
    feed(obj.inputs.image, img);
    obj.inputs.connectivity.value = cv::Connectivity::Four;
    obj();
    CHECK(obj.outputs.count.value == 1);
    CHECK(lbl_at(obj, 3, 3) == Approx(16.f));
  }

  SECTION("1x1")
  {
    cv::Label white;
    Image w(1, 1, 255);
    feed(white.inputs.image, w);
    white();
    CHECK(white.outputs.count.value == 1);
    CHECK(lbl_at(white, 0, 0) == Approx(1.f));

    cv::Label black;
    Image b(1, 1, 0);
    feed(black.inputs.image, b);
    black();
    CHECK(black.outputs.count.value == 0);
    CHECK(lbl_at(black, 0, 0) == Approx(0.f));
  }

  SECTION("dimension change mid-stream")
  {
    cv::Label obj;
    Image a(8, 8, 0);
    a.fillRect(1, 1, 2, 2, 255);
    feed(obj.inputs.image, a);
    obj();
    REQUIRE(obj.outputs.count.value == 1);

    Image b(16, 4, 0);
    b.fillRect(1, 1, 2, 2, 255);
    b.fillRect(8, 1, 2, 2, 255);
    feed(obj.inputs.image, b);
    obj();
    CHECK(obj.outputs.count.value == 2);
    CHECK(obj.outputs.labels.texture.width == 16);
    CHECK(obj.outputs.labels.texture.height == 4);
  }
}

// ============================================================ the r32f [0,1] contract
//
// score converts every r32f texture output to RGBA8 by interpreting the float as [0,1]
// (`gray = qBound(0, int(v*255.f), 255)`, Crousti/TextureConversion.hpp), and EVERY texture
// input in this addon is RGBA8. `Labels` used to carry raw integer ids, so every id >= 1
// arrived at the next object as 255: a 2-blob field and a 200-blob field were the same flat
// white rectangle. These tests pin the normalisation and the `Max label` divisor down.
TEST_CASE("Label's label field obeys the r32f [0,1] contract", "[label][contract]")
{
  SECTION("ids are spread over [0,1] and stay distinguishable through the conversion")
  {
    cv::Label obj;
    Image img(16, 3, 0);
    for(int i = 0; i < 4; ++i)          // 4 separated single pixels -> ids 1..4
      img.setGray(2 + 3 * i, 1, 255);
    feed(obj.inputs.image, img);
    obj();
    REQUIRE(obj.outputs.count.value == 4);
    REQUIRE(r32f_in_unit_range(obj.outputs.labels));
    CHECK(obj.outputs.max_label.value == Approx(4.f));

    // Raw texture: 0, 0.25, 0.5, 0.75, 1.0 — id/Max label.
    CHECK(lbl_raw(obj, 0, 0) == Approx(0.f));
    CHECK(lbl_raw(obj, 2, 1) == Approx(0.25f));
    CHECK(lbl_raw(obj, 5, 1) == Approx(0.5f));
    CHECK(lbl_raw(obj, 8, 1) == Approx(0.75f));
    CHECK(lbl_raw(obj, 11, 1) == Approx(1.f));
    // ...and the ids come back exactly.
    for(int i = 0; i < 4; ++i)
      CHECK(lbl_at(obj, 2 + 3 * i, 1) == Approx(float(i + 1)));

    // Through score's real conversion the four blobs are four DIFFERENT greys. With the
    // raw-id field they were all 255.
    Image conv = score_r32f_to_rgba8(obj.outputs.labels);
    const int g[4]
        = {conv.px[(1 * 16 + 2) * 4], conv.px[(1 * 16 + 5) * 4],
           conv.px[(1 * 16 + 8) * 4], conv.px[(1 * 16 + 11) * 4]};
    CHECK(g[0] == 63);  // int(0.25*255) = int(63.75)
    CHECK(g[1] == 127); // int(0.50*255) = int(127.5)
    CHECK(g[2] == 191); // int(0.75*255) = int(191.25)
    CHECK(g[3] == 255);
    CHECK(int(conv.px[0]) == 0); // background stays black
  }

  SECTION("an all-background frame reports Max label 0 and an all-zero field")
  {
    cv::Label obj;
    Image img(8, 8, 0);
    feed(obj.inputs.image, img);
    obj();
    CHECK(obj.outputs.count.value == 0);
    CHECK(obj.outputs.max_label.value == Approx(0.f));
    REQUIRE(r32f_in_unit_range(obj.outputs.labels)); // no division by zero, no NaN
    for(int y = 0; y < 8; ++y)
      for(int x = 0; x < 8; ++x)
        REQUIRE(lbl_raw(obj, x, y) == 0.f);
  }

  SECTION("Mass mode: `Max label` is the biggest blob's pixel count")
  {
    // 9 px and 4 px blobs; in Mass mode the field carries the masses, so the divisor is 9.
    cv::Label obj;
    Image img(16, 8, 0);
    img.fillRect(1, 1, 3, 3, 255); // 9 px
    img.fillRect(9, 1, 2, 2, 255); // 4 px
    feed(obj.inputs.image, img);
    obj.inputs.mode.value = cv::LabelMode::Mass;
    obj();
    REQUIRE(obj.outputs.count.value == 2);
    REQUIRE(r32f_in_unit_range(obj.outputs.labels));
    CHECK(obj.outputs.max_label.value == Approx(9.f));
    CHECK(lbl_raw(obj, 2, 2) == Approx(1.f));
    CHECK(lbl_raw(obj, 9, 1) == Approx(4.f / 9.f));
    CHECK(lbl_at(obj, 2, 2) == Approx(9.f));
    CHECK(lbl_at(obj, 9, 1) == Approx(4.f));
  }

  SECTION("a large field: 300 ids still fit, where 8-bit output cannot")
  {
    // 300 single-pixel blobs on a 2x2 grid pitch. In long/Sequential mode cv.jit's values
    // run 1..300; the normalised field keeps them all distinct and recoverable, which is
    // exactly what the raw-r32f version could not do (everything >= 1 became white).
    cv::Label obj;
    Image img(40, 30, 0);
    for(int j = 0; j < 15; ++j)
      for(int i = 0; i < 20; ++i)
        img.setGray(2 * i, 2 * j, 255);
    feed(obj.inputs.image, img);
    obj();
    REQUIRE(obj.outputs.count.value == 300);
    REQUIRE(r32f_in_unit_range(obj.outputs.labels));
    CHECK(obj.outputs.max_label.value == Approx(300.f));
    CHECK(lbl_at(obj, 0, 0) == Approx(1.f));
    CHECK(lbl_at(obj, 2 * 19, 2 * 14) == Approx(300.f));
    CHECK(lbl_raw(obj, 2 * 19, 2 * 14) == Approx(1.f));
    CHECK(lbl_raw(obj, 0, 0) == Approx(1.f / 300.f));
  }
}

// ================================================================== the blobs chain
namespace
{
// A 2-blob image: a wide 8x4 rectangle and a tall 4x10 rectangle, well separated.
Image two_blob_image()
{
  Image img(32, 32, 0);
  img.fillRect(4, 4, 8, 4, 255);   // 32 px, wide  -> horizontal principal axis
  img.fillRect(20, 16, 4, 10, 255); // 40 px, tall -> vertical principal axis
  return img;
}
}

TEST_CASE("Blobs chain matches BlobStats' own per-blob values", "[blobschain]")
{
  cv::BlobStats bs;
  Image img = two_blob_image();
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs.inputs.min_size.value = 4;
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 2);

  auto run = [&](auto& obj) {
    obj.inputs.blobs.value = bs.outputs.blobs.value;
    obj();
  };

  cv::BlobsOrientation ori;
  cv::BlobsElongation elo;
  cv::BlobsDirection dir;
  run(ori);
  run(elo);
  run(dir);

  REQUIRE(ori.outputs.orientation.value.size() == 2);
  REQUIRE(elo.outputs.elongation.value.size() == 2);
  REQUIRE(dir.outputs.direction.value.size() == 2);

  for(std::size_t i = 0; i < 2; ++i)
  {
    const auto& b = bs.outputs.blobs.value[i];
    CHECK(ori.outputs.orientation.value[i] == Approx(b.orientation).margin(1e-5));
    CHECK(dir.outputs.direction.value[i] == Approx(b.direction).margin(1e-5));
    CHECK(elo.outputs.elongation.value[i] == Approx(b.elongation).epsilon(1e-4));
  }

  // The wide blob's cv.jit orientation is exactly 0 (nu11 == 0 for an axis-aligned
  // rectangle, nu20 > nu02 -> atan(0)/2 == 0); the tall one is pi/2 (nu20 < nu02 ->
  // 0 + pi/2). This is the cv.jit branch structure, not atan2.
  CHECK(ori.outputs.orientation.value[0] == Approx(0.f).margin(1e-6));
  CHECK(ori.outputs.orientation.value[1] == Approx(cv::blobs_chain::PI * 0.5).margin(1e-5));
}

TEST_CASE("Blobs chain: Normalized formula mode also agrees with BlobStats", "[blobschain]")
{
  cv::BlobStats bs;
  Image img = two_blob_image();
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs.inputs.formula.value = cv::BlobFormula::Normalized;
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 2);

  cv::BlobsOrientation ori;
  cv::BlobsElongation elo;
  ori.inputs.blobs.value = bs.outputs.blobs.value;
  ori.inputs.formula.value = cv::BlobFormula::Normalized;
  ori();
  elo.inputs.blobs.value = bs.outputs.blobs.value;
  elo.inputs.formula.value = cv::BlobFormula::Normalized;
  elo();

  for(std::size_t i = 0; i < 2; ++i)
  {
    const auto& b = bs.outputs.blobs.value[i];
    CHECK(ori.outputs.orientation.value[i] == Approx(b.orientation).margin(1e-5));
    CHECK(elo.outputs.elongation.value[i] == Approx(b.elongation).epsilon(1e-4));
  }
}

TEST_CASE("Blobs chain: degrees and flip", "[blobschain]")
{
  // two_blob_image()'s blobs are both AXIS-ALIGNED rectangles, so nu11 == 0 and their cv.jit
  // orientations are exactly 0 and pi/2. On the 0 blob the radians->degrees assertion reduces
  // to `0 == 0 * 180/pi`, which holds for ANY scale factor (including a missing one) and
  // proves nothing -- the same trap that was already removed from
  // tests/test_blobs.cpp "BlobStats degrees toggle scales angles by 180/pi".
  // A third, deliberately ASYMMETRIC blob (the L of drawAsymBlob) is added here: nu11 != 0,
  // orientation ~2.9159 rad ~ 167.07 deg, so the conversion is genuinely observable.
  cv::BlobStats bs;
  Image img(32, 32, 0);
  img.fillRect(4, 4, 8, 4, 255);    // 32 px, wide -> orientation 0
  img.fillRect(20, 16, 4, 10, 255); // 40 px, tall -> orientation pi/2
  img.fillRect(22, 4, 4, 1, 255);   // asymmetric L: (22..25, 4)
  img.fillRect(22, 5, 2, 1, 255);   //               (22..23, 5)
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs.inputs.min_size.value = 0; // the L is only 6 px
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 3);

  cv::BlobsOrientation rad, deg;
  rad.inputs.blobs.value = bs.outputs.blobs.value;
  rad();
  deg.inputs.blobs.value = bs.outputs.blobs.value;
  deg.inputs.degrees.value = true;
  deg();

  int nontrivial = 0;
  for(std::size_t i = 0; i < 3; ++i)
  {
    INFO("blob " << i << " rad = " << rad.outputs.orientation.value[i]);
    CHECK(
        deg.outputs.orientation.value[i]
        == Approx(rad.outputs.orientation.value[i] * 180.0 / cv::blobs_chain::PI)
               .margin(1e-4));
    if(std::abs(rad.outputs.orientation.value[i]) > 1e-3f)
      ++nontrivial;
  }
  // ...and the scaling was actually exercised on a non-zero angle (otherwise every check
  // above would be 0 == 0 * k). The L gives ~2.916 rad, the tall bar pi/2.
  CHECK(nontrivial == 2);

  // flip shifts the direction by exactly pi (sign chosen so it stays near the range).
  cv::BlobsDirection d0, d1;
  d0.inputs.blobs.value = bs.outputs.blobs.value;
  d0();
  d1.inputs.blobs.value = bs.outputs.blobs.value;
  d1.inputs.flip.value = true;
  d1();
  for(std::size_t i = 0; i < 3; ++i)
    CHECK(
        std::abs(d1.outputs.direction.value[i] - d0.outputs.direction.value[i])
        == Approx(cv::blobs_chain::PI).margin(1e-5));
}

// cv.jit's elongation is NOT an aspect ratio: a square blob gives exactly 0, not 1.
// A textbook sqrt(lambda1/lambda2) implementation would return 1 here and fail.
TEST_CASE("Blobs chain: elongation of a square is 0 (cv.jit, not an aspect ratio)",
          "[blobschain]")
{
  cv::BlobStats bs;
  Image img(32, 32, 0);
  img.fillRect(8, 8, 6, 6, 255); // square -> nu20 == nu02, nu11 == 0
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 1);

  cv::BlobsElongation elo;
  elo.inputs.blobs.value = bs.outputs.blobs.value;
  elo();
  REQUIRE(elo.outputs.elongation.value.size() == 1);
  CHECK(elo.outputs.elongation.value[0] == Approx(0.f).margin(1e-9));

  cv::BlobsOrientation ori;
  ori.inputs.blobs.value = bs.outputs.blobs.value;
  ori();
  CHECK(ori.outputs.orientation.value[0] == 0.f); // exactly 0: the nu20 == nu02 branch

  // The Normalized mode is the textbook one and reports 1 for the same square.
  cv::BlobsElongation eloN;
  eloN.inputs.blobs.value = bs.outputs.blobs.value;
  eloN.inputs.formula.value = cv::BlobFormula::Normalized;
  eloN();
  CHECK(eloN.outputs.elongation.value[0] == Approx(1.f).epsilon(1e-5));
}

// A one-pixel-wide blob has nu20 == 0 exactly, so cv.jit's elongation quotient is +inf.
TEST_CASE("Blobs chain: a 1-pixel-wide blob elongates to +inf (cv.jit)", "[blobschain]")
{
  cv::BlobStats bs;
  Image img(16, 16, 0);
  img.fillRect(5, 3, 1, 6, 255); // 1 wide, 6 tall -> mu20 == 0
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 1);

  cv::BlobsElongation elo;
  elo.inputs.blobs.value = bs.outputs.blobs.value;
  elo();
  CHECK(std::isinf(elo.outputs.elongation.value[0]));

  cv::BlobsElongation eloN;
  eloN.inputs.blobs.value = bs.outputs.blobs.value;
  eloN.inputs.formula.value = cv::BlobFormula::Normalized;
  eloN();
  CHECK(std::isfinite(eloN.outputs.elongation.value[0]));
}

TEST_CASE("Blobs chain: empty blob list yields empty outputs", "[blobschain]")
{
  cv::BlobsOrientation ori;
  cv::BlobsElongation elo;
  cv::BlobsDirection dir;
  cv::BlobsBounds bnd;
  cv::BlobsCentroids cen;
  ori();
  elo();
  dir();
  bnd();
  cen();
  CHECK(ori.outputs.orientation.value.empty());
  CHECK(elo.outputs.elongation.value.empty());
  CHECK(dir.outputs.direction.value.empty());
  CHECK(bnd.outputs.bounds.value.empty());
  CHECK(cen.outputs.centroids.value.empty());

  bnd.inputs.label_indexed.value = true;
  cen.inputs.label_indexed.value = true;
  bnd();
  cen();
  CHECK(bnd.outputs.bounds.value.empty());
  CHECK(cen.outputs.centroids.value.empty());
}

// ================================================== bounds / centroids in pixel space
TEST_CASE("BlobsBounds emits inclusive pixel bounding boxes", "[blobschain]")
{
  cv::BlobStats bs;
  Image img(32, 32, 0);
  img.fillRect(4, 6, 8, 5, 255); // x in [4,11], y in [6,10], 40 px
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 1);

  cv::BlobsBounds bnd;
  bnd.inputs.blobs.value = bs.outputs.blobs.value;
  bnd.inputs.width.value = 32;
  bnd.inputs.height.value = 32;
  bnd();

  REQUIRE(bnd.outputs.bounds.value.size() == 1);
  const auto& r = bnd.outputs.bounds.value[0];
  CHECK(r.left == 4);
  CHECK(r.top == 6);
  CHECK(r.right == 11);
  CHECK(r.bottom == 10);
}

TEST_CASE("BlobsCentroids emits pixel centroids and raw mass", "[blobschain]")
{
  cv::BlobStats bs;
  Image img(32, 32, 0);
  img.fillRect(4, 6, 8, 5, 255); // centroid ((4+11)/2, (6+10)/2) = (7.5, 8), mass 40
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs();
  REQUIRE(bs.outputs.blobs.value.size() == 1);

  cv::BlobsCentroids cen;
  cen.inputs.blobs.value = bs.outputs.blobs.value;
  cen.inputs.width.value = 32;
  cen.inputs.height.value = 32;
  cen();

  REQUIRE(cen.outputs.centroids.value.size() == 1);
  CHECK(cen.outputs.centroids.value[0].x == Approx(7.5f));
  CHECK(cen.outputs.centroids.value[0].y == Approx(8.0f));
  CHECK(cen.outputs.centroids.value[0].mass == Approx(40.f));
}

// cv.jit's downstream objects address rows BY LABEL, so a missing label must still take a
// row -- filled with the sentinel each object initialises its tables with:
//   bounds:    left/top = 0x7FFFFFFF, right/bottom = 0
//   centroids: (-1, -1, 0)
TEST_CASE("Bounds/centroids sentinel gap rows for a missing label", "[blobschain]")
{
  // Hand-built list with a hole at label 2.
  std::vector<cv::blob_info> blobs;
  {
    cv::blob_info b{};
    b.id = 1;
    b.centroid = {0.25f, 0.5f};
    b.bbox = cv::rect{0.125f, 0.25f, 0.25f, 0.5f};
    b.mass = 10.f;
    blobs.push_back(b);
    b.id = 3;
    b.centroid = {0.75f, 0.25f};
    b.bbox = cv::rect{0.5f, 0.125f, 0.25f, 0.25f};
    b.mass = 20.f;
    blobs.push_back(b);
  }

  cv::BlobsBounds bnd;
  bnd.inputs.blobs.value = blobs;
  bnd.inputs.width.value = 64;
  bnd.inputs.height.value = 32;

  SECTION("compact (default) skips the hole")
  {
    bnd();
    REQUIRE(bnd.outputs.bounds.value.size() == 2);
    CHECK(bnd.outputs.bounds.value[0].left == 8);   // 0.125 * 64
    CHECK(bnd.outputs.bounds.value[1].left == 32);  // 0.5 * 64
  }

  SECTION("label indexed inserts the sentinel row")
  {
    bnd.inputs.label_indexed.value = true;
    bnd();
    REQUIRE(bnd.outputs.bounds.value.size() == 3);
    // label 1
    CHECK(bnd.outputs.bounds.value[0].left == 8);
    CHECK(bnd.outputs.bounds.value[0].top == 8);     // 0.25 * 32
    CHECK(bnd.outputs.bounds.value[0].right == 8 + 16 - 1);  // w = 0.25 * 64 = 16
    CHECK(bnd.outputs.bounds.value[0].bottom == 8 + 16 - 1); // h = 0.5 * 32 = 16
    // label 2: absent -> sentinel
    CHECK(bnd.outputs.bounds.value[1].left == std::numeric_limits<int>::max());
    CHECK(bnd.outputs.bounds.value[1].top == std::numeric_limits<int>::max());
    CHECK(bnd.outputs.bounds.value[1].right == 0);
    CHECK(bnd.outputs.bounds.value[1].bottom == 0);
    // label 3
    CHECK(bnd.outputs.bounds.value[2].left == 32);
    CHECK(bnd.outputs.bounds.value[2].top == 4); // 0.125 * 32
  }

  cv::BlobsCentroids cen;
  cen.inputs.blobs.value = blobs;
  cen.inputs.width.value = 64;
  cen.inputs.height.value = 32;

  SECTION("centroids compact")
  {
    cen();
    REQUIRE(cen.outputs.centroids.value.size() == 2);
    CHECK(cen.outputs.centroids.value[0].x == Approx(16.f)); // 0.25 * 64
    CHECK(cen.outputs.centroids.value[0].y == Approx(16.f)); // 0.5 * 32
    CHECK(cen.outputs.centroids.value[1].mass == Approx(20.f));
  }

  SECTION("centroids label indexed inserts the (-1,-1,0) sentinel row")
  {
    cen.inputs.label_indexed.value = true;
    cen();
    REQUIRE(cen.outputs.centroids.value.size() == 3);
    CHECK(cen.outputs.centroids.value[0].x == Approx(16.f));
    CHECK(cen.outputs.centroids.value[1].x == Approx(-1.f));
    CHECK(cen.outputs.centroids.value[1].y == Approx(-1.f));
    CHECK(cen.outputs.centroids.value[1].mass == Approx(0.f));
    CHECK(cen.outputs.centroids.value[2].x == Approx(48.f)); // 0.75 * 64
    CHECK(cen.outputs.centroids.value[2].mass == Approx(20.f));
  }
}

// Full path: Label -> BlobStats -> chain, on the same image, must agree on blob count.
TEST_CASE("Label and the blobs chain agree on a 2-blob image", "[blobschain][label]")
{
  Image img = two_blob_image();

  cv::Label lab;
  feed(lab.inputs.image, img);
  lab.inputs.threshold.value = 0.5f;
  lab();

  cv::BlobStats bs;
  feed(bs.inputs.image, img);
  bs.inputs.threshold.value = 0.5f;
  bs();

  REQUIRE(lab.outputs.count.value == 2);
  REQUIRE(bs.outputs.count.value == 2);

  cv::BlobsCentroids cen;
  cen.inputs.blobs.value = bs.outputs.blobs.value;
  cen.inputs.width.value = 32;
  cen.inputs.height.value = 32;
  cen();
  REQUIRE(cen.outputs.centroids.value.size() == 2);
  // Blob A: x in [4,11] -> 7.5 ; y in [4,7] -> 5.5 ; mass 32
  CHECK(cen.outputs.centroids.value[0].x == Approx(7.5f));
  CHECK(cen.outputs.centroids.value[0].y == Approx(5.5f));
  CHECK(cen.outputs.centroids.value[0].mass == Approx(32.f));
  // Blob B: x in [20,23] -> 21.5 ; y in [16,25] -> 20.5 ; mass 40
  CHECK(cen.outputs.centroids.value[1].x == Approx(21.5f));
  CHECK(cen.outputs.centroids.value[1].y == Approx(20.5f));
  CHECK(cen.outputs.centroids.value[1].mass == Approx(40.f));
}
