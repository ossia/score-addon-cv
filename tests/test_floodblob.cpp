// Tests for the two cv.jit-parity completions: FloodFill's Binary mode + pixel seed, and
// BlobSort's chainable list input + pixel-unit latching distance.
//
// Derivations used below (all hand-computed, none read off an implementation):
//
// FloodFill
// ---------
//  * Rec.601 luma of a pure grey (v,v,v) is 0.299v + 0.587v + 0.114v = 1.000*v = v, so
//    "luma" and "grey level" are interchangeable for every image built here.
//  * Binary mode foreground test is `luma > Threshold*255`. With Threshold == 0 that is
//    exactly cv.jit's `in[p] != 0`, so grey 40 and grey 200 are both foreground while
//    grey 0 is background.
//  * Tolerance mode test is `|luma - luma(seed)| <= Tolerance*255`. With Tolerance = 0.1
//    the half-window is 25.5 grey levels: seeded on grey 200, grey 200 is in, grey 40
//    (distance 160) and grey 0 (distance 200) are out.
//  * Normalised seed -> pixel mapping is `clamp(int(sx * width), 0, width-1)`. On a 16-wide
//    image, 6/16 = 0.375 -> int(6.0) = 6 and 7/16 = 0.4375 -> int(7.0) = 7.
//
// BlobSort
// --------
//  * A w x h axis-aligned block whose top-left pixel is (x0,y0) has its centroid at pixel
//    (x0 + (w-1)/2, y0 + (h-1)/2). A 6x6 block at (10,10) is centred on (12.5, 12.5); the
//    same block at (16,10) on (18.5, 12.5). Displacement: 6 px in x, 0 in y.
//  * BlobStats/BlobSort centroids are normalised by the frame size, so on a 48x48 frame the
//    same displacement is 6/48 = 0.125 in normalised units.
//  * Hence, for a square frame, `Max distance (px) = Max distance * 48` is the exact
//    equivalence between the two unit systems. 0.2 <-> 9.6 both accept the 6 px move;
//    0.1 <-> 4.8 both reject it.
//  * A rejected move is *not* observable through the id (the vanished blob frees id 1 and
//    the newcomer immediately recycles it) but is observable through `age`: a matched blob
//    reports the previous age + 1, a brand-new one reports 0.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/BlobSort.hpp>
#include <CV/Cpu/BlobStats.hpp>
#include <CV/Cpu/FloodFill.hpp>

#include <algorithm>
#include <cstdint>

using Catch::Approx;
using namespace cvtest;

namespace
{
// Count the non-zero bytes of an r8 output.
int countMask(const cv::FloodFill& obj)
{
  const auto& t = obj.outputs.image.texture;
  int n = 0;
  for(int i = 0; i < t.width * t.height; ++i)
    if(t.bytes[i] != 0)
      ++n;
  return n;
}

cv::blob_info makeBlob(float cx, float cy, float area = 0.01f)
{
  cv::blob_info b{};
  b.centroid = {cx, cy};
  b.area = area;
  return b;
}
}

// =============================================================================== FloodFill

TEST_CASE("FloodFill defaults are the legacy ones", "[floodfill][compat]")
{
  // Documents the deliberate deviation from cv.jit: cv.jit's operation is the Binary one,
  // but Binary cannot be the default here without breaking
  // tests/test_main_objects.cpp's "FloodFill on background fills the complement", which
  // seeds a black pixel and expects the background to be filled -- impossible under
  // cv.jit semantics where a zero seed is a no-op. If someone flips the enum order this
  // test fires before the other suite does.
  cv::FloodFill obj;
  CHECK(obj.inputs.mode.value == cv::FloodMode::Tolerance);
  CHECK(obj.inputs.seed_mode.value == cv::FloodSeedMode::Normalized);
  CHECK(obj.inputs.seed_px.value.x == 0); // cv.jit's `seed` default is `0 0`
  CHECK(obj.inputs.seed_px.value.y == 0);
}

TEST_CASE("FloodFill Binary spans varying intensities where Tolerance does not",
          "[floodfill][binary]")
{
  // A single connected 10-pixel horizontal bar on row 8 of a 16x16 black image:
  //   x = 2..5  -> grey 200   (4 pixels, contains the seed)
  //   x = 6..11 -> grey 40    (6 pixels)
  // Binary  (Threshold 0): every non-zero pixel joins   -> 10.
  // Tolerance (0.1 = +-25.5): only the grey-200 run     -> 4.
  Image img(16, 16, 0);
  for(int x = 2; x <= 5; ++x)
    img.setGray(x, 8, 200);
  for(int x = 6; x <= 11; ++x)
    img.setGray(x, 8, 40);

  SECTION("Binary")
  {
    cv::FloodFill obj;
    obj.inputs.mode.value = cv::FloodMode::Binary;
    obj.inputs.threshold.value = 0.f; // exactly cv.jit's `!= 0`
    obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
    obj.inputs.seed_px.value = {3, 8};
    feed(obj.inputs.image, img);
    obj();

    CHECK(obj.outputs.filled.value == 10);
    CHECK(countMask(obj) == 10);
    // The grey-40 tail really is in the mask.
    CHECK(obj.outputs.image.texture.bytes[8 * 16 + 11] == 255);
    CHECK(obj.outputs.image.texture.bytes[8 * 16 + 12] == 0);
  }

  SECTION("Tolerance on the same image")
  {
    cv::FloodFill obj;
    obj.inputs.mode.value = cv::FloodMode::Tolerance;
    obj.inputs.tolerance.value = 0.1f;
    obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
    obj.inputs.seed_px.value = {3, 8};
    feed(obj.inputs.image, img);
    obj();

    CHECK(obj.outputs.filled.value == 4);
    CHECK(obj.outputs.image.texture.bytes[8 * 16 + 6] == 0); // grey 40 excluded
  }
}

TEST_CASE("FloodFill Binary honours a non-zero Threshold", "[floodfill][binary]")
{
  // Same bar, but Threshold 0.5 -> foreground is luma > 127.5. Now the grey-40 run is
  // background and the fill stops at x == 5.
  Image img(16, 16, 0);
  for(int x = 2; x <= 5; ++x)
    img.setGray(x, 8, 200);
  for(int x = 6; x <= 11; ++x)
    img.setGray(x, 8, 40);

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {3, 8};
  feed(obj.inputs.image, img);
  obj();

  CHECK(obj.outputs.filled.value == 4);
}

TEST_CASE("FloodFill Binary does not cross a zero gap", "[floodfill][binary]")
{
  // Row 8: x=2..5 grey 200, x=6 grey 0 (the gap), x=7..11 grey 200.
  // Seeded at x=3 the fill takes the left run only: 4 pixels.
  Image img(16, 16, 0);
  for(int x = 2; x <= 5; ++x)
    img.setGray(x, 8, 200);
  img.setGray(6, 8, 0);
  for(int x = 7; x <= 11; ++x)
    img.setGray(x, 8, 200);

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {3, 8};
  feed(obj.inputs.image, img);
  obj();

  CHECK(obj.outputs.filled.value == 4);
  CHECK(obj.outputs.image.texture.bytes[8 * 16 + 7] == 0); // right run untouched
}

TEST_CASE("FloodFill is 4-connected, not 8-connected", "[floodfill][binary]")
{
  // The clearest discriminator: (2,2) and (3,3) touch only at a corner. cv.jit's fill --
  // and ours -- is 4-connected, so seeding (2,2) fills exactly 1 pixel. An 8-connected
  // implementation would report 2.
  Image img(8, 8, 0);
  img.setGray(2, 2, 255);
  img.setGray(3, 3, 255);

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {2, 2};
  feed(obj.inputs.image, img);
  obj();

  CHECK(obj.outputs.filled.value == 1);
  CHECK(obj.outputs.image.texture.bytes[3 * 8 + 3] == 0);

  // ... and the same must hold with a 4-neighbour bridge added: (2,3) links the two, so
  // now all three pixels are one region. This rules out "the fill is just broken".
  Image linked(8, 8, 0);
  linked.setGray(2, 2, 255);
  linked.setGray(2, 3, 255);
  linked.setGray(3, 3, 255);
  feed(obj.inputs.image, linked);
  obj();
  CHECK(obj.outputs.filled.value == 3);
}

TEST_CASE("FloodFill Binary with a background seed fills nothing", "[floodfill][binary]")
{
  // cv.jit: `if(inData[seedx] == 0) return 0;` -- not an error, just an empty result.
  Image img(16, 16, 0);
  img.fillRect(4, 4, 5, 5, 255);

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {0, 0}; // black
  feed(obj.inputs.image, img);
  obj();

  CHECK(obj.outputs.filled.value == 0);
  REQUIRE(obj.outputs.image.texture.bytes != nullptr);
  CHECK(obj.outputs.image.texture.width == 16);
  CHECK(countMask(obj) == 0); // output was cleared and emitted, not left stale

  // Tolerance mode on the very same input does the opposite: it grows the background,
  // 256 - 25 = 231 pixels. This is the behavioural fork the Mode enum exists for.
  cv::FloodFill tol;
  tol.inputs.mode.value = cv::FloodMode::Tolerance;
  tol.inputs.tolerance.value = 0.1f;
  tol.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  tol.inputs.seed_px.value = {0, 0};
  feed(tol.inputs.image, img);
  tol();
  CHECK(tol.outputs.filled.value == 231);
}

TEST_CASE("FloodFill with an out-of-bounds pixel seed fills nothing", "[floodfill][binary]")
{
  Image img(16, 16, 0);
  img.fillRect(4, 4, 5, 5, 255);

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;

  // cv.jit: `if((seedx>=width)||(seedx<0)||(seedy>=height)||(seedy<0)) return 0;`
  const std::pair<int, int> outside[]
      = {{100, 100}, {-1, 3}, {3, -1}, {16, 3}, {3, 16}, {-5, -5}};
  for(auto [sx, sy] : outside)
  {
    obj.inputs.seed_px.value = {sx, sy};
    feed(obj.inputs.image, img);
    obj();
    INFO("seed " << sx << "," << sy);
    CHECK(obj.outputs.filled.value == 0);
    CHECK(countMask(obj) == 0);
  }

  // In bounds again -> the fill works, so the guard is not simply wedged.
  obj.inputs.seed_px.value = {6, 6};
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.filled.value == 25);
}

TEST_CASE("FloodFill pixel and normalised seeds select the same pixel", "[floodfill][seed]")
{
  // 16x16 black with an isolated single-pixel dot at (6,6) and a 5x5 block at (9,9).
  // Under Binary mode `Filled` is a fingerprint of the chosen pixel:
  //   (6,6)                -> 1
  //   any background pixel -> 0
  //   inside the block     -> 25
  // 6/16 = 0.375 and int(0.375*16) = 6, so the normalised seed 0.375,0.375 must land on
  // the dot; 7/16 = 0.4375 -> pixel 7, which is background.
  Image img(16, 16, 0);
  img.setGray(6, 6, 255);
  img.fillRect(9, 9, 5, 5, 255);

  cv::FloodFill norm;
  norm.inputs.mode.value = cv::FloodMode::Binary;
  norm.inputs.threshold.value = 0.f;
  norm.inputs.seed_mode.value = cv::FloodSeedMode::Normalized;
  norm.inputs.seed.value = {6.f / 16.f, 6.f / 16.f};
  feed(norm.inputs.image, img);
  norm();
  CHECK(norm.outputs.filled.value == 1);
  CHECK(norm.outputs.image.texture.bytes[6 * 16 + 6] == 255);

  cv::FloodFill px;
  px.inputs.mode.value = cv::FloodMode::Binary;
  px.inputs.threshold.value = 0.f;
  px.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  px.inputs.seed_px.value = {6, 6};
  feed(px.inputs.image, img);
  px();
  CHECK(px.outputs.filled.value == 1);
  CHECK(px.outputs.image.texture.bytes[6 * 16 + 6] == 255);

  // One column over the mapping must give a *different* answer, which is what makes the
  // agreement above meaningful.
  norm.inputs.seed.value = {7.f / 16.f, 6.f / 16.f};
  feed(norm.inputs.image, img);
  norm();
  CHECK(norm.outputs.filled.value == 0);

  px.inputs.seed_px.value = {10, 10};
  feed(px.inputs.image, img);
  px();
  CHECK(px.outputs.filled.value == 25);
}

TEST_CASE("FloodFill normalised seed stays clamped at the edges", "[floodfill][seed]")
{
  // Deliberate deviation: the normalised pad is clamped (x == 1.0 selects the last column)
  // instead of being treated as out of bounds. Only the pixel seed can be out of bounds.
  Image img(16, 16, 0);
  img.setGray(15, 15, 255);

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Normalized;
  obj.inputs.seed.value = {1.f, 1.f}; // int(1.0*16) == 16 -> clamped to 15
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.filled.value == 1);
}

TEST_CASE("FloodFill Binary on a 1x1 image", "[floodfill][edge]")
{
  Image white(1, 1, 255);
  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {0, 0};
  feed(obj.inputs.image, white);
  obj();
  CHECK(obj.outputs.filled.value == 1);

  Image black(1, 1, 0);
  feed(obj.inputs.image, black);
  obj();
  CHECK(obj.outputs.filled.value == 0);
}

TEST_CASE("FloodFill handles a dimension change mid-stream", "[floodfill][edge]")
{
  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {5, 5};

  Image big(32, 32, 255);
  feed(obj.inputs.image, big);
  obj();
  CHECK(obj.outputs.filled.value == 32 * 32);

  Image small(8, 8, 255);
  feed(obj.inputs.image, small);
  obj();
  CHECK(obj.outputs.filled.value == 8 * 8);
  CHECK(obj.outputs.image.texture.width == 8);
  CHECK(obj.outputs.image.texture.height == 8);

  // And a seed that was valid on the big frame but is not on the small one.
  obj.inputs.seed_px.value = {20, 20};
  feed(obj.inputs.image, small);
  obj();
  CHECK(obj.outputs.filled.value == 0);
}

TEST_CASE("FloodFill fills a large solid frame without truncating", "[floodfill][binary]")
{
  // cv.jit caps its segment stack at width*height>>2 and silently stops. This port has no
  // cap: a comb of 64 alternating full-height columns bridged by row 0 needs far more
  // segments than cv.jit would allow, and must still come out complete.
  const int W = 128, H = 128;
  Image img(W, H, 0);
  for(int x = 0; x < W; x += 2)
    for(int y = 0; y < H; ++y)
      img.setGray(x, y, 255);
  for(int x = 0; x < W; ++x)
    img.setGray(x, 0, 255); // bridge row

  // 64 columns x 128 rows = 8192, minus the 64 bridge pixels already counted in those
  // columns... easier: 64 columns fully set (8192 px) + the 64 odd-x pixels of row 0.
  const int expected = 64 * H + 64;

  cv::FloodFill obj;
  obj.inputs.mode.value = cv::FloodMode::Binary;
  obj.inputs.threshold.value = 0.f;
  obj.inputs.seed_mode.value = cv::FloodSeedMode::Pixels;
  obj.inputs.seed_px.value = {0, 127};
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.filled.value == expected);
  CHECK(countMask(obj) == expected);
}

// ================================================================================ BlobSort

TEST_CASE("BlobSort chains from a BlobStats blob list", "[blobsort][chain]")
{
  // Drive BlobSort purely from BlobStats' output list -- no texture on BlobSort at all,
  // which is exactly how cv.jit.blobs.sort is used (it sits after blobs.centroids).
  cv::BlobStats stats;
  stats.inputs.threshold.value = 0.5f;
  stats.inputs.min_size.value = 4;

  cv::BlobSort sort;

  clearTexture(sort.inputs.image);
  sort.inputs.max_distance.value = 0.3f;

  Image f0(48, 48, 0);
  f0.fillRect(10, 10, 6, 6, 255);
  feed(stats.inputs.image, f0);
  stats();
  REQUIRE(stats.outputs.blobs.value.size() == 1);

  sort.inputs.blobs_in.value = stats.outputs.blobs.value;
  sort();
  REQUIRE(sort.outputs.count.value == 1);
  const int id0 = sort.outputs.blobs.value.front().id;
  CHECK(id0 == 1);
  CHECK(sort.outputs.blobs.value.front().age == 0);
  // The centroid survives the hop: 6x6 block at (10,10) -> pixel centre 12.5 -> 12.5/48.
  CHECK(sort.outputs.blobs.value.front().centroid.x == Approx(12.5f / 48.f).margin(1e-5));

  // Frame 1: the same blob, moved 3 px right and 2 px down.
  Image f1(48, 48, 0);
  f1.fillRect(13, 12, 6, 6, 255);
  feed(stats.inputs.image, f1);
  stats();
  sort.inputs.blobs_in.value = stats.outputs.blobs.value;
  sort();
  REQUIRE(sort.outputs.count.value == 1);
  CHECK(sort.outputs.blobs.value.front().id == id0);
  CHECK(sort.outputs.blobs.value.front().age == 1);

  // Frame 2: still the same blob -> the id is stable across three frames.
  Image f2(48, 48, 0);
  f2.fillRect(16, 13, 6, 6, 255);
  feed(stats.inputs.image, f2);
  stats();
  sort.inputs.blobs_in.value = stats.outputs.blobs.value;
  sort();
  REQUIRE(sort.outputs.count.value == 1);
  CHECK(sort.outputs.blobs.value.front().id == id0);
  CHECK(sort.outputs.blobs.value.front().age == 2);
}

TEST_CASE("BlobSort list input wins over the texture", "[blobsort][chain]")
{
  // A texture is connected *and* a list is present: the list must be authoritative.
  // The texture holds three blobs; the list holds one.
  cv::BlobSort sort;
  clearTexture(sort.inputs.image);
  sort.inputs.max_distance.value = 0.05f;

  Image img(48, 48, 0);
  img.fillRect(2, 2, 5, 5, 255);
  img.fillRect(20, 20, 5, 5, 255);
  img.fillRect(40, 40, 5, 5, 255);
  feed(sort.inputs.image, img);
  sort.inputs.blobs_in.value = {makeBlob(0.25f, 0.25f)};
  sort();
  CHECK(sort.outputs.count.value == 1);

  // Empty the list again and the legacy image path takes over -> 3 blobs.
  sort.inputs.blobs_in.value.clear();
  feed(sort.inputs.image, img);
  sort();
  CHECK(sort.outputs.count.value == 3);
}

TEST_CASE("BlobSort frees a vanished id and reuses the lowest one", "[blobsort][chain]")
{
  cv::BlobSort sort;
  clearTexture(sort.inputs.image);
  sort.inputs.max_distance.value = 0.05f; // 0.05 normalised; all moves below are 0

  // Frame 0: three blobs, in list order -> ids 1, 2, 3.
  sort.inputs.blobs_in.value
      = {makeBlob(0.10f, 0.10f), makeBlob(0.50f, 0.50f), makeBlob(0.90f, 0.90f)};
  sort();
  REQUIRE(sort.outputs.count.value == 3);
  CHECK(sort.outputs.blobs.value[0].id == 1);
  CHECK(sort.outputs.blobs.value[1].id == 2);
  CHECK(sort.outputs.blobs.value[2].id == 3);

  // Frame 1: the first two vanish -> ids 1 and 2 are freed, 3 survives.
  sort.inputs.blobs_in.value = {makeBlob(0.90f, 0.90f)};
  sort();
  REQUIRE(sort.outputs.count.value == 1);
  CHECK(sort.outputs.blobs.value[0].id == 3);
  CHECK(sort.outputs.blobs.value[0].age == 1);

  // Frame 2: one newcomer -> it must take the *lowest* free id, 1 (not 2, not 4).
  sort.inputs.blobs_in.value = {makeBlob(0.90f, 0.90f), makeBlob(0.30f, 0.70f)};
  sort();
  REQUIRE(sort.outputs.count.value == 2);
  CHECK(sort.outputs.blobs.value[0].id == 3);
  CHECK(sort.outputs.blobs.value[1].id == 1);
  CHECK(sort.outputs.blobs.value[1].age == 0);

  // Frame 3: another newcomer -> next lowest free id, 2.
  sort.inputs.blobs_in.value
      = {makeBlob(0.90f, 0.90f), makeBlob(0.30f, 0.70f), makeBlob(0.70f, 0.30f)};
  sort();
  REQUIRE(sort.outputs.count.value == 3);
  CHECK(sort.outputs.blobs.value[2].id == 2);

  // Frame 4: free list exhausted -> a fresh id, 4.
  sort.inputs.blobs_in.value
      = {makeBlob(0.90f, 0.90f), makeBlob(0.30f, 0.70f), makeBlob(0.70f, 0.30f),
         makeBlob(0.10f, 0.90f)};
  sort();
  REQUIRE(sort.outputs.count.value == 4);
  CHECK(sort.outputs.blobs.value[3].id == 4);
}

TEST_CASE("BlobSort gives one id per blob where cv.jit would double-latch",
          "[blobsort][divergence]")
{
  // cv.jit's latch loop never marks a previous blob as taken, so two current blobs both
  // closest to the same previous blob both get that blob's id. Here: one previous blob at
  // (0.50,0.50); next frame two blobs at (0.49,0.50) and (0.51,0.50), both within
  // threshold of it. cv.jit would emit the same id twice. We must emit two distinct ids.
  cv::BlobSort sort;
  clearTexture(sort.inputs.image);
  sort.inputs.max_distance.value = 0.1f;

  sort.inputs.blobs_in.value = {makeBlob(0.50f, 0.50f)};
  sort();
  REQUIRE(sort.outputs.count.value == 1);
  CHECK(sort.outputs.blobs.value[0].id == 1);

  sort.inputs.blobs_in.value = {makeBlob(0.49f, 0.50f), makeBlob(0.51f, 0.50f)};
  sort();
  REQUIRE(sort.outputs.count.value == 2);
  CHECK(sort.outputs.blobs.value[0].id != sort.outputs.blobs.value[1].id);
  // The first one latches (age 1), the second is new (age 0, fresh id 2).
  CHECK(sort.outputs.blobs.value[0].id == 1);
  CHECK(sort.outputs.blobs.value[0].age == 1);
  CHECK(sort.outputs.blobs.value[1].id == 2);
  CHECK(sort.outputs.blobs.value[1].age == 0);
}

TEST_CASE("BlobSort pixel and normalised distance thresholds agree", "[blobsort][units]")
{
  // 48x48 frames, 6x6 block moving from (10,10) to (16,10): centroid (12.5,12.5) ->
  // (18.5,12.5), i.e. 6 px == 6/48 == 0.125 normalised. On a square frame the two unit
  // systems are exactly related by the factor 48.
  //   accept: 0.2 normalised == 9.6 px   (0.125 < 0.2, 6 < 9.6)
  //   reject: 0.1 normalised == 4.8 px   (0.125 > 0.1, 6 > 4.8)
  // A rejected latch is visible through `age` (0 = brand new) rather than through the id,
  // because the freed id 1 is immediately recycled by the newcomer.
  auto run = [](bool pixels, float thr) {
    cv::BlobSort sort;
    clearTexture(sort.inputs.image);
    sort.inputs.units.value
        = pixels ? cv::BlobDistanceUnits::Pixels : cv::BlobDistanceUnits::Normalized;
    if(pixels)
      sort.inputs.max_distance_px.value = thr;
    else
      sort.inputs.max_distance.value = thr;
    sort.inputs.threshold.value = 0.5f;
    sort.inputs.min_size.value = 4;

    Image f0(48, 48, 0);
    f0.fillRect(10, 10, 6, 6, 255);
    feed(sort.inputs.image, f0);
    sort();
    REQUIRE(sort.outputs.count.value == 1);
    REQUIRE(sort.outputs.blobs.value[0].centroid.x == Approx(12.5f / 48.f).margin(1e-5));

    Image f1(48, 48, 0);
    f1.fillRect(16, 10, 6, 6, 255);
    feed(sort.inputs.image, f1);
    sort();
    REQUIRE(sort.outputs.count.value == 1);
    REQUIRE(sort.outputs.blobs.value[0].centroid.x == Approx(18.5f / 48.f).margin(1e-5));
    return sort.outputs.blobs.value[0].age; // 1 == latched, 0 == treated as new
  };

  CHECK(run(false, 0.2f) == 1);
  CHECK(run(true, 9.6f) == 1);
  CHECK(run(false, 0.1f) == 0);
  CHECK(run(true, 4.8f) == 0);

  // The pixel slider really is in pixels: the default 10 (cv.jit's default) accepts a 6 px
  // move, and 0.1 -- which would accept it if it were being read as normalised -- rejects.
  CHECK(run(true, 10.f) == 1);
  CHECK(run(true, 0.1f) == 0);
}

TEST_CASE("BlobSort pixel units are anisotropic on a non-square frame", "[blobsort][units]")
{
  // 64x16 frame, 4x4 block moving from (10,4) to (18,4): centroid (11.5,5.5)->(19.5,5.5),
  // 8 px in x. In *normalised* units that is 8/64 = 0.125; in pixels it is 8.
  // Threshold 9 px accepts; the numerically equal normalised threshold 9 would accept
  // anything, so instead compare against the normalised value that matches on the *y*
  // axis to show the two axes are scaled independently: a normalised threshold of 0.13
  // accepts the x move, but the same 0.13 would also accept a y move of 0.13*16 = 2.08 px
  // only -- a 3 px y move (0.1875 normalised) is rejected, while 9 px in pixel units
  // accepts it. Pixel units therefore cannot be a single global rescale.
  auto move = [](bool pixels, float thr, int dx, int dy) {
    cv::BlobSort sort;
    clearTexture(sort.inputs.image);
    sort.inputs.units.value
        = pixels ? cv::BlobDistanceUnits::Pixels : cv::BlobDistanceUnits::Normalized;
    if(pixels)
      sort.inputs.max_distance_px.value = thr;
    else
      sort.inputs.max_distance.value = thr;
    sort.inputs.min_size.value = 4;

    Image f0(64, 16, 0);
    f0.fillRect(10, 4, 4, 4, 255);
    feed(sort.inputs.image, f0);
    sort();
    REQUIRE(sort.outputs.count.value == 1);

    Image f1(64, 16, 0);
    f1.fillRect(10 + dx, 4 + dy, 4, 4, 255);
    feed(sort.inputs.image, f1);
    sort();
    REQUIRE(sort.outputs.count.value == 1);
    return sort.outputs.blobs.value[0].age;
  };

  CHECK(move(true, 9.f, 8, 0) == 1);  // 8 px  < 9 px
  CHECK(move(true, 9.f, 0, 3) == 1);  // 3 px  < 9 px
  CHECK(move(false, 0.13f, 8, 0) == 1); // 0.125 < 0.13
  CHECK(move(false, 0.13f, 0, 3) == 0); // 0.1875 > 0.13  <- the anisotropy
  CHECK(move(true, 7.f, 8, 0) == 0);  // 8 px  > 7 px
}

TEST_CASE("BlobSort reproduces the (0,0) empty-image special case", "[blobsort][cvjit]")
{
  // cv.jit.blobs.sort: a 1-cell input whose centroid is exactly (0,0) means "the source
  // image had no blobs". It writes a single id 0 and returns *before* the lost-blob
  // cleanup, so the tracker's state survives untouched.
  cv::BlobSort sort;
  clearTexture(sort.inputs.image);
  sort.inputs.max_distance.value = 0.05f;

  sort.inputs.blobs_in.value = {makeBlob(0.20f, 0.20f), makeBlob(0.80f, 0.80f)};
  sort();
  REQUIRE(sort.outputs.count.value == 2);
  CHECK(sort.outputs.blobs.value[0].id == 1);
  CHECK(sort.outputs.blobs.value[1].id == 2);

  // The empty frame.
  sort.inputs.blobs_in.value = {makeBlob(0.f, 0.f, 0.f)};
  sort();
  REQUIRE(sort.outputs.count.value == 1);
  CHECK(sort.outputs.blobs.value[0].id == 0);
  CHECK(sort.outputs.blobs.value[0].centroid.x == 0.f);
  CHECK(sort.outputs.blobs.value[0].centroid.y == 0.f);

  // Both blobs come back: because the state was untouched they keep ids 1 and 2, and
  // their ages advance from 0 to 1 -- i.e. the empty frame was not counted as a frame in
  // which they were lost. Had the cleanup run, both ids would have been freed and the
  // ages reset to 0.
  sort.inputs.blobs_in.value = {makeBlob(0.20f, 0.20f), makeBlob(0.80f, 0.80f)};
  sort();
  REQUIRE(sort.outputs.count.value == 2);
  CHECK(sort.outputs.blobs.value[0].id == 1);
  CHECK(sort.outputs.blobs.value[1].id == 2);
  CHECK(sort.outputs.blobs.value[0].age == 1);
  CHECK(sort.outputs.blobs.value[1].age == 1);
}

TEST_CASE("BlobSort special case needs both the count and the exact origin",
          "[blobsort][cvjit]")
{
  // Guards against loosening `(cx==0 && cy==0 && w==1)`.
  {
    // Not at the origin -> a normal, state-consuming frame.
    cv::BlobSort sort;
    clearTexture(sort.inputs.image);
    sort.inputs.max_distance.value = 0.05f;
    sort.inputs.blobs_in.value = {makeBlob(0.20f, 0.20f), makeBlob(0.80f, 0.80f)};
    sort();
    sort.inputs.blobs_in.value = {makeBlob(0.001f, 0.f)};
    sort();
    REQUIRE(sort.outputs.count.value == 1);
    CHECK(sort.outputs.blobs.value[0].id != 0); // real blob, real id
    // Both previous ids were freed; the newcomer took the lowest, 1.
    CHECK(sort.outputs.blobs.value[0].id == 1);
    CHECK(sort.outputs.blobs.value[0].age == 0);
  }
  {
    // At the origin but not alone -> also a normal frame.
    cv::BlobSort sort;
    clearTexture(sort.inputs.image);
    sort.inputs.max_distance.value = 0.05f;
    sort.inputs.blobs_in.value = {makeBlob(0.f, 0.f), makeBlob(0.80f, 0.80f)};
    sort();
    REQUIRE(sort.outputs.count.value == 2);
    CHECK(sort.outputs.blobs.value[0].id == 1);
    CHECK(sort.outputs.blobs.value[1].id == 2);
  }
}

TEST_CASE("BlobSort with no list and no frame does nothing", "[blobsort][edge]")
{
  cv::BlobSort sort;
  clearTexture(sort.inputs.image);
  sort();
  CHECK(sort.outputs.count.value == 0);
  CHECK(sort.outputs.blobs.value.empty());
}
