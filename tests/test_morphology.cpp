// Tests for cv::Morphology (CV/Cpu/Morphology.hpp) -- the CPU port of cv.jit.dilate /
// cv.jit.erode, plus the open / close pairs (cv.jit.open / cv.jit.close abstractions).
//
// WHY THE NUMBERS BELOW ARE WHAT THEY ARE
//
// BINARY vs GREYSCALE. This is the capability the existing shader does not have, so it gets
// the first and sharpest test. cv.jit's `greyscale 0` mode (its DEFAULT) thresholds the
// neighbourhood to a mask, ORs / ANDs the mask, and writes 255 or 0. `greyscale 1` takes a
// plain max / min. Over a two-pixel neighbourhood {3, 200}:
//     binary    dilate = OR(3 != 0, 200 != 0)  = true  -> 255
//     greyscale dilate = max(3, 200)                   -> 200
//     binary    erode  = AND(3 != 0, 200 != 0) = true  -> 255
//     greyscale erode  = min(3, 200)                   -> 3
// Note the binary erode: 255, NOT 3 and NOT 0. A "textbook" implementation that binarised at
// the usual mid-grey 128 would call 3 background and emit 0 here; cv.jit tests `!= 0`, which
// is why Threshold defaults to 0. That is the assertion that stops someone "fixing" it.
//
// GEOMETRY. With a square structuring element of radius r, dilation of a solid k x k square
// gives a solid (k+2r) x (k+2r) square and erosion gives (k-2r) x (k-2r) (empty if k <= 2r),
// as long as the square is at least r away from the frame -- see BORDER below. So a 5x5 blob
// with r = 1 gives 7x7 = 49 dilated and 3x3 = 9 eroded; with r = 2, 9x9 = 81 and 1x1 = 1.
//
// SQUARE vs CROSS. The one tap that separates them is a pure diagonal. Dilating a single lit
// pixel gives 9 pixels with the square element and 5 with the cross (centre + N/S/E/W).
// Eroding a full white field whose only hole is at a diagonal neighbour kills the centre with
// the square element and leaves it alive with the cross.
//
// BORDER. Off-image taps are not tested (== clamp-to-edge), exactly as cv.jit's hand-written
// first-line / last-line cases behave, so a blob flush against the frame is not eaten from
// the outside: a 4-wide slab spanning the full height of an 8x8 image erodes to 3 columns
// (only the interior edge at x = 3 is lost), i.e. 3 * 8 = 24 pixels, not 2 * 6.
//
// OPEN / CLOSE. Open (erode -> dilate) deletes anything thinner than the element and restores
// everything else; Close (dilate -> erode) fills holes smaller than the element and restores
// larger ones. With a 3x3 element a 1x1 hole is filled and a 3x3 hole survives untouched
// (dilation shrinks it to 1x1, erosion grows it back to 3x3).

#include <catch2/catch_test_macros.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/Morphology.hpp>

#include <vector>

using namespace cvtest;
using cv::MorphOperation;
using cv::MorphShape;

namespace
{
// Run the object with an explicit set of attributes (all defaults spelled out so each test
// reads as a full specification of the case).
void run(
    cv::Morphology& obj, Image& img, MorphOperation op,
    MorphShape shape = MorphShape::Square, bool binary = true, int radius = 1,
    float threshold = 0.f)
{
  feed(obj.inputs.image, img);
  obj.inputs.operation.value = op;
  obj.inputs.shape.value = shape;
  obj.inputs.binary.value = binary;
  obj.inputs.radius.value = radius;
  obj.inputs.threshold.value = threshold;
  obj();
}

std::uint8_t chan(const cv::Morphology& obj, int x, int y, int c)
{
  auto& o = obj.outputs.image.texture;
  return o.bytes[(static_cast<std::size_t>(y) * o.width + x) * 4 + c];
}
std::uint8_t red(const cv::Morphology& obj, int x, int y)
{
  return chan(obj, x, y, 0);
}

// Number of lit pixels (red channel non-zero) in the output.
int litCount(const cv::Morphology& obj)
{
  auto& o = obj.outputs.image.texture;
  int n = 0;
  for(int i = 0; i < o.width * o.height; ++i)
    n += (o.bytes[static_cast<std::size_t>(i) * 4] != 0) ? 1 : 0;
  return n;
}

// Bounding box of the lit pixels; returns false if nothing is lit.
bool litBounds(const cv::Morphology& obj, int& x0, int& y0, int& x1, int& y1)
{
  auto& o = obj.outputs.image.texture;
  x0 = o.width;
  y0 = o.height;
  x1 = -1;
  y1 = -1;
  for(int y = 0; y < o.height; ++y)
    for(int x = 0; x < o.width; ++x)
      if(o.bytes[(static_cast<std::size_t>(y) * o.width + x) * 4] != 0)
      {
        x0 = std::min(x0, x);
        y0 = std::min(y0, y);
        x1 = std::max(x1, x);
        y1 = std::max(y1, y);
      }
  return x1 >= 0;
}

// Copy an output texture back into an Image, so a second object can be chained onto it.
Image toImage(const cv::Morphology& obj)
{
  auto& o = obj.outputs.image.texture;
  Image img(o.width, o.height, 0);
  for(int i = 0; i < o.width * o.height; ++i)
  {
    const std::uint8_t* p = o.bytes + static_cast<std::size_t>(i) * 4;
    img.px[static_cast<std::size_t>(i) * 4 + 0] = p[0];
    img.px[static_cast<std::size_t>(i) * 4 + 1] = p[1];
    img.px[static_cast<std::size_t>(i) * 4 + 2] = p[2];
    img.px[static_cast<std::size_t>(i) * 4 + 3] = p[3];
  }
  return img;
}
}

// ======================================================= the point of the object: binary mode

TEST_CASE(
    "Binary and greyscale disagree on the {3, 200} neighbourhood", "[morphology][binary]")
{
  // A 2x1 image, so every pixel sees exactly the neighbourhood {3, 200}, whatever the shape.
  Image img(2, 1, 0);
  img.setGray(0, 0, 3);
  img.setGray(1, 0, 200);

  SECTION("dilate: binary emits 255, greyscale emits 200")
  {
    cv::Morphology bin, grey;
    run(bin, img, MorphOperation::Dilate, MorphShape::Square, /*binary*/ true);
    run(grey, img, MorphOperation::Dilate, MorphShape::Square, /*binary*/ false);

    CHECK(int(red(bin, 0, 0)) == 255);
    CHECK(int(red(bin, 1, 0)) == 255);
    CHECK(int(red(grey, 0, 0)) == 200);
    CHECK(int(red(grey, 1, 0)) == 200);
    // The whole reason this object exists.
    CHECK(int(red(bin, 0, 0)) != int(red(grey, 0, 0)));
  }

  SECTION("erode: binary emits 255, greyscale emits 3")
  {
    // A mid-grey binarisation would call 3 "background" and emit 0 here. cv.jit tests
    // `!= 0`, so both pixels are foreground and the AND is true -> 255.
    cv::Morphology bin, grey;
    run(bin, img, MorphOperation::Erode, MorphShape::Square, /*binary*/ true);
    run(grey, img, MorphOperation::Erode, MorphShape::Square, /*binary*/ false);

    CHECK(int(red(bin, 0, 0)) == 255);
    CHECK(int(red(bin, 1, 0)) == 255);
    CHECK(int(red(grey, 0, 0)) == 3);
    CHECK(int(red(grey, 1, 0)) == 3);
  }

  SECTION("the defaults are the cv.jit-exact ones (binary, square, radius 1, threshold 0)")
  {
    cv::Morphology defaulted;
    CHECK(defaulted.inputs.binary.value == true);
    CHECK(defaulted.inputs.shape.value == MorphShape::Square);
    CHECK(defaulted.inputs.radius.value == 1);
    CHECK(defaulted.inputs.threshold.value == 0.f);

    // ... and running with nothing but the operation set reproduces the binary result.
    feed(defaulted.inputs.image, img);
    defaulted.inputs.operation.value = MorphOperation::Dilate;
    defaulted();
    CHECK(int(red(defaulted, 0, 0)) == 255);
  }
}

TEST_CASE("Threshold moves the binarisation point", "[morphology][binary]")
{
  // Same {3, 200} pair, but now with a mid-grey threshold: 3 falls below it and becomes
  // background, so the AND is false and erode emits 0 -- while dilate still emits 255.
  Image img(2, 1, 0);
  img.setGray(0, 0, 3);
  img.setGray(1, 0, 200);

  cv::Morphology er, di;
  run(er, img, MorphOperation::Erode, MorphShape::Square, true, 1, /*threshold*/ 0.5f);
  run(di, img, MorphOperation::Dilate, MorphShape::Square, true, 1, /*threshold*/ 0.5f);

  CHECK(int(red(er, 0, 0)) == 0);
  CHECK(int(red(er, 1, 0)) == 0);
  CHECK(int(red(di, 0, 0)) == 255);
  CHECK(int(red(di, 1, 0)) == 255);
}

TEST_CASE("Binary output is renormalised to 0 / 255 on a 2x2 checker", "[morphology][binary]")
{
  // Every pixel of a 2x2 sees all four pixels, so the result is uniform. Doubles as the
  // 2x2 "does not crash" case.
  Image img(2, 2, 0);
  img.setGray(0, 0, 3);
  img.setGray(1, 0, 200);
  img.setGray(0, 1, 200);
  img.setGray(1, 1, 3);

  cv::Morphology bin, grey;
  run(bin, img, MorphOperation::Erode, MorphShape::Square, true);
  run(grey, img, MorphOperation::Erode, MorphShape::Square, false);
  for(int y = 0; y < 2; ++y)
    for(int x = 0; x < 2; ++x)
    {
      CHECK(int(red(bin, x, y)) == 255);
      CHECK(int(red(grey, x, y)) == 3);
    }

  cv::Morphology bin2, grey2;
  run(bin2, img, MorphOperation::Dilate, MorphShape::Square, true);
  run(grey2, img, MorphOperation::Dilate, MorphShape::Square, false);
  for(int y = 0; y < 2; ++y)
    for(int x = 0; x < 2; ++x)
    {
      CHECK(int(red(bin2, x, y)) == 255);
      CHECK(int(red(grey2, x, y)) == 200);
    }
}

// ============================================================================== geometry

TEST_CASE("Dilate grows and erode shrinks a square by exactly one pixel", "[morphology]")
{
  // 5x5 blob at (3,3)..(7,7) in an 11x11 frame -- 3 px of margin, so the frame never
  // interferes.
  Image img(11, 11, 0);
  img.fillRect(3, 3, 5, 5, 255);

  int x0{}, y0{}, x1{}, y1{};

  SECTION("dilate r=1 -> 7x7 = 49")
  {
    cv::Morphology obj;
    run(obj, img, MorphOperation::Dilate);
    CHECK(litCount(obj) == 49);
    REQUIRE(litBounds(obj, x0, y0, x1, y1));
    CHECK(x0 == 2);
    CHECK(y0 == 2);
    CHECK(x1 == 8);
    CHECK(y1 == 8);
  }

  SECTION("erode r=1 -> 3x3 = 9")
  {
    cv::Morphology obj;
    run(obj, img, MorphOperation::Erode);
    CHECK(litCount(obj) == 9);
    REQUIRE(litBounds(obj, x0, y0, x1, y1));
    CHECK(x0 == 4);
    CHECK(y0 == 4);
    CHECK(x1 == 6);
    CHECK(y1 == 6);
  }

  SECTION("radius scales the effect: r=2 -> 9x9 = 81 dilated, 1x1 = 1 eroded")
  {
    cv::Morphology di, er;
    run(di, img, MorphOperation::Dilate, MorphShape::Square, true, /*radius*/ 2);
    CHECK(litCount(di) == 81);
    REQUIRE(litBounds(di, x0, y0, x1, y1));
    CHECK(x0 == 1);
    CHECK(y0 == 1);
    CHECK(x1 == 9);
    CHECK(y1 == 9);

    run(er, img, MorphOperation::Erode, MorphShape::Square, true, /*radius*/ 2);
    CHECK(litCount(er) == 1);
    CHECK(int(red(er, 5, 5)) == 255);
  }

  SECTION("r=3 erodes the 5x5 away entirely")
  {
    cv::Morphology obj;
    run(obj, img, MorphOperation::Erode, MorphShape::Square, true, /*radius*/ 3);
    CHECK(litCount(obj) == 0);
  }

  SECTION("radius below 1 is clamped to 1")
  {
    cv::Morphology zero, one;
    run(zero, img, MorphOperation::Erode, MorphShape::Square, true, /*radius*/ 0);
    run(one, img, MorphOperation::Erode, MorphShape::Square, true, /*radius*/ 1);
    CHECK(litCount(zero) == litCount(one));
    CHECK(litCount(zero) == 9);
  }
}

TEST_CASE("A k x k square erodes to (k-2) x (k-2)", "[morphology]")
{
  for(int k : {3, 4, 5, 6, 7})
  {
    Image img(16, 16, 0);
    img.fillRect(4, 4, k, k, 255);

    cv::Morphology obj;
    run(obj, img, MorphOperation::Erode);
    const int expected = (k - 2) * (k - 2);
    INFO("k = " << k);
    CHECK(litCount(obj) == expected);

    int x0{}, y0{}, x1{}, y1{};
    REQUIRE(litBounds(obj, x0, y0, x1, y1));
    CHECK(x0 == 5);
    CHECK(y0 == 5);
    CHECK(x1 == 4 + k - 2);
    CHECK(y1 == 4 + k - 2);
  }
}

// ========================================================================= square vs cross

TEST_CASE("Square and cross differ on a diagonal neighbour", "[morphology][shape]")
{
  SECTION("dilating one pixel: 9 (square) vs 5 (cross)")
  {
    Image img(5, 5, 0);
    img.setGray(2, 2, 255);

    cv::Morphology sq, cr;
    run(sq, img, MorphOperation::Dilate, MorphShape::Square);
    run(cr, img, MorphOperation::Dilate, MorphShape::Cross);

    CHECK(litCount(sq) == 9);
    CHECK(litCount(cr) == 5);
    // The discriminating tap is the pure diagonal (1,1).
    CHECK(int(red(sq, 1, 1)) == 255);
    CHECK(int(red(cr, 1, 1)) == 0);
    // The axial taps are lit in both.
    CHECK(int(red(sq, 1, 2)) == 255);
    CHECK(int(red(cr, 1, 2)) == 255);
  }

  SECTION("eroding a field with a single diagonal hole")
  {
    // Everything white except (1,1). Pixel (2,2) only touches that hole diagonally.
    Image img(5, 5, 255);
    img.setGray(1, 1, 0);

    cv::Morphology sq, cr;
    run(sq, img, MorphOperation::Erode, MorphShape::Square);
    run(cr, img, MorphOperation::Erode, MorphShape::Cross);

    CHECK(int(red(sq, 2, 2)) == 0);   // square sees the diagonal hole
    CHECK(int(red(cr, 2, 2)) == 255); // cross does not
    // (2,1) is an axial neighbour of the hole: dead in both.
    CHECK(int(red(sq, 2, 1)) == 0);
    CHECK(int(red(cr, 2, 1)) == 0);
    // Counts: square kills the 3x3 around (1,1) = 9 pixels -> 25 - 9 = 16 survive.
    // Cross kills (1,1) + its 4 axial neighbours = 5 -> 25 - 5 = 20 survive.
    CHECK(litCount(sq) == 16);
    CHECK(litCount(cr) == 20);
  }

  SECTION("radius 2 cross is a plus sign, not a 5x5 block")
  {
    Image img(9, 9, 0);
    img.setGray(4, 4, 255);

    cv::Morphology cr;
    run(cr, img, MorphOperation::Dilate, MorphShape::Cross, true, /*radius*/ 2);
    // Centre + 4 arms of length 2 = 1 + 8 = 9 pixels.
    CHECK(litCount(cr) == 9);
    CHECK(int(red(cr, 4, 2)) == 255);
    CHECK(int(red(cr, 2, 4)) == 255);
    CHECK(int(red(cr, 3, 3)) == 0);
  }
}

// ============================================================================ open / close

TEST_CASE("Open removes an isolated pixel and preserves a blob", "[morphology][open]")
{
  Image img(20, 20, 0);
  img.setGray(2, 2, 255);         // speck
  img.fillRect(10, 10, 5, 5, 255); // 5x5 blob, 25 px

  cv::Morphology obj;
  run(obj, img, MorphOperation::Open);

  // Only the blob survives, and it survives EXACTLY.
  CHECK(litCount(obj) == 25);
  CHECK(int(red(obj, 2, 2)) == 0);
  for(int y = 10; y < 15; ++y)
    for(int x = 10; x < 15; ++x)
      CHECK(int(red(obj, x, y)) == 255);
}

TEST_CASE("Close fills a 1x1 hole and preserves a 3x3 hole", "[morphology][close]")
{
  // Two blobs, both >= 2 px from the frame and from each other after dilation.
  //   A: 7x7 at (2,2)..(8,8) with a single-pixel hole at (5,5)
  //   B: 9x9 at (12,2)..(20,10) with a 3x3 hole at (15,5)..(17,7)
  Image img(26, 14, 0);
  img.fillRect(2, 2, 7, 7, 255);
  img.setGray(5, 5, 0);
  img.fillRect(12, 2, 9, 9, 255);
  img.fillRect(15, 5, 3, 3, 0);

  cv::Morphology obj;
  run(obj, img, MorphOperation::Close);

  // A: the hole is gone, the outline is unchanged -> a solid 7x7 = 49.
  for(int y = 2; y <= 8; ++y)
    for(int x = 2; x <= 8; ++x)
      CHECK(int(red(obj, x, y)) == 255);
  CHECK(int(red(obj, 1, 5)) == 0); // no growth on the left
  CHECK(int(red(obj, 9, 5)) == 0); // ... nor on the right

  // B: the 3x3 hole survives, unchanged, and the outline is unchanged -> 81 - 9 = 72.
  for(int y = 5; y <= 7; ++y)
    for(int x = 15; x <= 17; ++x)
      CHECK(int(red(obj, x, y)) == 0);
  CHECK(int(red(obj, 12, 2)) == 255);
  CHECK(int(red(obj, 20, 10)) == 255);
  CHECK(int(red(obj, 11, 6)) == 0);
  CHECK(int(red(obj, 21, 6)) == 0);

  CHECK(litCount(obj) == 49 + 72);
}

TEST_CASE("Erode then dilate is not the identity: a spur is lost", "[morphology][open]")
{
  // A 7x7 body at (4,4)..(10,10) with a 1-pixel-thick spur of 3 pixels sticking out to the
  // right at y = 7. Input has 49 + 3 = 52 lit pixels.
  Image img(20, 20, 0);
  img.fillRect(4, 4, 7, 7, 255);
  img.setGray(11, 7, 255);
  img.setGray(12, 7, 255);
  img.setGray(13, 7, 255);

  cv::Morphology opened;
  run(opened, img, MorphOperation::Open);

  // The body comes back exactly; the spur does not. Open != identity.
  CHECK(litCount(opened) == 49);
  for(int y = 4; y <= 10; ++y)
    for(int x = 4; x <= 10; ++x)
      CHECK(int(red(opened, x, y)) == 255);
  CHECK(int(red(opened, 11, 7)) == 0);
  CHECK(int(red(opened, 12, 7)) == 0);
  CHECK(int(red(opened, 13, 7)) == 0);

  // And Open really is a chained erode -> dilate: two objects wired by hand agree with it
  // pixel for pixel.
  cv::Morphology e, d;
  run(e, img, MorphOperation::Erode);
  CHECK(litCount(e) == 25); // 5x5 body core, spur already gone
  Image mid = toImage(e);
  run(d, mid, MorphOperation::Dilate);
  REQUIRE(d.outputs.image.texture.width == opened.outputs.image.texture.width);
  for(int y = 0; y < 20; ++y)
    for(int x = 0; x < 20; ++x)
      REQUIRE(int(red(d, x, y)) == int(red(opened, x, y)));

  // Close on the same shape is extensive: it keeps every input pixel, spur included.
  cv::Morphology closed;
  run(closed, img, MorphOperation::Close);
  CHECK(int(red(closed, 11, 7)) == 255);
  CHECK(int(red(closed, 13, 7)) == 255);
}

// ================================================================================== border

TEST_CASE("A blob flush against the frame is not eroded from outside", "[morphology][border]")
{
  // Left slab: columns 0..3, all 8 rows. It touches the left, top and bottom edges.
  Image img(8, 8, 0);
  img.fillRect(0, 0, 4, 8, 255);

  cv::Morphology obj;
  run(obj, img, MorphOperation::Erode);

  // Only the interior edge (column 3, which touches background at column 4) is lost.
  CHECK(litCount(obj) == 3 * 8);
  for(int y = 0; y < 8; ++y)
  {
    INFO("row " << y);
    CHECK(int(red(obj, 0, y)) == 255);
    CHECK(int(red(obj, 2, y)) == 255);
    CHECK(int(red(obj, 3, y)) == 0);
  }
  // Corners in particular: (0,0) and (0,7) survive.
  CHECK(int(red(obj, 0, 0)) == 255);
  CHECK(int(red(obj, 0, 7)) == 255);

  SECTION("a full white frame erodes to itself")
  {
    Image white(6, 6, 255);
    cv::Morphology w;
    run(w, white, MorphOperation::Erode);
    CHECK(litCount(w) == 36);
  }
}

// =============================================================================== greyscale

TEST_CASE("Greyscale mode works per channel and passes alpha through", "[morphology][grey]")
{
  Image img(3, 1, 0);
  img.set(0, 0, 10, 200, 30, 255);
  img.set(1, 0, 250, 5, 90, 77);
  img.set(2, 0, 0, 0, 0, 12);

  cv::Morphology di, er;
  run(di, img, MorphOperation::Dilate, MorphShape::Square, /*binary*/ false);
  run(er, img, MorphOperation::Erode, MorphShape::Square, /*binary*/ false);

  // Pixel 1 sees all three pixels: per-channel max / min.
  CHECK(int(chan(di, 1, 0, 0)) == 250);
  CHECK(int(chan(di, 1, 0, 1)) == 200);
  CHECK(int(chan(di, 1, 0, 2)) == 90);
  CHECK(int(chan(er, 1, 0, 0)) == 0);
  CHECK(int(chan(er, 1, 0, 1)) == 0);
  CHECK(int(chan(er, 1, 0, 2)) == 0);

  // Pixel 0 sees {0, 1}: max = (250,200,90), min = (10,5,30).
  CHECK(int(chan(di, 0, 0, 0)) == 250);
  CHECK(int(chan(di, 0, 0, 1)) == 200);
  CHECK(int(chan(di, 0, 0, 2)) == 90);
  CHECK(int(chan(er, 0, 0, 0)) == 10);
  CHECK(int(chan(er, 0, 0, 1)) == 5);
  CHECK(int(chan(er, 0, 0, 2)) == 30);

  // Alpha is never morphed: it is the input pixel's own alpha in both modes.
  CHECK(int(chan(di, 0, 0, 3)) == 255);
  CHECK(int(chan(di, 1, 0, 3)) == 77);
  CHECK(int(chan(di, 2, 0, 3)) == 12);

  cv::Morphology bin;
  run(bin, img, MorphOperation::Dilate, MorphShape::Square, /*binary*/ true);
  CHECK(int(chan(bin, 1, 0, 3)) == 77);
}

// =============================================================================== edge cases

TEST_CASE("Degenerate inputs do not crash", "[morphology][edge]")
{
  SECTION("no input at all: nothing is produced")
  {
    cv::Morphology obj;
    // halp::rgba_texture declares `bool changed;` with NO default initializer, so a
    // default-constructed texture_input has indeterminate changed/bytes/width/height.
    // Reading them is UB (UBSan: "load of value 184, which is not a valid value for type
    // 'bool'") and made this suite abort nondeterministically under Catch2's random order.
    // The host always initialises the texture before invoking an object, so state the
    // "nothing was ever fed" condition explicitly rather than relying on default-init.
    obj.inputs.image.texture = {};
    obj.inputs.operation.value = MorphOperation::Dilate;
    obj();
    CHECK(obj.outputs.image.texture.width == 0);
    CHECK(obj.outputs.image.texture.bytes == nullptr);
  }

  SECTION("zero-sized texture")
  {
    cv::Morphology obj;
    obj.inputs.image.texture.bytes = nullptr;
    obj.inputs.image.texture.width = 0;
    obj.inputs.image.texture.height = 0;
    obj.inputs.image.texture.changed = true;
    obj();
    CHECK(obj.outputs.image.texture.width == 0);
  }

  SECTION("unchanged texture is skipped")
  {
    Image img(4, 4, 255);
    cv::Morphology obj;
    feed(obj.inputs.image, img);
    obj.inputs.image.texture.changed = false;
    obj();
    CHECK(obj.outputs.image.texture.width == 0);
  }

  SECTION("1x1 image is a fixed point of every operation")
  {
    Image img(1, 1, 0);
    img.setGray(0, 0, 200);
    for(auto op :
        {MorphOperation::Erode, MorphOperation::Dilate, MorphOperation::Open,
         MorphOperation::Close})
    {
      cv::Morphology bin, grey;
      run(bin, img, op, MorphShape::Square, true);
      run(grey, img, op, MorphShape::Square, false);
      CHECK(int(red(bin, 0, 0)) == 255);
      CHECK(int(red(grey, 0, 0)) == 200);
    }
  }

  SECTION("empty (all black) image stays empty, all-white stays white")
  {
    Image black(7, 5, 0);
    Image white(7, 5, 255);
    for(auto op :
        {MorphOperation::Erode, MorphOperation::Dilate, MorphOperation::Open,
         MorphOperation::Close})
    {
      cv::Morphology b, w;
      run(b, black, op);
      run(w, white, op);
      CHECK(litCount(b) == 0);
      CHECK(litCount(w) == 35);
    }
  }

  SECTION("a mid-stream dimension change is handled")
  {
    Image small(5, 5, 0);
    small.fillRect(1, 1, 3, 3, 255);
    Image big(21, 17, 0);
    big.fillRect(4, 4, 9, 9, 255);

    cv::Morphology obj;
    run(obj, small, MorphOperation::Erode);
    CHECK(litCount(obj) == 1);
    run(obj, big, MorphOperation::Erode);
    CHECK(obj.outputs.image.texture.width == 21);
    CHECK(obj.outputs.image.texture.height == 17);
    CHECK(litCount(obj) == 49);
    run(obj, small, MorphOperation::Dilate);
    CHECK(obj.outputs.image.texture.width == 5);
    CHECK(litCount(obj) == 25);
  }

  SECTION("radius larger than the image erodes everything but the full-white case")
  {
    Image img(3, 3, 0);
    img.setGray(1, 1, 255);
    cv::Morphology er, di;
    run(er, img, MorphOperation::Erode, MorphShape::Square, true, /*radius*/ 16);
    CHECK(litCount(er) == 0);
    run(di, img, MorphOperation::Dilate, MorphShape::Square, true, /*radius*/ 16);
    CHECK(litCount(di) == 9);
  }
}
