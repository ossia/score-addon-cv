// Tests for the small cv.jit primitives: CartoPol / PolToCar, FrameSub, Perimeter,
// Circularity. Every expectation below is hand-computable; the non-obvious ones carry the
// derivation in a comment.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ScoreTextureModel.hpp"
#include "TestImage.hpp"

#include <CV/Cpu/CartoPol.hpp>
#include <CV/Cpu/Circularity.hpp>
#include <CV/Cpu/FrameSub.hpp>
#include <CV/Cpu/Perimeter.hpp>
#include <CV/Cpu/PolToCar.hpp>

#include <cmath>

using Catch::Approx;
using namespace cvtest;

namespace
{
constexpr float pi = 3.14159265358979323846f;

// Fill a whole image with one grey level.
Image grey(int w, int h, std::uint8_t v)
{
  Image img(w, h, 0);
  for(int y = 0; y < h; ++y)
    for(int x = 0; x < w; ++x)
      img.setGray(x, y, v);
  return img;
}

// Rasterise a filled disc with the classic (x-cx)^2 + (y-cy)^2 <= r^2 test.
void drawDisc(Image& img, float cx, float cy, float r)
{
  for(int y = 0; y < img.height; ++y)
    for(int x = 0; x < img.width; ++x)
      if((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r)
        img.setGray(x, y, 255);
}

// ------------------------------------------------------------------ r32f contract readers
// CartoPol / PolToCar r32f outputs are normalised to [0,1] (see the contract block at the
// top of CV/Cpu/CartoPol.hpp). These undo the encoding with the object's own published
// scale, which is exactly what a downstream patch does — never read the raw float and
// pretend it is a world value.
float amp_of(cv::CartoPol& o, int i)
{
  return o.outputs.amplitude.texture.bytes[i] * o.outputs.amplitude_scale.value;
}
float phase_of(cv::CartoPol& o, int i)
{
  return cv::polar_codec::decode_phase(o.outputs.phase.texture.bytes[i]);
}
float x_of(cv::PolToCar& o, int i)
{
  return cv::polar_codec::decode_signed(
      o.outputs.x.texture.bytes[i], o.outputs.component_scale.value);
}
float y_of(cv::PolToCar& o, int i)
{
  return cv::polar_codec::decode_signed(
      o.outputs.y.texture.bytes[i], o.outputs.component_scale.value);
}
}

// =========================================================================== CartoPol
// With `Signed input` off and Range = 255 the input codec is the identity on grey levels
// (v = luma/255 * 255 = luma), so the arithmetic is exact and hand-checkable. The amplitude
// comes back out of the [0,1] r32f through `Amplitude scale` == sqrt(2)*255 = 360.62.
TEST_CASE("CartoPol maps (3,4) to amplitude 5", "[cartopol]")
{
  cv::CartoPol obj;
  Image ix = grey(2, 2, 3);
  Image iy = grey(2, 2, 4);
  feed(obj.inputs.x, ix);
  feed(obj.inputs.y, iy);
  obj.inputs.range.value = 255.f;
  obj.inputs.is_signed.value = false;

  obj();

  auto& amp = obj.outputs.amplitude.texture;
  auto& pha = obj.outputs.phase.texture;
  REQUIRE(amp.bytes != nullptr);
  REQUIRE(amp.width == 2);
  REQUIRE(amp.height == 2);
  REQUIRE(pha.bytes != nullptr);
  CHECK(r32f_in_unit_range(obj.outputs.amplitude));
  CHECK(r32f_in_unit_range(obj.outputs.phase));
  CHECK(obj.outputs.amplitude_scale.value == Approx(std::sqrt(2.f) * 255.f));
  for(int i = 0; i < 4; ++i)
  {
    CHECK(amp_of(obj, i) == Approx(5.0).margin(1e-3));                 // sqrt(9 + 16)
    CHECK(phase_of(obj, i) == Approx(std::atan2(4.f, 3.f)).margin(1e-4)); // 0.9273 rad
  }
}

TEST_CASE("CartoPol maps (1,1) to phase pi/4", "[cartopol]")
{
  cv::CartoPol obj;
  Image ix = grey(3, 3, 1);
  Image iy = grey(3, 3, 1);
  feed(obj.inputs.x, ix);
  feed(obj.inputs.y, iy);
  obj.inputs.range.value = 255.f;
  obj.inputs.is_signed.value = false;

  obj();

  REQUIRE(obj.outputs.phase.texture.bytes != nullptr);
  CHECK(r32f_in_unit_range(obj.outputs.amplitude));
  CHECK(r32f_in_unit_range(obj.outputs.phase));
  for(int i = 0; i < 9; ++i)
  {
    CHECK(phase_of(obj, i) == Approx(pi / 4.f).margin(1e-5));
    CHECK(amp_of(obj, i) == Approx(std::sqrt(2.f)).margin(1e-3));
  }
}

TEST_CASE("CartoPol decodes negative components in signed mode", "[cartopol]")
{
  // encode_signed(-0.5, 1) == 64; decoding 64 gives (2*64/255 - 1) = -0.4980392.
  REQUIRE(cv::polar_codec::encode_signed(-0.5f, 1.f) == 64);
  const float v = cv::polar_codec::decode_signed(64.f / 255.f, 1.f);
  CHECK(v == Approx(-0.4980392f).margin(1e-6));

  cv::CartoPol obj;
  Image ix = grey(2, 1, 64);
  Image iy = grey(2, 1, 64);
  feed(obj.inputs.x, ix);
  feed(obj.inputs.y, iy);
  obj.inputs.range.value = 1.f;
  obj.inputs.is_signed.value = true; // default

  obj();

  REQUIRE(obj.outputs.amplitude.texture.bytes != nullptr);
  CHECK(r32f_in_unit_range(obj.outputs.amplitude));
  CHECK(r32f_in_unit_range(obj.outputs.phase));
  // Both components negative and equal -> third quadrant, phase = -3*pi/4.
  // A NEGATIVE phase is the case that used to be destroyed on the way out (it clamped to
  // 0 and decoded as -pi); the bipolar encoding puts it at l = 0.125.
  CHECK(obj.outputs.phase.texture.bytes[0] == Approx(0.125f).margin(1e-6));
  CHECK(phase_of(obj, 0) == Approx(-3.f * pi / 4.f).margin(1e-5));
  CHECK(amp_of(obj, 0) == Approx(std::sqrt(2.f) * 0.4980392f).margin(1e-4));

  // Mid-grey is (very nearly) zero: the exact zero of a bipolar 8-bit encoding falls
  // between 127 and 128; 128 decodes to +1/255 * Range.
  cv::CartoPol z;
  Image zx = grey(1, 1, 128);
  Image zy = grey(1, 1, 128);
  feed(z.inputs.x, zx);
  feed(z.inputs.y, zy);
  z.inputs.range.value = 1.f;
  z();
  CHECK(amp_of(z, 0) == Approx(0.f).margin(0.01f));
}

TEST_CASE("CartoPol rejects mismatched input dimensions", "[cartopol]")
{
  cv::CartoPol obj;
  Image ix = grey(4, 4, 200);
  Image iy = grey(4, 3, 200); // different height
  feed(obj.inputs.x, ix);
  feed(obj.inputs.y, iy);

  obj(); // must not read out of bounds, must not emit

  CHECK(obj.outputs.amplitude.texture.width == 0);
  CHECK(obj.outputs.phase.texture.width == 0);

  // ... and the transposed case too.
  cv::CartoPol obj2;
  Image jx = grey(3, 5, 10);
  Image jy = grey(4, 5, 10); // different width
  feed(obj2.inputs.x, jx);
  feed(obj2.inputs.y, jy);
  obj2();
  CHECK(obj2.outputs.amplitude.texture.width == 0);
}

TEST_CASE("CartoPol ignores unchanged inputs", "[cartopol]")
{
  cv::CartoPol obj;
  Image ix = grey(4, 4, 200);
  Image iy = grey(4, 4, 100);
  feed(obj.inputs.x, ix);
  feed(obj.inputs.y, iy);
  obj.inputs.y.texture.changed = false;
  obj();
  CHECK(obj.outputs.amplitude.texture.width == 0);

  obj.inputs.y.texture.changed = true;
  obj.inputs.x.texture.changed = false;
  obj();
  CHECK(obj.outputs.amplitude.texture.width == 0);
}

TEST_CASE("CartoPol handles a 1x1 image", "[cartopol]")
{
  cv::CartoPol obj;
  Image ix = grey(1, 1, 255);
  Image iy = grey(1, 1, 0);
  feed(obj.inputs.x, ix);
  feed(obj.inputs.y, iy);
  obj.inputs.range.value = 1.f;
  obj();
  REQUIRE(obj.outputs.amplitude.texture.width == 1);
  REQUIRE(obj.outputs.amplitude.texture.height == 1);
  // x = +1 (white), y = -1 (black) -> amp = sqrt(2), phase = -pi/4.
  // This is the amplitude CEILING: sqrt(2)*Range, so the encoded value is exactly 1.0 and
  // nothing is clipped. Normalising by `Range` instead of sqrt(2)*Range would have lost it.
  CHECK(obj.outputs.amplitude.texture.bytes[0] == Approx(1.f).margin(1e-6));
  CHECK(amp_of(obj, 0) == Approx(std::sqrt(2.f)).margin(1e-4));
  CHECK(phase_of(obj, 0) == Approx(-pi / 4.f).margin(1e-5));
}

// =========================================================================== PolToCar
TEST_CASE("PolToCar maps (amp=1, phase=pi/2) to (0,1)", "[poltocar]")
{
  cv::PolToCar obj;
  Image ia = grey(2, 2, 255);                                    // amp = 1 * Range
  Image ip = grey(2, 2, cv::polar_codec::encode_phase(pi / 2.f)); // ~pi/2
  feed(obj.inputs.amplitude, ia);
  feed(obj.inputs.phase, ip);
  obj.inputs.range.value = 1.f;

  obj();

  REQUIRE(obj.outputs.x.texture.bytes != nullptr);
  REQUIRE(obj.outputs.y.texture.width == 2);
  CHECK(r32f_in_unit_range(obj.outputs.x));
  CHECK(r32f_in_unit_range(obj.outputs.y));
  CHECK(obj.outputs.component_scale.value == Approx(1.f));
  // Phase quantisation is 2*pi/255 = 0.0246 rad, so allow ~0.02 on the near-zero component.
  CHECK(x_of(obj, 0) == Approx(0.f).margin(0.02f));
  CHECK(y_of(obj, 0) == Approx(1.f).margin(0.01f));
}

TEST_CASE("PolToCar maps (amp=1, phase=pi) to (-1,0)", "[poltocar]")
{
  cv::PolToCar obj;
  Image ia = grey(1, 1, 255);
  Image ip = grey(1, 1, 255); // luma 1.0 -> phase = +pi
  feed(obj.inputs.amplitude, ia);
  feed(obj.inputs.phase, ip);
  obj.inputs.range.value = 2.f; // amp = 2

  obj();

  REQUIRE(obj.outputs.x.texture.bytes != nullptr);
  // x = -2 is the negative end of the bipolar swing: encoded 0.0, decoded back to -Range.
  CHECK(obj.outputs.x.texture.bytes[0] == Approx(0.f).margin(1e-6));
  CHECK(x_of(obj, 0) == Approx(-2.f).margin(1e-5));
  CHECK(y_of(obj, 0) == Approx(0.f).margin(1e-5));
}

TEST_CASE("CartoPol -> PolToCar round-trips, negatives included", "[cartopol][poltocar]")
{
  // THE regression guard for the r32f contract.
  //
  // The previous version of this test hand-called encode_unsigned()/encode_phase() between
  // the two objects. That inserted, by hand, the step that neither object ever performs, so
  // it "passed" while the real patch was broken. Here the CartoPol -> PolToCar cord goes
  // through cvtest::connect_r32f(), which is score's REAL r32f -> RGBA8 conversion
  // (Crousti/TextureConversion.hpp): gray = qBound(0, int(v * 255.f), 255).
  //
  // Signed components in [-2,2] encoded 8-bit (that part IS what a texture input carries),
  // through amplitude/phase, back to x/y.
  const float in_range = 2.f;
  const float xs[4] = {1.0f, -1.0f, -0.5f, 1.75f};
  const float ys[4] = {0.5f, 0.75f, -1.5f, -0.25f};

  Image ix(4, 1, 0), iy(4, 1, 0);
  for(int i = 0; i < 4; ++i)
  {
    ix.setGray(i, 0, cv::polar_codec::encode_signed(xs[i], in_range));
    iy.setGray(i, 0, cv::polar_codec::encode_signed(ys[i], in_range));
  }

  cv::CartoPol fwd;
  feed(fwd.inputs.x, ix);
  feed(fwd.inputs.y, iy);
  fwd.inputs.range.value = in_range;
  fwd.inputs.is_signed.value = true;
  fwd();
  REQUIRE(fwd.outputs.amplitude.texture.bytes != nullptr);

  // Contract: an r32f output must already be in [0,1], because that is the only thing
  // score's conversion can carry.
  CHECK(r32f_in_unit_range(fwd.outputs.amplitude));
  CHECK(r32f_in_unit_range(fwd.outputs.phase));

  // The amplitude ceiling in signed mode is sqrt(2)*Range (both components at +/-Range);
  // CartoPol publishes exactly that on `Amplitude scale`, which is what PolToCar's `Range`
  // must be set to.
  CHECK(fwd.outputs.amplitude_scale.value == Approx(std::sqrt(2.f) * in_range));

  cv::PolToCar inv;
  Image ia, ip;
  connect_r32f(fwd.outputs.amplitude, inv.inputs.amplitude, ia);
  connect_r32f(fwd.outputs.phase, inv.inputs.phase, ip);
  inv.inputs.range.value = fwd.outputs.amplitude_scale.value;
  inv();
  REQUIRE(inv.outputs.x.texture.bytes != nullptr);
  CHECK(r32f_in_unit_range(inv.outputs.x));
  CHECK(r32f_in_unit_range(inv.outputs.y));

  // PolToCar's components are bipolar-encoded with the same codec; `Component scale` says
  // what the full swing stands for.
  const float cs = inv.outputs.component_scale.value;
  CHECK(cs == Approx(std::sqrt(2.f) * in_range));
  auto rx = [&](int i) {
    return cv::polar_codec::decode_signed(inv.outputs.x.texture.bytes[i], cs);
  };
  auto ry = [&](int i) {
    return cv::polar_codec::decode_signed(inv.outputs.y.texture.bytes[i], cs);
  };

  // Error budget, all three quantisations being TRUNCATIONS (a full step of downward bias,
  // not half a step):
  //   input components : 2*2/255            = 0.0157 per component
  //   amplitude        : sqrt(2)*2/255      = 0.0111
  //   phase            : 2*pi/255           = 0.0246 rad -> amp*0.0246, worst amp here
  //                                           1.766 (1.75,-0.25) -> 0.0435
  // Sum 0.070; the assertion allows 0.09.
  for(int i = 0; i < 4; ++i)
  {
    INFO("component " << i);
    CHECK(rx(i) == Approx(xs[i]).margin(0.09f));
    CHECK(ry(i) == Approx(ys[i]).margin(0.09f));
  }
}

TEST_CASE(
    "CartoPol -> PolToCar -> CartoPol closes the loop through score's conversion",
    "[cartopol][poltocar]")
{
  // The full documented cycle, every hop going through score's real r32f -> RGBA8 step.
  const float in_range = 1.f;
  const float xs[3] = {0.8f, -0.6f, -0.25f};
  const float ys[3] = {0.4f, -0.5f, 0.9f};

  Image ix(3, 1, 0), iy(3, 1, 0);
  for(int i = 0; i < 3; ++i)
  {
    ix.setGray(i, 0, cv::polar_codec::encode_signed(xs[i], in_range));
    iy.setGray(i, 0, cv::polar_codec::encode_signed(ys[i], in_range));
  }

  cv::CartoPol c1;
  feed(c1.inputs.x, ix);
  feed(c1.inputs.y, iy);
  c1.inputs.range.value = in_range;
  c1();
  REQUIRE(c1.outputs.amplitude.texture.bytes != nullptr);

  cv::PolToCar p;
  Image ia, ip;
  connect_r32f(c1.outputs.amplitude, p.inputs.amplitude, ia);
  connect_r32f(c1.outputs.phase, p.inputs.phase, ip);
  p.inputs.range.value = c1.outputs.amplitude_scale.value;
  p();
  REQUIRE(p.outputs.x.texture.bytes != nullptr);

  // Back into a CartoPol: `Range` takes PolToCar's `Component scale`, signed mode on.
  cv::CartoPol c2;
  Image jx, jy;
  connect_r32f(p.outputs.x, c2.inputs.x, jx);
  connect_r32f(p.outputs.y, c2.inputs.y, jy);
  c2.inputs.range.value = p.outputs.component_scale.value;
  c2.inputs.is_signed.value = true;
  c2();
  REQUIRE(c2.outputs.amplitude.texture.bytes != nullptr);
  CHECK(r32f_in_unit_range(c2.outputs.amplitude));
  CHECK(r32f_in_unit_range(c2.outputs.phase));

  // The second CartoPol sees a different Range (sqrt(2)*1 instead of 1), so compare in
  // world units and radians, not in encoded units.
  for(int i = 0; i < 3; ++i)
  {
    INFO("component " << i);
    const float a1 = c1.outputs.amplitude.texture.bytes[i] * c1.outputs.amplitude_scale.value;
    const float a2 = c2.outputs.amplitude.texture.bytes[i] * c2.outputs.amplitude_scale.value;
    const float f1 = cv::polar_codec::decode_phase(c1.outputs.phase.texture.bytes[i]);
    const float f2 = cv::polar_codec::decode_phase(c2.outputs.phase.texture.bytes[i]);
    // Five truncating 8-bit hops; 0.05 in amplitude and 0.09 rad in phase covers them.
    CHECK(a2 == Approx(a1).margin(0.05f));
    CHECK(f2 == Approx(f1).margin(0.09f));
  }
}

TEST_CASE("PolToCar rejects mismatched input dimensions", "[poltocar]")
{
  cv::PolToCar obj;
  Image ia = grey(5, 5, 128);
  Image ip = grey(2, 2, 128);
  feed(obj.inputs.amplitude, ia);
  feed(obj.inputs.phase, ip);
  obj();
  CHECK(obj.outputs.x.texture.width == 0);
  CHECK(obj.outputs.y.texture.width == 0);
}

TEST_CASE("PolToCar ignores unchanged inputs", "[poltocar]")
{
  cv::PolToCar obj;
  Image ia = grey(4, 4, 200);
  Image ip = grey(4, 4, 100);
  feed(obj.inputs.amplitude, ia);
  feed(obj.inputs.phase, ip);
  obj.inputs.amplitude.texture.changed = false;
  obj();
  CHECK(obj.outputs.x.texture.width == 0);
}

TEST_CASE("PolToCar handles a 1x1 image", "[poltocar]")
{
  cv::PolToCar obj;
  Image ia = grey(1, 1, 0); // amplitude 0
  Image ip = grey(1, 1, 200);
  feed(obj.inputs.amplitude, ia);
  feed(obj.inputs.phase, ip);
  obj();
  REQUIRE(obj.outputs.x.texture.width == 1);
  // Zero components sit at the middle of the bipolar swing, not at 0.0.
  CHECK(obj.outputs.x.texture.bytes[0] == Approx(0.5f).margin(1e-6));
  CHECK(x_of(obj, 0) == Approx(0.f).margin(1e-6));
  CHECK(y_of(obj, 0) == Approx(0.f).margin(1e-6));
}

// =========================================================================== FrameSub
TEST_CASE("FrameSub outputs black on the first frame", "[framesub]")
{
  cv::FrameSub obj;
  Image a(4, 4, 0);
  a.fillRect(0, 0, 4, 4, 200);
  feed(obj.inputs.image, a);

  obj();

  auto& out = obj.outputs.image.texture;
  REQUIRE(out.bytes != nullptr);
  REQUIRE(out.width == 4);
  REQUIRE(out.height == 4);
  for(int i = 0; i < 16; ++i)
  {
    CHECK(out.bytes[i * 4 + 0] == 0);
    CHECK(out.bytes[i * 4 + 1] == 0);
    CHECK(out.bytes[i * 4 + 2] == 0);
    CHECK(out.bytes[i * 4 + 3] == 255); // opaque, not transparent
  }
}

TEST_CASE("FrameSub gives |B - A| per channel", "[framesub]")
{
  cv::FrameSub obj;

  Image a(2, 2, 0);
  a.set(0, 0, 10, 20, 30);
  a.set(1, 0, 200, 100, 50);
  a.set(0, 1, 0, 0, 0);
  a.set(1, 1, 255, 255, 255);
  feed(obj.inputs.image, a);
  obj(); // first frame: primes the history

  Image b(2, 2, 0);
  b.set(0, 0, 40, 5, 30);    // |30|, |15|, |0|
  b.set(1, 0, 100, 100, 60); // |100|, |0|, |10|
  b.set(0, 1, 7, 8, 9);      // |7|, |8|, |9|
  b.set(1, 1, 0, 128, 255);  // |255|, |127|, |0|
  feed(obj.inputs.image, b);
  obj();

  auto& out = obj.outputs.image.texture;
  REQUIRE(out.bytes != nullptr);
  const std::uint8_t expected[4][3]
      = {{30, 15, 0}, {100, 0, 10}, {7, 8, 9}, {255, 127, 0}};
  for(int i = 0; i < 4; ++i)
    for(int c = 0; c < 3; ++c)
      CHECK(static_cast<int>(out.bytes[i * 4 + c]) == static_cast<int>(expected[i][c]));
}

TEST_CASE("FrameSub gives zero for a repeated frame", "[framesub]")
{
  cv::FrameSub obj;
  Image a(8, 8, 0);
  a.fillRect(2, 2, 4, 4, 255);
  a.fillRect(0, 0, 2, 8, 77);

  feed(obj.inputs.image, a);
  obj(); // first frame -> black
  feed(obj.inputs.image, a);
  obj(); // same frame -> zero everywhere

  auto& out = obj.outputs.image.texture;
  REQUIRE(out.bytes != nullptr);
  for(int i = 0; i < 64; ++i)
    for(int c = 0; c < 3; ++c)
      CHECK(static_cast<int>(out.bytes[i * 4 + c]) == 0);
}

TEST_CASE("FrameSub survives a dimension change", "[framesub]")
{
  cv::FrameSub obj;
  Image a(8, 8, 200);
  feed(obj.inputs.image, a);
  obj();

  Image b(3, 5, 100); // resized input: history is unusable
  feed(obj.inputs.image, b);
  obj();

  auto& out = obj.outputs.image.texture;
  REQUIRE(out.bytes != nullptr);
  CHECK(out.width == 3);
  CHECK(out.height == 5);
  for(int i = 0; i < 15; ++i)
    for(int c = 0; c < 3; ++c)
      CHECK(static_cast<int>(out.bytes[i * 4 + c]) == 0); // treated as a first frame

  // and the next frame at the new size differences normally
  Image c(3, 5, 130);
  feed(obj.inputs.image, c);
  obj();
  for(int i = 0; i < 15; ++i)
    CHECK(static_cast<int>(obj.outputs.image.texture.bytes[i * 4]) == 30);
}

TEST_CASE("FrameSub ignores an unchanged frame", "[framesub]")
{
  cv::FrameSub obj;
  Image a(4, 4, 128);
  feed(obj.inputs.image, a);
  obj.inputs.image.texture.changed = false;
  obj();
  CHECK(obj.outputs.image.texture.width == 0);
}

TEST_CASE("FrameSub handles a 1x1 image", "[framesub]")
{
  cv::FrameSub obj;
  Image a(1, 1, 10);
  feed(obj.inputs.image, a);
  obj();
  REQUIRE(obj.outputs.image.texture.width == 1);
  CHECK(static_cast<int>(obj.outputs.image.texture.bytes[0]) == 0);

  Image b(1, 1, 45);
  feed(obj.inputs.image, b);
  obj();
  CHECK(static_cast<int>(obj.outputs.image.texture.bytes[0]) == 35);
}

// ========================================================================== Perimeter
TEST_CASE("Perimeter of a solid k x k square is 4k-4", "[perimeter]")
{
  // A 6x6 block has 36 pixels of which the inner 4x4 = 16 have all 8 neighbours set,
  // so 36 - 16 = 20 boundary pixels = 4*6 - 4.
  cv::Perimeter obj;
  Image img(16, 16, 0);
  img.fillRect(4, 4, 6, 6, 255);
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.perimeter.value == 20);

  // 3x3 -> 4*3 - 4 = 8 (only the centre pixel is interior).
  cv::Perimeter obj2;
  Image img2(16, 16, 0);
  img2.fillRect(5, 5, 3, 3, 255);
  feed(obj2.inputs.image, img2);
  obj2();
  CHECK(obj2.outputs.perimeter.value == 8);
}

TEST_CASE("Perimeter of a single isolated pixel is 1", "[perimeter]")
{
  cv::Perimeter obj;
  Image img(9, 9, 0);
  img.setGray(4, 4, 255);
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.perimeter.value == 1);
}

TEST_CASE("Perimeter border rule: a fully white image", "[perimeter]")
{
  Image img = grey(8, 8, 255);

  // cv.jit.binedge rule (default): off-image neighbours are not tested, so no pixel of a
  // uniformly white image has a background neighbour -> nothing bounds it, perimeter 0.
  cv::Perimeter open;
  feed(open.inputs.image, img);
  open.inputs.closed_border.value = false;
  open();
  CHECK(open.outputs.perimeter.value == 0);

  // Closed-border option: the image edge counts as background, so the answer is the border
  // ring, 2*8 + 2*8 - 4 = 28 pixels.
  cv::Perimeter closed;
  feed(closed.inputs.image, img);
  closed.inputs.closed_border.value = true;
  closed();
  CHECK(closed.outputs.perimeter.value == 28);
}

TEST_CASE("Perimeter of an empty image is 0", "[perimeter]")
{
  cv::Perimeter obj;
  Image img(12, 7, 0);
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.perimeter.value == 0);

  // ... including with the closed-border rule: there is no foreground to bound.
  obj.inputs.closed_border.value = true;
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.perimeter.value == 0);
}

TEST_CASE("Perimeter honours the threshold", "[perimeter]")
{
  cv::Perimeter obj;
  Image img(16, 16, 0);
  img.fillRect(4, 4, 6, 6, 100); // mid-dark square
  feed(obj.inputs.image, img);

  obj.inputs.threshold.value = 0.5f; // 128 -> square is background
  obj();
  CHECK(obj.outputs.perimeter.value == 0);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.25f; // 64 -> square is foreground
  obj();
  CHECK(obj.outputs.perimeter.value == 20);
}

TEST_CASE("Perimeter ignores an unchanged frame and survives 1x1", "[perimeter]")
{
  cv::Perimeter obj;
  Image img = grey(4, 4, 255);
  feed(obj.inputs.image, img);
  obj.inputs.image.texture.changed = false;
  obj.outputs.perimeter = 12345;
  obj();
  CHECK(obj.outputs.perimeter.value == 12345); // untouched

  cv::Perimeter tiny;
  Image one = grey(1, 1, 255);
  feed(tiny.inputs.image, one);
  tiny();
  CHECK(tiny.outputs.perimeter.value == 0); // no neighbour is testable
  tiny.inputs.closed_border.value = true;
  feed(tiny.inputs.image, one);
  tiny();
  CHECK(tiny.outputs.perimeter.value == 1);
}

// ======================================================================== Circularity
TEST_CASE("Circularity of a solid square matches 4*pi*A/P^2", "[circularity]")
{
  // 6x6 square: A = 36, P = 20 -> 4*pi*36/400 = 1.1309734.
  cv::Circularity obj;
  Image img(40, 40, 0);
  img.fillRect(10, 10, 6, 6, 255);
  feed(obj.inputs.image, img);
  obj();

  CHECK(obj.outputs.area.value == 36);
  CHECK(obj.outputs.perimeter.value == 20);
  CHECK(obj.outputs.circularity.value == Approx(1.1309734f).epsilon(1e-5));

  // 20x20 square: A = 400, P = 76 -> 4*pi*400/5776 = 0.8702473.
  cv::Circularity obj2;
  Image img2(40, 40, 0);
  img2.fillRect(10, 10, 20, 20, 255);
  feed(obj2.inputs.image, img2);
  obj2();
  CHECK(obj2.outputs.area.value == 400);
  CHECK(obj2.outputs.perimeter.value == 76);
  CHECK(obj2.outputs.circularity.value == Approx(0.8702473f).epsilon(1e-5));
}

TEST_CASE("Circularity ranks a disc far above a thin bar", "[circularity]")
{
  // Disc of radius 10 centred on (19.5,19.5) in a 40x40 frame: 316 pixels, 76 of them
  // boundary -> 4*pi*316/5776 = 0.6874953.
  cv::Circularity disc;
  Image dimg(40, 40, 0);
  drawDisc(dimg, 19.5f, 19.5f, 10.f);
  feed(disc.inputs.image, dimg);
  disc();
  CHECK(disc.outputs.area.value == 316);
  CHECK(disc.outputs.perimeter.value == 76);
  CHECK(disc.outputs.circularity.value == Approx(0.6874953f).epsilon(1e-5));

  // A 30x1 bar is all boundary (A = P = 30) -> 4*pi/30 = 0.4188790.
  cv::Circularity bar;
  Image bimg(40, 40, 0);
  bimg.fillRect(5, 20, 30, 1, 255);
  feed(bar.inputs.image, bimg);
  bar();
  CHECK(bar.outputs.area.value == 30);
  CHECK(bar.outputs.perimeter.value == 30);
  CHECK(bar.outputs.circularity.value == Approx(0.4188790f).epsilon(1e-5));

  CHECK(disc.outputs.circularity.value > bar.outputs.circularity.value);

  // NOTE: with the digital (binedge+mass) perimeter a square does NOT score below a disc:
  // the 8-connected boundary count over-estimates the length of a curve (each diagonal step
  // counts as one pixel, though it spans sqrt(2)) while tracking an axis-aligned edge
  // exactly. That is cv.jit's metric and we reproduce it rather than "fixing" it, so the
  // score is only comparable between shapes of similar scale. Pin the relation so a future
  // change to the perimeter rule is visible:
  cv::Circularity square;
  Image s2(40, 40, 0);
  s2.fillRect(10, 10, 18, 18, 255); // area 324, comparable to the disc's 316
  feed(square.inputs.image, s2);
  square();
  CHECK(square.outputs.area.value == 324);
  CHECK(square.outputs.circularity.value > disc.outputs.circularity.value);
}

TEST_CASE("Circularity of an empty image does not divide by zero", "[circularity]")
{
  cv::Circularity obj;
  Image img(10, 10, 0);
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.area.value == 0);
  CHECK(obj.outputs.perimeter.value == 0);
  CHECK(obj.outputs.circularity.value == 0.f);
  CHECK(std::isfinite(obj.outputs.circularity.value));
}

TEST_CASE("Circularity ignores an unchanged frame and survives 1x1", "[circularity]")
{
  cv::Circularity obj;
  Image img = grey(4, 4, 255);
  feed(obj.inputs.image, img);
  obj.inputs.image.texture.changed = false;
  obj.outputs.circularity = 42.f;
  obj();
  CHECK(obj.outputs.circularity.value == 42.f);

  cv::Circularity tiny;
  Image one = grey(1, 1, 255);
  feed(tiny.inputs.image, one);
  tiny();
  CHECK(tiny.outputs.area.value == 1);
  CHECK(tiny.outputs.perimeter.value == 0); // cv.jit border rule
  CHECK(tiny.outputs.circularity.value == 0.f);

  tiny.inputs.closed_border.value = true;
  feed(tiny.inputs.image, one);
  tiny();
  CHECK(tiny.outputs.perimeter.value == 1);
  CHECK(tiny.outputs.circularity.value == Approx(4.f * pi).epsilon(1e-5));
}
