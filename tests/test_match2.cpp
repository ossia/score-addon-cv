// Tests for the restored cv.jit keypoint pipeline:
//   * FeatureMatch in TWO-SET mode  — matching two independently produced keypoint sets fed
//     as lists, which is what cv.jit.keypoints.match's 4 matrix inlets actually are;
//   * the Lowe ratio, whose cv.jit value is a hard-coded 0.7;
//   * OrbFeatures' scale space (`Octaves`, cv.jit's `octaves` attribute, default 4) and the
//     keypoint fields cv.jit's 6-plane row carries (size / response / octave).
//
// ---------------------------------------------------------------------------------------
// DERIVATIONS (everything asserted below is exact, not "looks about right")
//
// 1. Two-set translation. makeTexture(W,H,sx,sy) writes hash(x-sx, y-sy), i.e. the SAME
//    pattern translated by (sx,sy). Detection is deterministic and translation-equivariant
//    away from the borders, so a corner at (x,y) in set A appears at (x+sx, y+sy) in set B
//    with a bit-identical descriptor. Normalised, the displacement is exactly (sx/W, sy/H).
//    With (sx,sy) = (3,2) on 96x96 that is (0.03125, 0.0208333...).
//
// 2. Scale space. The pyramid downsamples by a 2x2 BOX AVERAGE. `upscale2x` replicates each
//    pixel into a 2x2 block. Box-averaging a 2x replication returns the original value
//    *exactly* (a+a+a+a)*0.25 == a in IEEE arithmetic, and the grayscale conversion is
//    applied per-pixel before that. Therefore, for base image I:
//        level 1 of upscale2x(I)  ==  level 0 of I,   bit for bit.
//    So every octave-0 keypoint of I reappears as an octave-1 keypoint of upscale2x(I) with
//    a bit-identical descriptor => Hamming distance exactly 0. Base-image coordinates:
//    level-1 pixel x maps to (x + 0.5) * 2 - 0.5 = 2x + 0.5, so
//        posB_px  ==  2 * posA_px + 0.5     exactly.
//    At Octaves = 1 the pyramid does not exist, upscale2x(I) is only ever examined at its
//    blocky full resolution, and NOT ONE correspondence survives (asserted: 0 matches).
//
// 3. Lowe's ratio. matchRatio accepts when best < ratio * second. With hand-built
//    descriptors giving best = 10, second = 12:
//        ratio 0.7 -> 10 < 8.4  : REJECTED  (cv.jit's hard-coded value)
//        ratio 0.9 -> 10 < 10.8 : ACCEPTED
//    and with best = 2, second = 100: 2 < 70 at ratio 0.7 -> ACCEPTED, proving the
//    rejection above comes from ambiguity and not from a broken threshold.
//
// 4. size / octave. size == 31 * 2^octave by construction (kBaseKeypointSize = 31, the ORB
//    patch size, scaled by the octave's 2^o).
//
// 5. GENERAL scale. `upscale2x` is the exact right-inverse of the pyramid's 2x2 box average,
//    so derivation 2 proves only that the pyramid inverts its own downsampler -- it says
//    nothing about a scale change the pyramid does not land on. The test at the bottom of
//    this file therefore also rescales by 1.4x (bilinear) and measures the recall.
//    That factor is the WORST CASE for a halving pyramid: consecutive levels are a factor 2
//    apart, so a feature's true scale is at worst sqrt(2) = 1.414 away from the nearest level.
//    The band-limited `makeSmooth` scene is used instead of `makeTexture`, because hash noise
//    is white at the pixel scale and no resampling of it could preserve any descriptor --
//    that would test the test image, not the detector.
// ---------------------------------------------------------------------------------------

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/FeatureMatch.hpp>
#include <CV/Cpu/OrbFeatures.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

using Catch::Approx;
using namespace cvtest;

namespace
{
// Same deterministic, spatially-unique texture used by test_features.cpp: every feature sits
// in a locally-unique neighbourhood, so BRIEF descriptors are discriminative. (sx,sy)
// translates the whole pattern.
Image makeTexture(int W, int H, int sx, int sy)
{
  Image img(W, H, 0);
  auto hash = [](int x, int y) -> std::uint8_t {
    std::uint64_t h
        = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x + 1000)) * 374761u
          + static_cast<std::uint64_t>(static_cast<std::uint32_t>(y + 1000)) * 668265u;
    h = ((h ^ (h >> 13)) & 0xFFFFFFu) * 12743u; // masked so the product stays < 2^40
    return static_cast<std::uint8_t>((h >> 5) & 0xFF);
  };
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
      img.setGray(x, y, hash(x - sx, y - sy));
  return img;
}

// Exact 2x nearest-neighbour magnification: pixel (x,y) becomes the 2x2 block at (2x,2y).
// This is the exact right-inverse of the pyramid's 2x2 box average (see derivation 2).
Image upscale2x(const Image& s)
{
  Image d(s.width * 2, s.height * 2, 0);
  for(int y = 0; y < d.height; ++y)
    for(int x = 0; x < d.width; ++x)
    {
      const auto* p = &s.px[(static_cast<std::size_t>(y / 2) * s.width + x / 2) * 4];
      d.set(x, y, p[0], p[1], p[2], p[3]);
    }
  return d;
}

// A band-limited "natural-looking" scene: the same hash evaluated on a coarse grid of `cell`
// px and smoothstep-interpolated up. All its structure lives at ~cell px, i.e. well above the
// Nyquist limit, so it survives resampling — unlike makeTexture, whose per-pixel hash is white
// noise and is destroyed by any non-integer rescale.
Image makeSmooth(int W, int H, int cell)
{
  auto hash = [](int x, int y) -> float {
    std::uint64_t h
        = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x + 1000)) * 374761u
          + static_cast<std::uint64_t>(static_cast<std::uint32_t>(y + 1000)) * 668265u;
    h = ((h ^ (h >> 13)) & 0xFFFFFFu) * 12743u;
    return static_cast<float>((h >> 5) & 0xFF);
  };
  Image d(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      const float fx = static_cast<float>(x) / cell, fy = static_cast<float>(y) / cell;
      const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
      float tx = fx - x0, ty = fy - y0;
      tx = tx * tx * (3.f - 2.f * tx); // smoothstep -> C1, no interpolation creases
      ty = ty * ty * (3.f - 2.f * ty);
      const float v = hash(x0, y0) * (1 - tx) * (1 - ty) + hash(x0 + 1, y0) * tx * (1 - ty)
                      + hash(x0, y0 + 1) * (1 - tx) * ty + hash(x0 + 1, y0 + 1) * tx * ty;
      d.setGray(x, y, static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.f, 255.f))));
    }
  return d;
}

// Bilinear rescale by an ARBITRARY factor, half-pixel convention: destination pixel centre
// (x+0.5)/f maps to source centre, so the mapping is dst_px = (src_px + 0.5)*f - 0.5.
Image rescale(const Image& s, float f)
{
  const int W = static_cast<int>(std::lround(s.width * f));
  const int H = static_cast<int>(std::lround(s.height * f));
  Image d(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      const float sxf = (x + 0.5f) / f - 0.5f, syf = (y + 0.5f) / f - 0.5f;
      const int x0 = static_cast<int>(std::floor(sxf)), y0 = static_cast<int>(std::floor(syf));
      const float tx = sxf - x0, ty = syf - y0;
      auto g = [&](int xx, int yy) -> float {
        xx = std::clamp(xx, 0, s.width - 1);
        yy = std::clamp(yy, 0, s.height - 1);
        return static_cast<float>(s.px[(static_cast<std::size_t>(yy) * s.width + xx) * 4]);
      };
      const float v = g(x0, y0) * (1 - tx) * (1 - ty) + g(x0 + 1, y0) * tx * (1 - ty)
                      + g(x0, y0 + 1) * (1 - tx) * ty + g(x0 + 1, y0 + 1) * tx * ty;
      d.setGray(x, y, static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.f, 255.f))));
    }
  return d;
}

std::vector<cv::keypoint>
detect(Image& img, int octaves, int maxFeatures, float threshold = 0.08f)
{
  cv::OrbFeatures o;
  o.inputs.threshold.value = threshold;
  o.inputs.max_features.value = maxFeatures;
  o.inputs.octaves.value = octaves;
  feed(o.inputs.image, img);
  o();
  REQUIRE(o.outputs.count.value == static_cast<int>(o.outputs.keypoints.value.size()));
  return o.outputs.keypoints.value;
}

// Match two externally produced sets. NO texture is ever fed to the matcher here: this is
// exactly the capability that did not exist before (cv.jit's 4 matrix inlets).
std::vector<cv::feature_match> matchSets(
    const std::vector<cv::keypoint>& a, const std::vector<cv::keypoint>& b, float ratio)
{
  cv::FeatureMatch m;
  m.inputs.ratio.value = ratio;
  m.inputs.set_a.value = a;
  m.inputs.set_b.value = b;
  m(); // inputs.image.texture is left completely untouched (bytes == nullptr)
  CHECK(m.outputs.count.value == static_cast<int>(m.outputs.matches.value.size()));
  return m.outputs.matches.value;
}

float median(std::vector<float> v)
{
  REQUIRE(!v.empty());
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

cv_support::descriptor descBits(std::initializer_list<std::pair<int, int>> ranges)
{
  cv_support::descriptor d{};
  for(auto [from, count] : ranges)
    for(int b = from; b < from + count; ++b)
      d[static_cast<std::size_t>(b >> 5)] |= (1u << (b & 31));
  return d;
}

cv::keypoint mk(float x, float y, const cv_support::descriptor& d)
{
  cv::keypoint k;
  k.position = {x, y};
  k.size = 31.f;
  k.angle = 0.f;
  k.response = 1.f;
  k.octave = 0;
  k.desc = d;
  return k;
}

// Expected `size` for a keypoint, in the object's ACTIVE coordinate mode.
//
// cv.jit.keypoints scales SIZE by max(scale_x, scale_y), and the port's default mode is
// Normalized (scale_x = 1/W, scale_y = 1/H), so the factor is max(1/W, 1/H) = 1/min(W,H).
// The base BRIEF patch diameter is 31 px at octave 0 and doubles per octave.
// NB: these are NOT base-image pixels -- asserting a raw 31.f here was correct only before
// the SIZE-scaling parity fix.
inline float expectedSize(int w, int h, int octave = 0)
{
  return 31.f * static_cast<float>(1 << octave) / static_cast<float>(std::min(w, h));
}
} // namespace

// ============================================================ parity: cv.jit's Lowe ratio
TEST_CASE("FeatureMatch defaults to cv.jit's hard-coded Lowe ratio of 0.7", "[match2]")
{
  cv::FeatureMatch m;
  // cv.jit.keypoints.match.cpp: `constexpr float LOWES_RATIO = 0.7f;`
  CHECK(m.inputs.ratio.value == Approx(0.7f));
}

TEST_CASE("OrbFeatures defaults to cv.jit's 4 octaves", "[match2]")
{
  cv::OrbFeatures o;
  // cv.jit.keypoints.cpp: `x->octaves = 4;`
  CHECK(o.inputs.octaves.value == 4);
}

// ============================================================== two independent sets
TEST_CASE(
    "FeatureMatch matches two externally supplied keypoint sets (no internal detection)",
    "[match2]")
{
  // Two DIFFERENT images, each run through its OWN OrbFeatures instance — the cv.jit
  // help-patch topology (template image vs live camera). The matcher never sees a texture.
  Image i0 = makeTexture(96, 96, 0, 0);
  Image i1 = makeTexture(96, 96, 3, 2);

  auto A = detect(i0, 1, 512);
  auto B = detect(i1, 1, 512);
  REQUIRE(A.size() == 512u);
  REQUIRE(B.size() == 512u);

  auto matches = matchSets(A, B, 0.7f);
  REQUIRE(matches.size() > 100u);

  // Derivation 1: displacement is exactly (3/96, 2/96). Use the median to be robust against
  // the handful of spurious matches any descriptor matcher produces.
  std::vector<float> dxs, dys;
  for(auto& m : matches)
  {
    dxs.push_back(m.cur.x - m.prev.x);
    dys.push_back(m.cur.y - m.prev.y);
  }
  CHECK(median(dxs) == Approx(3.f / 96.f).margin(1e-6));
  CHECK(median(dys) == Approx(2.f / 96.f).margin(1e-6));

  // The full 6-plane keypoint payload of BOTH sides is carried through, as cv.jit does with
  // its two output matrices — not just the coordinates.
  for(auto& m : matches)
  {
    CHECK(m.prev_octave == 0);
    CHECK(m.cur_octave == 0);
    CHECK(m.prev_size == Approx(expectedSize(96, 96)));
    CHECK(m.cur_size == Approx(expectedSize(96, 96)));
    CHECK(m.prev_response > 0.f);
    CHECK(m.cur_response > 0.f);
    CHECK(std::abs(m.prev_angle) <= 3.15f); // radians, not degrees
    CHECK(std::abs(m.cur_angle) <= 3.15f);
    CHECK(m.distance >= 0.f);
  }

  // Every reported pair must really be present in the inputs it claims to come from.
  for(auto& m : matches)
  {
    const bool inA = std::any_of(A.begin(), A.end(), [&](const cv::keypoint& k) {
      return k.position.x == m.prev.x && k.position.y == m.prev.y;
    });
    const bool inB = std::any_of(B.begin(), B.end(), [&](const cv::keypoint& k) {
      return k.position.x == m.cur.x && k.position.y == m.cur.y;
    });
    CHECK(inA);
    CHECK(inB);
  }
}

TEST_CASE("FeatureMatch two-set mode performs NO detection of its own", "[match2]")
{
  // Feed a texture *as well as* the two sets. If the object detected internally, (a) the
  // result would differ from the pure-list result, and (b) its temporal state would be
  // primed, so the very next texture-only tick would already produce matches. Assert neither.
  Image i0 = makeTexture(96, 96, 0, 0);
  Image i1 = makeTexture(96, 96, 3, 2);
  auto A = detect(i0, 1, 512);
  auto B = detect(i1, 1, 512);
  const auto expected = matchSets(A, B, 0.7f);

  cv::FeatureMatch m;
  m.inputs.ratio.value = 0.7f;
  m.inputs.set_a.value = A;
  m.inputs.set_b.value = B;

  for(int tick = 0; tick < 3; ++tick)
  {
    feed(m.inputs.image, i0); // a perfectly valid, changed texture — must be ignored
    m();
    REQUIRE(m.outputs.matches.value.size() == expected.size());
    for(std::size_t i = 0; i < expected.size(); ++i)
    {
      CHECK(m.outputs.matches.value[i].prev.x == expected[i].prev.x);
      CHECK(m.outputs.matches.value[i].cur.x == expected[i].cur.x);
      CHECK(m.outputs.matches.value[i].distance == expected[i].distance);
    }
  }

  // Now disconnect the lists. Texture mode has NO previous frame yet — proof that the three
  // texture ticks above were never detected on. The first texture tick therefore yields 0.
  m.inputs.set_a.value.clear();
  m.inputs.set_b.value.clear();
  feed(m.inputs.image, i0);
  m();
  CHECK(m.outputs.count.value == 0);

  // ...and only the SECOND texture tick can match, as it always could.
  feed(m.inputs.image, i1);
  m();
  CHECK(m.outputs.count.value > 0);
}

// =============================================== degenerate sets: cv.jit's silent tolerance
TEST_CASE("FeatureMatch tolerates degenerate keypoint sets without crashing", "[match2]")
{
  Image i0 = makeTexture(96, 96, 0, 0);
  auto full = detect(i0, 1, 512);
  REQUIRE(full.size() > 2u);

  const auto d0 = descBits({{0, 4}});
  const auto d1 = descBits({{64, 4}});

  SECTION("empty A, non-empty B -> silent empty (cv.jit returns JIT_ERR_NONE)")
  {
    CHECK(matchSets({}, full, 0.7f).empty());
  }
  SECTION("non-empty A, empty B -> silent empty")
  {
    CHECK(matchSets(full, {}, 0.7f).empty());
  }
  SECTION("exactly one descriptor on each side -> empty, no crash")
  {
    // THIS IS THE cv.jit CRASH CASE. cv.jit hands k = 2 to knnMatch then reads match[1]
    // unconditionally; with a single train descriptor that inner vector has one element and
    // the read is out of bounds. Here it must simply produce nothing.
    CHECK(matchSets({mk(0.1f, 0.1f, d0)}, {mk(0.2f, 0.2f, d1)}, 0.7f).empty());
  }
  SECTION("many queries against exactly one train descriptor -> empty, no crash")
  {
    CHECK(matchSets(full, {mk(0.2f, 0.2f, d1)}, 0.7f).empty());
  }
  SECTION("exactly one query against many train descriptors -> at most one match")
  {
    // The reverse direction is NOT degenerate (a second-nearest exists), so this must work.
    auto r = matchSets({full[0]}, full, 0.7f);
    CHECK(r.size() <= 1u);
  }
  SECTION("a set that was populated then emptied clears the output")
  {
    cv::FeatureMatch m;
    m.inputs.ratio.value = 0.7f;
    m.inputs.set_a.value = full;
    m.inputs.set_b.value = detect(i0, 1, 512);
    m();
    REQUIRE(m.outputs.count.value > 0);
    m.inputs.set_b.value.clear();
    m();
    CHECK(m.outputs.count.value == 0);
    CHECK(m.outputs.matches.value.empty());
  }
}

TEST_CASE("FeatureMatch falls back to texture mode when both lists are empty", "[match2]")
{
  // Both list inputs empty == "not wired": the historical texture-in temporal behaviour must
  // be completely untouched, at the new 0.7 default ratio.
  cv::FeatureMatch m;
  m.inputs.threshold.value = 0.08f;
  m.inputs.max_features.value = 512;
  // ratio left at its default (0.7) on purpose.

  Image f0 = makeTexture(96, 96, 0, 0);
  feed(m.inputs.image, f0);
  m();
  CHECK(m.outputs.count.value == 0); // no previous frame

  Image f1 = makeTexture(96, 96, 2, 1);
  feed(m.inputs.image, f1);
  m();
  REQUIRE(m.outputs.count.value > 0);

  std::vector<float> dxs, dys;
  for(auto& mm : m.outputs.matches.value)
  {
    dxs.push_back(mm.cur.x - mm.prev.x);
    dys.push_back(mm.cur.y - mm.prev.y);
  }
  CHECK(median(dxs) == Approx(2.f / 96.f).margin(1e-6));
  CHECK(median(dys) == Approx(1.f / 96.f).margin(1e-6));

  // Texture mode also fills the extra keypoint planes now.
  for(auto& mm : m.outputs.matches.value)
  {
    CHECK(mm.prev_octave == 0);
    CHECK(mm.cur_octave == 0);
    CHECK(mm.prev_size == Approx(expectedSize(96, 96)));
    CHECK(mm.prev_response > 0.f);
  }
}

// ====================================================== query/train direction of the two paths
//
// Lowe's ratio test is ASYMMETRIC: it walks the QUERY set and keeps at most one match per
// query keypoint, while a TRAIN keypoint may be claimed by any number of queries. So the
// direction is not a detail — it decides which side of `feature_match` carries the "at most
// one" guarantee. cv.jit is unambiguous (cv.jit.keypoints.match.cpp calls
// knnMatch(descriptors1, descriptors2, ...) and queryIdx indexes set 1, the template side),
// which is the `prev` side here; template -> scene is also the right direction for reference
// matching, since it asks "where did each template feature go?".
//
// Texture mode used to query with the CURRENT frame instead, so `prev` was the train side and
// the very same output field meant opposite things depending on which inlet was wired.
//
// The two paths are otherwise fed identically (OrbFeatures at Octaves = 1 runs exactly the
// detectAndDescribe() that texture mode runs, and both normalise positions by 1/W, 1/H and
// scale `size` by max(1/W, 1/H)), so with the direction fixed their outputs must agree
// ELEMENT FOR ELEMENT, including the order — both enumerate the query set in index order.
TEST_CASE("Texture mode and two-set mode agree exactly: prev is the query side in both",
          "[match2][direction]")
{
  Image i0 = makeTexture(96, 96, 0, 0);
  Image i1 = makeTexture(96, 96, 3, 2);

  auto A = detect(i0, 1, 512);
  auto B = detect(i1, 1, 512);
  const auto twoSet = matchSets(A, B, 0.7f);
  REQUIRE(twoSet.size() > 100u);

  cv::FeatureMatch m;
  m.inputs.threshold.value = 0.08f; // same values OrbFeatures ran with
  m.inputs.max_features.value = 512;
  m.inputs.ratio.value = 0.7f;
  feed(m.inputs.image, i0);
  m();
  feed(m.inputs.image, i1);
  m();
  const auto texture = m.outputs.matches.value;

  REQUIRE(texture.size() == twoSet.size());
  for(std::size_t i = 0; i < twoSet.size(); ++i)
  {
    INFO("match " << i);
    CHECK(texture[i].prev.x == twoSet[i].prev.x);
    CHECK(texture[i].prev.y == twoSet[i].prev.y);
    CHECK(texture[i].cur.x == twoSet[i].cur.x);
    CHECK(texture[i].cur.y == twoSet[i].cur.y);
    CHECK(texture[i].distance == twoSet[i].distance);
    CHECK(texture[i].prev_size == twoSet[i].prev_size);
    CHECK(texture[i].cur_size == twoSet[i].cur_size);
    CHECK(texture[i].prev_octave == twoSet[i].prev_octave);
    CHECK(texture[i].cur_octave == twoSet[i].cur_octave);
  }

  // The uniqueness guarantee itself, on both paths: no `prev` position may appear twice.
  auto duplicates = [](const std::vector<cv::feature_match>& ms, bool prev) {
    std::vector<std::pair<float, float>> k;
    k.reserve(ms.size());
    for(const auto& x : ms)
      k.emplace_back(prev ? x.prev.x : x.cur.x, prev ? x.prev.y : x.cur.y);
    std::sort(k.begin(), k.end());
    return static_cast<int>(
        k.size() - static_cast<std::size_t>(std::unique(k.begin(), k.end()) - k.begin()));
  };
  CHECK(duplicates(twoSet, true) == 0);
  CHECK(duplicates(texture, true) == 0);

  // ...and the asymmetry is REAL on this input, so the two checks above are not vacuous: the
  // train side does contain a keypoint claimed by more than one query. (Were the direction
  // reversed, this duplicate would land on `prev` instead, and the checks above would fail.)
  CHECK(duplicates(twoSet, false) > 0);
  CHECK(duplicates(texture, false) > 0);
}

// ================================================================== Lowe's ratio behaviour
TEST_CASE("Lowe ratio rejects an ambiguous pair at 0.7 and accepts it when loosened",
          "[match2]")
{
  // Query = all-zero descriptor. Train[0] has 10 bits set, Train[1] has 12 *different* bits
  // set, so the Hamming distances are exactly 10 and 12 (derivation 3).
  const std::vector<cv::keypoint> query{mk(0.5f, 0.5f, cv_support::descriptor{})};
  const std::vector<cv::keypoint> train{
      mk(0.1f, 0.1f, descBits({{0, 10}})), mk(0.9f, 0.9f, descBits({{10, 12}}))};

  // Sanity: the distances really are 10 and 12.
  REQUIRE(cv_support::hamming(query[0].desc, train[0].desc) == 10);
  REQUIRE(cv_support::hamming(query[0].desc, train[1].desc) == 12);

  // 10 < 0.7 * 12 == 8.4  ->  false  ->  rejected (this is cv.jit's setting)
  CHECK(matchSets(query, train, 0.7f).empty());
  // 10 < 0.8 * 12 == 9.6  ->  false  ->  still rejected (the PREVIOUS port default)
  CHECK(matchSets(query, train, 0.8f).empty());
  // 10 < 0.9 * 12 == 10.8 ->  true   ->  accepted
  auto loose = matchSets(query, train, 0.9f);
  REQUIRE(loose.size() == 1u);
  CHECK(loose[0].distance == Approx(10.f));
  CHECK(loose[0].cur.x == Approx(0.1f)); // the nearer of the two, i.e. train[0]

  // The rejection above is caused by AMBIGUITY, not by a broken threshold: an unambiguous
  // pair (distances 2 and 100) sails through at the very same 0.7.
  const std::vector<cv::keypoint> clear{
      mk(0.1f, 0.1f, descBits({{0, 2}})), mk(0.9f, 0.9f, descBits({{10, 100}}))};
  REQUIRE(cv_support::hamming(query[0].desc, clear[0].desc) == 2);
  REQUIRE(cv_support::hamming(query[0].desc, clear[1].desc) == 100);
  auto ok = matchSets(query, clear, 0.7f);
  REQUIRE(ok.size() == 1u);
  CHECK(ok[0].distance == Approx(2.f));
}

// ============================================================================ scale space
TEST_CASE("OrbFeatures with Octaves == 1 reproduces the single-scale detection exactly",
          "[match2][octaves]")
{
  // The octave-0 subset of a multi-octave run must be BIT-IDENTICAL to a single-scale run:
  // same count, same positions, same descriptors. This is what keeps the pre-pyramid
  // behaviour (and tests/test_features.cpp) intact.
  Image img = upscale2x(makeTexture(64, 64, 0, 0)); // 128x128 -> the pyramid has >1 level
  auto single = detect(img, 1, 4096);
  auto multi = detect(img, 4, 4096);

  REQUIRE(!single.empty());
  REQUIRE(multi.size() > single.size()); // extra octaves really did contribute

  std::vector<cv::keypoint> octave0;
  for(auto& k : multi)
    if(k.octave == 0)
      octave0.push_back(k);

  REQUIRE(octave0.size() == single.size());
  for(std::size_t i = 0; i < single.size(); ++i)
  {
    CHECK(octave0[i].position.x == single[i].position.x);
    CHECK(octave0[i].position.y == single[i].position.y);
    CHECK(octave0[i].angle == single[i].angle);
    CHECK(octave0[i].response == single[i].response);
    CHECK(octave0[i].desc == single[i].desc);
    CHECK(octave0[i].size == Approx(expectedSize(128, 128)));
  }
}

TEST_CASE("OrbFeatures reports plausible octave and size fields", "[match2][octaves]")
{
  Image img = upscale2x(makeTexture(64, 64, 0, 0)); // 128x128: levels 128, 64 (32 too small)
  auto kps = detect(img, 4, 4096);
  REQUIRE(!kps.empty());

  int perOctave[8]{};
  for(auto& k : kps)
  {
    REQUIRE(k.octave >= 0);
    REQUIRE(k.octave < 8);
    ++perOctave[k.octave];
    // size == 31 * 2^octave by construction (derivation 4)
    CHECK(k.size == Approx(expectedSize(128, 128, k.octave)));
    CHECK(k.response > 0.f);
    // Reported in BASE-image normalised coordinates whatever octave they came from, and far
    // enough from the border that the steered BRIEF pattern stays in bounds *at its level*.
    CHECK(k.position.x >= 0.f);
    CHECK(k.position.x <= 1.f);
    CHECK(k.position.y >= 0.f);
    CHECK(k.position.y <= 1.f);
  }
  // 128x128 -> levels 128 and 64; 32 is below the 2*border+1 minimum, so nothing above 1.
  CHECK(perOctave[0] > 0);
  CHECK(perOctave[1] > 0);
  CHECK(perOctave[2] == 0);
  CHECK(perOctave[3] == 0);
}

TEST_CASE("Octaves buys scale invariance: a 2x-scaled pattern matches only with a pyramid",
          "[match2][octaves]")
{
  Image base = makeTexture(64, 64, 0, 0);
  Image scaled = upscale2x(base); // the SAME pattern, exactly twice the size

  auto A = detect(base, 1, 4096); // the reference/template set, single scale
  REQUIRE(!A.empty());

  // ---- direction 1: WITH a scale space, every reference feature is found again ----------
  for(int octaves : {3, 4})
  {
    INFO("octaves = " << octaves);
    auto B = detect(scaled, octaves, 4096);
    auto matches = matchSets(A, B, 0.7f);

    // Derivation 2: level 1 of the upscaled image IS the base image, bit for bit, so every
    // reference keypoint has an exact twin.
    CHECK(matches.size() == A.size());

    std::vector<float> residX, residY;
    for(auto& m : matches)
    {
      // Descriptors are identical, so the Hamming distance is exactly zero.
      CHECK(m.distance == Approx(0.f));
      // The match must come from the pyramid, not from the blocky full-resolution level.
      CHECK(m.prev_octave == 0);
      CHECK(m.cur_octave == 1);
      // 31 * 2^1: size really is scale-dependent. Reported in the active coordinate mode --
      // cv.jit scales SIZE by max(scale_x, scale_y), = 1/128 on this square 128x128 image.
      CHECK(m.cur_size == Approx(expectedSize(128, 128, 1)));
      // posB_px == 2 * posA_px + 0.5 (exact).
      const float ax = m.prev.x * 64.f, ay = m.prev.y * 64.f;
      const float bx = m.cur.x * 128.f, by = m.cur.y * 128.f;
      residX.push_back(bx - (2.f * ax + 0.5f));
      residY.push_back(by - (2.f * ay + 0.5f));
    }
    CHECK(median(residX) == Approx(0.f).margin(1e-3));
    CHECK(median(residY) == Approx(0.f).margin(1e-3));
    CHECK(*std::max_element(residX.begin(), residX.end()) == Approx(0.f).margin(1e-3));
    CHECK(*std::max_element(residY.begin(), residY.end()) == Approx(0.f).margin(1e-3));
  }

  // ---- direction 2: WITHOUT a scale space it demonstrably fails ------------------------
  // Same images, same matcher, same ratio; only Octaves changes. A textbook single-scale ORB
  // finds NOTHING here — that asymmetry is the whole point of the octaves attribute.
  auto B1 = detect(scaled, 1, 4096);
  REQUIRE(!B1.empty()); // it does detect plenty of corners — they are just the wrong scale
  auto none = matchSets(A, B1, 0.7f);
  CHECK(none.empty());
  // Even wide open, nothing scale-consistent turns up. A correspondence is scale-consistent
  // only if it satisfies posB == 2*posA + 0.5 on BOTH axes; checking x alone (as this did
  // originally) accepts a random pair with probability ~2/128, so a single coincidence in a
  // few dozen wide-open matches was expected noise rather than a signal. Requiring both axes
  // is the correct predicate and squares that probability.
  auto loose = matchSets(A, B1, 0.95f);
  int consistent = 0;
  for(auto& m : loose)
  {
    const float ax = m.prev.x * 64.f, ay = m.prev.y * 64.f;
    const float bx = m.cur.x * 128.f, by = m.cur.y * 128.f;
    if(std::abs(bx - (2.f * ax + 0.5f)) < 1.f && std::abs(by - (2.f * ay + 0.5f)) < 1.f)
      ++consistent;
  }
  CHECK(consistent == 0);
}

// ============================================== general (non-power-of-2) scale invariance
//
// The test above proves only that the pyramid inverts its own downsampler: `upscale2x` is the
// exact right-inverse of the 2x2 box average, so level 1 of the upscaled image IS the base
// image bit for bit and a Hamming distance of 0 is a tautology. It says nothing about a scale
// change the pyramid does not land on.
//
// This one measures a real one. Base scene: `makeSmooth`, band-limited at ~8 px so it survives
// resampling. Two factors, both applied with the SAME bilinear resampler, so the only
// difference between them is the factor itself:
//
//   * 2.0x — the pyramid's own step. Expected: near-total recall.
//   * 1.4x — the WORST CASE for a halving pyramid. Levels are a factor 2 apart, so a feature
//     whose true scale is 1.4x the base is sqrt(2) = 1.414 away from the nearest level in
//     either direction (1.4/1 = 1.40, and 1/(1.4*0.5) = 1.43). BRIEF's sampling pattern has a
//     fixed radius in level pixels and is not itself scale-invariant, so a 1.4x scale error
//     between the two patches randomises the descriptor.
//
// MEASURED (this build):
//   detector repeatability   1.4x: 2576/2676 = 96.3%   2.0x: 2674/2676 = 99.9%
//   matched AND geometrically correct, at cv.jit's ratio 0.7:
//                            1.4x:    3/2676 =  0.11%  2.0x: 2648/2676 = 98.9%
//
// So this is NOT a detector failure — FAST finds essentially all the same corners — it is a
// DESCRIPTOR failure caused by the octave step being 2.0. It is a real limitation of this
// port, recorded here rather than hidden: OpenCV's ORB uses scaleFactor = 1.2 (worst-case
// scale error sqrt(1.2) = 1.095), which is what would have to change. Fixing that means
// replacing the halving pyramid with a geometric one plus a proper resampler.
//
// The 1.4x assertion below is therefore an UPPER bound that documents the collapse. When the
// pyramid gains a finer scale factor this test MUST FAIL, and the fix is to invert it into a
// recall floor — not to relax the bound.
TEST_CASE("Scale invariance holds at the pyramid's factor of 2 and collapses at 1.4x",
          "[match2][octaves][scale]")
{
  const int W = 128, H = 128;
  Image base = makeSmooth(W, H, 8);
  auto A = detect(base, 4, 4096, 0.04f);
  REQUIRE(A.size() > 1000u);
  REQUIRE(A.size() < 4096u); // not truncated by the cap, so `recall / A.size()` is meaningful

  // Number of A keypoints that have SOME B keypoint at the geometrically predicted place.
  auto repeatability = [&](const std::vector<cv::keypoint>& B, const Image& img, float f) {
    int n = 0;
    for(const auto& a : A)
    {
      const float ex = (a.position.x * W + 0.5f) * f - 0.5f;
      const float ey = (a.position.y * H + 0.5f) * f - 0.5f;
      for(const auto& b : B)
        if(std::abs(b.position.x * img.width - ex) < 2.5f
           && std::abs(b.position.y * img.height - ey) < 2.5f)
        {
          ++n;
          break;
        }
    }
    return n;
  };
  // Number of accepted matches that are ALSO geometrically correct.
  auto correctMatches
      = [&](const std::vector<cv::feature_match>& ms, const Image& img, float f) {
          int n = 0;
          for(const auto& m : ms)
          {
            const float ex = (m.prev.x * W + 0.5f) * f - 0.5f;
            const float ey = (m.prev.y * H + 0.5f) * f - 0.5f;
            if(std::abs(m.cur.x * img.width - ex) < 2.5f
               && std::abs(m.cur.y * img.height - ey) < 2.5f)
              ++n;
          }
          return n;
        };

  const int n = static_cast<int>(A.size());

  SECTION("2.0x — the pyramid's own step: near-total recall")
  {
    Image img = rescale(base, 2.f);
    auto B = detect(img, 4, 4096, 0.04f);
    CHECK(repeatability(B, img, 2.f) >= n * 9 / 10);
    CHECK(correctMatches(matchSets(A, B, 0.7f), img, 2.f) >= n * 9 / 10);
  }

  SECTION("1.4x — worst case for a halving pyramid: detected, but not described")
  {
    Image img = rescale(base, 1.4f);
    auto B = detect(img, 4, 4096, 0.04f);
    // The corners ARE found again: the detector is scale-robust here.
    CHECK(repeatability(B, img, 1.4f) >= n * 9 / 10);
    // ...but almost nothing survives descriptor matching. See the block comment above: this
    // bound documents a known limitation of the factor-2 pyramid. Do NOT relax it.
    CHECK(correctMatches(matchSets(A, B, 0.7f), img, 1.4f) <= n / 50);
  }
}

TEST_CASE("OrbFeatures handles tiny images and out-of-range octaves safely", "[match2][octaves]")
{
  // Below the border the pyramid must simply stop; nothing may be detected, nothing may be
  // read out of bounds (this whole suite runs under ASan/UBSan).
  for(int side : {1, 2, 8, 41, 42})
  {
    Image img = makeTexture(side, side, 0, 0);
    for(int oct : {0, 1, 4, 8})
    {
      auto kps = detect(img, oct, 4096);
      for(auto& k : kps)
      {
        CHECK(k.octave >= 0);
        CHECK(k.position.x >= 0.f);
        CHECK(k.position.x <= 1.f);
      }
    }
  }
  // Non-square, odd dimensions: the box downsample must not read the dangling last row/col.
  Image odd = makeTexture(101, 87, 0, 0);
  auto kps = detect(odd, 4, 4096);
  // `size` is reported in the active coordinate mode, not in base-image pixels: cv.jit scales
  // SIZE by max(scale_x, scale_y), and the default mode here is Normalized, so
  // scale = max(1/W, 1/H) = 1/min(W,H) = 1/87. The base patch diameter is 31 px at octave 0,
  // doubling per octave.
  for(auto& k : kps)
    CHECK(k.size == Approx(expectedSize(101, 87, k.octave)));
}
