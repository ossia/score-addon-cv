// Tests for feature objects: FastCorners, OrbFeatures, FeatureMatch.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/FastCorners.hpp>
#include <CV/Cpu/OrbFeatures.hpp>
#include <CV/Cpu/FeatureMatch.hpp>

#include <algorithm>
#include <bit>
#include <vector>
#include <cmath>

using Catch::Approx;
using namespace cvtest;

namespace
{
// A scene with several distinct bright corner-like dots on a dark field — gives FAST a set
// of well-isolated corners (a single bright pixel surrounded by dark passes the segment test).
Image makeDots(int W, int H)
{
  Image img(W, H, 0);
  const int pts[][2]
      = {{10, 10}, {30, 12}, {18, 28}, {40, 35}, {25, 45}, {48, 20}, {12, 40}};
  for(auto& p : pts)
  {
    // a small 2x2 bright blob makes a stable corner
    img.setGray(p[0], p[1], 255);
    img.setGray(p[0] + 1, p[1], 255);
    img.setGray(p[0], p[1] + 1, 255);
    img.setGray(p[0] + 1, p[1] + 1, 255);
  }
  return img;
}
}

// --------------------------------------------------------------------------- FastCorners
TEST_CASE("FastCorners finds corners on a dotted scene", "[fast]")
{
  cv::FastCorners obj;
  obj.inputs.threshold.value = 0.1f;
  obj.inputs.suppress.value = true;
  obj.inputs.max_corners.value = 1024;

  Image img = makeDots(64, 64);
  feed(obj.inputs.image, img);
  obj();

  CHECK(obj.outputs.count.value > 0);
  CHECK(obj.outputs.count.value == static_cast<int>(obj.outputs.corners.value.size()));
  // All reported corners are in normalised [0,1].
  for(auto& c : obj.outputs.corners.value)
  {
    CHECK(c.position.x >= 0.f);
    CHECK(c.position.x <= 1.f);
    CHECK(c.score > 0.f);
  }
}

TEST_CASE("FastCorners finds nothing on a flat image", "[fast]")
{
  cv::FastCorners obj;
  obj.inputs.threshold.value = 0.1f;
  Image flat(48, 48, 128);
  feed(obj.inputs.image, flat);
  obj();
  CHECK(obj.outputs.count.value == 0);
}

TEST_CASE("FastCorners respects the max-corners cap", "[fast]")
{
  // `count <= 3` alone is satisfied by a detector that finds NOTHING, so the uncapped run is
  // asserted first: this scene yields far more than 3 corners, hence the cap must BIND and
  // the capped count is exactly the cap, not merely below it.
  auto run = [](int cap) {
    cv::FastCorners obj;
    obj.inputs.threshold.value = 0.05f;
    obj.inputs.suppress.value = false;
    obj.inputs.max_corners.value = cap;
    Image img = makeDots(64, 64);
    feed(obj.inputs.image, img);
    obj();
    return obj.outputs.count.value;
  };

  const int uncapped = run(8192);
  REQUIRE(uncapped > 3); // the cap has something to cut
  CHECK(run(3) == 3);
  CHECK(run(1) == 1);
  CHECK(run(uncapped) == uncapped); // a cap at/above the true count changes nothing
}

// ------------------------------------------------------ FAST-9 arc phase coverage
// THE regression test for the FAST-12 quick-rejection bug.
//
// FAST-9 fires when 9 CONTIGUOUS samples of the 16-point Bresenham circle are all brighter
// than Ip+t (or all darker than Ip-t). There are exactly 16 rotational phases such an arc can
// have, and the detector must accept all 16 -- FAST is meant to be rotation-covariant.
//
// The implementation used to quick-reject with "at least 3 of the 4 compass points 0/4/8/12",
// which is the high-speed test for FAST-*12*. A 9-long arc starting at phase p covers indices
// p..p+8; it contains three multiples of 4 only when p is itself a multiple of 4, and two
// otherwise. So exactly 4 of the 16 phases passed and the other 12 were thrown away before
// the arc test ever ran. The pair test (k, k+8) that replaced it is valid for N = 9 because
// two indices 8 apart cannot both fall in the 7-wide complement of a 9-arc. Measured effect
// on a 256x256 frame at threshold 0.08: 6.0% more corners on hash noise, 29% more on
// band-limited noise with 4 px structure, 66% more with 8 px structure.
//
// The scenes below are built so a *textbook* FAST-9 must answer yes/no unambiguously: the
// background, the centre and the 7 (resp. 8) non-arc circle samples are all exactly Ip, hence
// classified "similar", so the longest run is exactly the arc length by construction.
TEST_CASE("FAST-9 accepts a 9-pixel arc at every one of the 16 phases", "[fast][arc]")
{
  constexpr int S = 32;      // image side
  constexpr int C = 16;      // centre
  constexpr float mid = 0.5f;
  constexpr float t = 0.1f;  // |1.0 - 0.5| and |0.0 - 0.5| are both 0.5 > t

  // Paint `len` contiguous circle samples starting at `phase` with `arcVal`, everything else
  // (including the centre) at `mid`.
  auto makeArc = [&](int phase, int len, float arcVal) {
    std::vector<float> g(static_cast<std::size_t>(S) * S, mid);
    for(int i = 0; i < len; ++i)
    {
      const auto& o = cv_support::kCircle[static_cast<std::size_t>((phase + i) % 16)];
      g[static_cast<std::size_t>(C + o.second) * S + (C + o.first)] = arcVal;
    }
    return g;
  };

  SECTION("segment test: all 16 phases, both polarities")
  {
    for(int phase = 0; phase < 16; ++phase)
    {
      INFO("phase = " << phase);
      // A 9-arc IS a corner, brighter or darker. (Pre-fix: only phases 0, 4, 8, 12 passed.)
      CHECK(cv_support::fastScore(makeArc(phase, 9, 1.f).data(), S, C, C, t) > 0.f);
      CHECK(cv_support::fastScore(makeArc(phase, 9, 0.f).data(), S, C, C, t) > 0.f);
      // An 8-arc is NOT: this pins the test to N == 9 rather than "accepts anything".
      CHECK(cv_support::fastScore(makeArc(phase, 8, 1.f).data(), S, C, C, t) == 0.f);
      CHECK(cv_support::fastScore(makeArc(phase, 8, 0.f).data(), S, C, C, t) == 0.f);
    }
  }

  SECTION("FastCorners object reports the centre at all 16 phases")
  {
    for(int phase = 0; phase < 16; ++phase)
    {
      INFO("phase = " << phase);
      const auto g = makeArc(phase, 9, 1.f);
      Image img(S, S, 0);
      for(int y = 0; y < S; ++y)
        for(int x = 0; x < S; ++x)
          img.setGray(
              x, y,
              static_cast<std::uint8_t>(
                  std::lround(g[static_cast<std::size_t>(y) * S + x] * 255.f)));

      cv::FastCorners obj;
      obj.inputs.threshold.value = t;
      obj.inputs.suppress.value = false; // no NMS: we assert on this exact pixel
      obj.inputs.max_corners.value = 8192;
      obj.inputs.min_distance.value = 0.f;
      feed(obj.inputs.image, img);
      obj();

      // FastCorners emits position = pixel / side, so the centre is exactly 16/32 == 0.5.
      const bool found = std::any_of(
          obj.outputs.corners.value.begin(), obj.outputs.corners.value.end(),
          [](const cv::corner_point& c) {
            return c.position.x == 0.5f && c.position.y == 0.5f;
          });
      CHECK(found);
    }
  }
}

namespace
{
// Find the corner nearest a normalised target position; returns its index or -1.
int nearestCorner(
    const std::vector<cv::corner_point>& cs, float tx, float ty, float tol)
{
  int best = -1;
  float bestD = tol * tol;
  for(int i = 0; i < static_cast<int>(cs.size()); ++i)
  {
    const float dx = cs[static_cast<std::size_t>(i)].position.x - tx;
    const float dy = cs[static_cast<std::size_t>(i)].position.y - ty;
    const float d = dx * dx + dy * dy;
    if(d <= bestD)
    {
      bestD = d;
      best = i;
    }
  }
  return best;
}
}

TEST_CASE("FastCorners keeps the STRONGEST corners under a small cap", "[fast]")
{
  // One very strong corner (bright 2x2 blob on black) and several weaker ones (dimmer
  // blobs). With a small cap, the strong corner must survive and a weak one must be dropped.
  // The FAST score is the sum of abs differences around the circle, so a brighter blob scores
  // higher. Place the strong corner at a known location.
  const int W = 64, H = 64;
  Image img(W, H, 0);
  auto blob = [&](int x, int y, std::uint8_t v) {
    img.setGray(x, y, v);
    img.setGray(x + 1, y, v);
    img.setGray(x, y + 1, v);
    img.setGray(x + 1, y + 1, v);
  };
  // Strong corner, far from the others.
  const int strongX = 12, strongY = 12;
  blob(strongX, strongY, 255);
  // Weaker corners (dimmer) — these should be dropped first under a tight cap.
  const int weakX = 50, weakY = 50;
  blob(weakX, weakY, 90);
  blob(30, 50, 80);
  blob(50, 14, 85);
  blob(14, 50, 70);

  cv::FastCorners obj;
  obj.inputs.threshold.value = 0.05f;
  obj.inputs.suppress.value = true;
  obj.inputs.max_corners.value = 1;
  obj.inputs.min_distance.value = 0.f;
  feed(obj.inputs.image, img);
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  // The single kept corner must be the strong one, not a weak one (raster order would have
  // picked the top-left-most surviving corner instead).
  const float sx = (strongX + 0.5f) / W;
  const float sy = (strongY + 0.5f) / H;
  const auto& kept = obj.outputs.corners.value[0];
  CHECK(std::abs(kept.position.x - sx) < 0.05f);
  CHECK(std::abs(kept.position.y - sy) < 0.05f);

  // And the weak corner must NOT be present.
  CHECK(nearestCorner(
            obj.outputs.corners.value, (weakX + 0.5f) / W, (weakY + 0.5f) / H, 0.05f)
        == -1);
}

TEST_CASE("FastCorners min_distance de-clusters corners", "[fast]")
{
  // A tight cluster of equally-strong corners plus a couple far away. With min_distance > 0
  // the close ones collapse to one, so the count drops vs min_distance == 0, and every pair
  // of returned corners is at least min_distance apart.
  const int W = 64, H = 64;
  Image img(W, H, 0);
  auto blob = [&](int x, int y) {
    img.setGray(x, y, 255);
    img.setGray(x + 1, y, 255);
    img.setGray(x, y + 1, 255);
    img.setGray(x + 1, y + 1, 255);
  };
  // Cluster around (20,20), spaced 3px apart (well under any reasonable min_distance).
  for(int gy = 0; gy < 4; ++gy)
    for(int gx = 0; gx < 4; ++gx)
      blob(20 + gx * 3, 20 + gy * 3);
  // Two well-separated extras.
  blob(52, 8);
  blob(8, 52);

  auto run = [&](float md) {
    cv::FastCorners obj;
    obj.inputs.threshold.value = 0.05f;
    obj.inputs.suppress.value = false;
    obj.inputs.max_corners.value = 8192;
    obj.inputs.min_distance.value = md;
    feed(obj.inputs.image, img);
    obj();
    return obj.outputs.corners.value;
  };

  auto none = run(0.f);
  auto spaced = run(0.1f); // 0.1 * 64 = 6.4 px minimum spacing

  REQUIRE(none.size() > spaced.size());

  // Every pair of returned corners must be >= min_distance apart (in normalised units,
  // measured against the same max(W,H) the implementation uses).
  const float minPx = 0.1f * std::max(W, H);
  for(std::size_t i = 0; i < spaced.size(); ++i)
    for(std::size_t j = i + 1; j < spaced.size(); ++j)
    {
      const float dx = (spaced[i].position.x - spaced[j].position.x) * W;
      const float dy = (spaced[i].position.y - spaced[j].position.y) * H;
      CHECK(std::sqrt(dx * dx + dy * dy) >= minPx - 1e-3f);
    }
}

// ---------------------------------------------------------------------------- OrbFeatures
TEST_CASE("OrbFeatures produces keypoints with non-zero descriptors", "[orb]")
{
  cv::OrbFeatures obj;
  obj.inputs.threshold.value = 0.1f;
  obj.inputs.max_features.value = 512;

  Image img = makeDots(64, 64);
  feed(obj.inputs.image, img);
  obj();

  REQUIRE(obj.outputs.count.value > 0);
  for(auto& kp : obj.outputs.keypoints.value)
  {
    CHECK(kp.position.x >= 0.f);
    CHECK(kp.position.x <= 1.f);
    // descriptor must not be all-zero (some bits set).
    std::uint32_t orv = 0;
    for(auto w : kp.desc)
      orv |= w;
    CHECK(orv != 0u);
  }
}

TEST_CASE("OrbFeatures keypoints stay outside the patch border", "[orb]")
{
  // Place corners right at the image edges (1 px from the border). The descriptor steers the
  // BRIEF pattern by up to 13*sqrt(2) px, so any emitted keypoint must be at least
  // kPatchRadius (= 19) px from every edge; otherwise describe() would clamp samples and
  // corrupt the descriptor. Assert no emitted keypoint is within that border.
  const int W = 80, H = 80;
  Image img(W, H, 0);
  // A simple deterministic varying background so interior corners exist too.
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
      img.setGray(x, y, static_cast<std::uint8_t>(((x * 7 + y * 13) * 5) & 0xFF));
  // Add strong corners right next to the edges to tempt the detector into the border zone.
  auto blob = [&](int x, int y) {
    img.setGray(x, y, 255);
    img.setGray(x + 1, y, 255);
    img.setGray(x, y + 1, 255);
    img.setGray(x + 1, y + 1, 255);
  };
  blob(1, 1);
  blob(W - 3, 1);
  blob(1, H - 3);
  blob(W - 3, H - 3);

  cv::OrbFeatures obj;
  obj.inputs.threshold.value = 0.08f;
  obj.inputs.max_features.value = 1024;
  feed(obj.inputs.image, img);
  obj();

  REQUIRE(obj.outputs.count.value > 0);
  const int border = cv_support::kPatchRadius;
  for(auto& kp : obj.outputs.keypoints.value)
  {
    const float px = kp.position.x * W;
    const float py = kp.position.y * H;
    CHECK(px >= static_cast<float>(border));
    CHECK(px <= static_cast<float>(W - border));
    CHECK(py >= static_cast<float>(border));
    CHECK(py <= static_cast<float>(H - border));
    // Descriptors remain non-degenerate (not all-zero, not all-ones garbage from clamping).
    std::uint32_t orv = 0;
    int bits = 0;
    for(auto w : kp.desc)
    {
      orv |= w;
      bits += std::popcount(w);
    }
    CHECK(orv != 0u);
    CHECK(bits < 256); // not fully saturated
  }
}

// ------------------------------------------------------------------ BRIEF pre-smoothing
namespace
{
// Grayscale a test Image exactly as the detector does (Rec.601 / 255).
std::vector<float> toGrayF(const Image& img)
{
  std::vector<float> g(static_cast<std::size_t>(img.width) * img.height);
  for(std::size_t i = 0; i < g.size(); ++i)
  {
    const std::uint8_t* p = &img.px[i * 4];
    g[i] = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }
  return g;
}
}

TEST_CASE("BRIEF samples a SMOOTHED patch, as ORB does", "[orb][brief]")
{
  // BRIEF is defined on a smoothed patch (OpenCV ORB: GaussianBlur 7x7 sigma 2 per level;
  // Calonder's paper: 9x9 sigma ~2). Comparing raw single pixels makes every bit whose two
  // samples are close in value a coin flip under sensor noise, so descriptors of the same
  // physical corner in two consecutive camera frames disagree in a large number of bits and
  // the Lowe ratio test discards the match. Purely synthetic images hide this because they
  // have no noise; these two sections put the noise back.
  const int W = 96, H = 96;
  // Band-limited scene (structure at ~8 px), so blurring does not simply erase the content.
  Image clean(W, H, 0);
  auto cell = [](int x, int y) -> float {
    std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x + 1000)) * 374761u
                      + static_cast<std::uint64_t>(static_cast<std::uint32_t>(y + 1000)) * 668265u;
    h = ((h ^ (h >> 13)) & 0xFFFFFFu) * 12743u;
    return static_cast<float>((h >> 5) & 0xFF);
  };
  auto smoothAt = [&](int x, int y) {
    const float fx = x / 8.f, fy = y / 8.f;
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    float tx = fx - x0, ty = fy - y0;
    tx = tx * tx * (3.f - 2.f * tx);
    ty = ty * ty * (3.f - 2.f * ty);
    return cell(x0, y0) * (1 - tx) * (1 - ty) + cell(x0 + 1, y0) * tx * (1 - ty)
           + cell(x0, y0 + 1) * (1 - tx) * ty + cell(x0 + 1, y0 + 1) * tx * ty;
  };
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
      clean.setGray(x, y, static_cast<std::uint8_t>(std::lround(smoothAt(x, y))));

  SECTION("the emitted descriptor is the one computed on the blurred level, not the raw one")
  {
    // This is the wiring assertion: recompute a keypoint's descriptor both ways and check
    // which one the object actually shipped. Removing the pre-filter flips the result.
    cv::OrbFeatures obj;
    obj.inputs.threshold.value = 0.04f;
    obj.inputs.max_features.value = 64;
    obj.inputs.octaves.value = 1; // single scale => level 0 == the base image
    feed(obj.inputs.image, clean);
    obj();
    REQUIRE(obj.outputs.count.value > 0);

    const auto gray = toGrayF(clean);
    std::vector<float> blurred, tmp;
    cv_support::blurForBrief(gray.data(), W, H, blurred, tmp);

    int matchesBlurred = 0, matchesRaw = 0;
    for(const auto& kp : obj.outputs.keypoints.value)
    {
      // Normalized mode at octave 0: position == pixel / W, exactly.
      const int px = static_cast<int>(std::lround(kp.position.x * W));
      const int py = static_cast<int>(std::lround(kp.position.y * H));
      if(kp.desc == cv_support::describe(blurred.data(), W, H, px, py, kp.angle))
        ++matchesBlurred;
      if(kp.desc == cv_support::describe(gray.data(), W, H, px, py, kp.angle))
        ++matchesRaw;
    }
    CHECK(matchesBlurred == obj.outputs.count.value);
    CHECK(matchesRaw == 0); // and it is genuinely a different descriptor
  }

  SECTION("smoothing makes the descriptor robust to +/-2 LSB of sensor noise")
  {
    // Deterministic +/-2/255 dither, i.e. two frames of the same static scene off a real
    // sensor. Compare, at fixed points, the Hamming distance between the clean and noisy
    // descriptors when sampled raw and when sampled from the pre-filtered buffer.
    Image noisy = clean;
    for(int y = 0; y < H; ++y)
      for(int x = 0; x < W; ++x)
      {
        const int n = ((x * 7 + y * 13) % 5) - 2; // -2..+2
        const int v = std::clamp(static_cast<int>(clean.px[(static_cast<std::size_t>(y) * W + x) * 4]) + n, 0, 255);
        noisy.setGray(x, y, static_cast<std::uint8_t>(v));
      }

    const auto gc = toGrayF(clean);
    const auto gn = toGrayF(noisy);
    std::vector<float> bc, bn, tmp;
    cv_support::blurForBrief(gc.data(), W, H, bc, tmp);
    cv_support::blurForBrief(gn.data(), W, H, bn, tmp);

    long rawTotal = 0, blurTotal = 0;
    int n = 0;
    for(int y = 24; y < H - 24; y += 4)
      for(int x = 24; x < W - 24; x += 4)
      {
        const float a = cv_support::orientation(gc.data(), W, H, x, y);
        rawTotal += cv_support::hamming(
            cv_support::describe(gc.data(), W, H, x, y, a),
            cv_support::describe(gn.data(), W, H, x, y, a));
        blurTotal += cv_support::hamming(
            cv_support::describe(bc.data(), W, H, x, y, a),
            cv_support::describe(bn.data(), W, H, x, y, a));
        ++n;
      }
    REQUIRE(n > 0);
    INFO("raw bits flipped/descriptor = " << (double(rawTotal) / n)
         << ", smoothed = " << (double(blurTotal) / n));
    // Fewer bits flip after smoothing, by a wide margin — the whole point of the pre-filter.
    CHECK(blurTotal * 3 < rawTotal);
  }
}

// ---------------------------------------------------------------------------- FeatureMatch
namespace
{
// A deterministic, spatially-varying noisy texture. Unlike identical dots, every feature
// sits in a locally-unique neighbourhood, so BRIEF descriptors are discriminative — the
// realistic case descriptor matching is designed for. `sx,sy` shift the whole pattern.
Image makeTexture(int W, int H, int sx, int sy)
{
  Image img(W, H, 0);
  // 64-bit accumulation keeps every product well within range (this build also sanitizes
  // unsigned overflow), giving a deterministic, spatially-unique value per pixel.
  auto hash = [](int x, int y) -> std::uint8_t {
    std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x + 1000)) * 374761u
                      + static_cast<std::uint64_t>(static_cast<std::uint32_t>(y + 1000)) * 668265u;
    h = ((h ^ (h >> 13)) & 0xFFFFFFu) * 12743u; // masked so the product stays < 2^40
    return static_cast<std::uint8_t>((h >> 5) & 0xFF);
  };
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
      img.setGray(x, y, hash(x - sx, y - sy)); // pattern translates with (sx,sy)
  return img;
}
}

TEST_CASE("FeatureMatch needs two frames; matches a translated textured scene", "[match]")
{
  cv::FeatureMatch obj;
  obj.inputs.threshold.value = 0.08f;
  obj.inputs.max_features.value = 512;
  obj.inputs.ratio.value = 0.8f;

  Image f0 = makeTexture(96, 96, 0, 0);
  feed(obj.inputs.image, f0);
  obj();
  // First frame: no previous descriptors -> no matches.
  CHECK(obj.outputs.count.value == 0);

  // Second frame: the same texture translated by (+2,+1).
  Image f1 = makeTexture(96, 96, 2, 1);
  feed(obj.inputs.image, f1);
  obj();

  REQUIRE(obj.outputs.count.value > 0);
  // Most matches should reflect the true (+2/96, +1/96) displacement. Use the MEDIAN to be
  // robust against the handful of spurious matches any descriptor matcher produces.
  std::vector<float> dxs, dys;
  for(auto& m : obj.outputs.matches.value)
  {
    dxs.push_back(m.cur.x - m.prev.x);
    dys.push_back(m.cur.y - m.prev.y);
  }
  std::sort(dxs.begin(), dxs.end());
  std::sort(dys.begin(), dys.end());
  float medx = dxs[dxs.size() / 2];
  float medy = dys[dys.size() / 2];
  // Tolerance is half a pixel (0.5/96 == 0.0052), not 0.02. FAST reports integer corner
  // positions and the translation is an exact integer shift of the same pattern, so the
  // median displacement is exactly the true one; a 0.02 margin is ~2 px and, for the +1 px
  // y component, spanned zero -- it could not tell a correct match from no displacement.
  const float halfPx = 0.5f / 96.f;
  CHECK(medx == Approx(2.f / 96.f).margin(halfPx));
  CHECK(medy == Approx(1.f / 96.f).margin(halfPx));
}

TEST_CASE("FeatureMatch reference mode matches against a stored template", "[match]")
{
  cv::FeatureMatch obj;
  obj.inputs.threshold.value = 0.08f;
  obj.inputs.max_features.value = 512;
  obj.inputs.ratio.value = 0.8f;
  obj.inputs.set_reference.value = false;

  // Frame 0: capture as the reference on a rising edge of set_reference.
  Image f0 = makeTexture(96, 96, 0, 0);
  feed(obj.inputs.image, f0);
  obj.inputs.set_reference.value = true; // rising edge this frame
  obj();
  // The reference is captured from the current frame; nothing to report yet (no prior frame
  // and the reference is THIS frame).
  obj.inputs.set_reference.value = false; // drop the toggle so it stays a single edge

  // Several unrelated frames pass; in reference mode they must keep matching the REFERENCE,
  // not the immediately preceding frame.
  // Frame A: shifted by (+3,+2) from the reference.
  Image fa = makeTexture(96, 96, 3, 2);
  feed(obj.inputs.image, fa);
  obj();

  REQUIRE(obj.outputs.count.value > 0);
  {
    std::vector<float> dxs, dys;
    for(auto& m : obj.outputs.matches.value)
    {
      dxs.push_back(m.cur.x - m.prev.x);
      dys.push_back(m.cur.y - m.prev.y);
    }
    std::sort(dxs.begin(), dxs.end());
    std::sort(dys.begin(), dys.end());
    // Displacement is measured against the REFERENCE (frame 0), i.e. (+3,+2).
    // Half-pixel tolerance: the shift is an exact integer translation of the same pattern.
    CHECK(dxs[dxs.size() / 2] == Approx(3.f / 96.f).margin(0.5f / 96.f));
    CHECK(dys[dys.size() / 2] == Approx(2.f / 96.f).margin(0.5f / 96.f));
  }

  // Frame B: shifted by (+5,+4) from the reference. Even though the previous frame was the
  // (+3,+2) one, reference mode must still report displacement vs the reference (+5,+4),
  // proving it is NOT temporal frame-to-frame here.
  Image fb = makeTexture(96, 96, 5, 4);
  feed(obj.inputs.image, fb);
  obj();

  REQUIRE(obj.outputs.count.value > 0);
  {
    std::vector<float> dxs, dys;
    for(auto& m : obj.outputs.matches.value)
    {
      dxs.push_back(m.cur.x - m.prev.x);
      dys.push_back(m.cur.y - m.prev.y);
    }
    std::sort(dxs.begin(), dxs.end());
    std::sort(dys.begin(), dys.end());
    CHECK(dxs[dxs.size() / 2] == Approx(5.f / 96.f).margin(0.5f / 96.f));
    CHECK(dys[dys.size() / 2] == Approx(4.f / 96.f).margin(0.5f / 96.f));
  }
}
