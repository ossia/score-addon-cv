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
  cv::FastCorners obj;
  obj.inputs.threshold.value = 0.05f;
  obj.inputs.suppress.value = false;
  obj.inputs.max_corners.value = 3;
  Image img = makeDots(64, 64);
  feed(obj.inputs.image, img);
  obj();
  CHECK(obj.outputs.count.value <= 3);
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
  CHECK(medx == Approx(2.f / 96.f).margin(0.02));
  CHECK(medy == Approx(1.f / 96.f).margin(0.02));
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
    CHECK(dxs[dxs.size() / 2] == Approx(3.f / 96.f).margin(0.02));
    CHECK(dys[dys.size() / 2] == Approx(2.f / 96.f).margin(0.02));
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
    CHECK(dxs[dxs.size() / 2] == Approx(5.f / 96.f).margin(0.02));
    CHECK(dys[dys.size() / 2] == Approx(4.f / 96.f).margin(0.02));
  }
}
