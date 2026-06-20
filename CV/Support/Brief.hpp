#pragma once

/* score-addon-cv — ORB-style oriented FAST + rotated BRIEF helpers (OpenCV-free).
 *
 * Self-contained, header-only. Used by OrbFeatures and FeatureMatch.
 *  - FAST-9 corner detection on a float grayscale buffer (reusing the FastCorners approach).
 *  - Intensity-centroid orientation angle per corner (the "oriented FAST" of ORB).
 *  - A 256-bit rotated-BRIEF descriptor sampled from a fixed test pattern, steered by the
 *    corner orientation, packed as std::array<uint32_t, 8>.
 *  - Brute-force Hamming matching (std::popcount over the 8 words) with Lowe's ratio test.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cv_support
{

// 256-bit descriptor packed into 8 x 32-bit words.
using descriptor = std::array<std::uint32_t, 8>;

// Hamming distance between two 256-bit descriptors.
[[nodiscard]] inline int hamming(const descriptor& a, const descriptor& b) noexcept
{
  int d = 0;
  for(int i = 0; i < 8; ++i)
    d += std::popcount(a[i] ^ b[i]);
  return d;
}

// A detected & described keypoint, in *pixel* coordinates (caller normalises for ports).
struct kp
{
  float x{}, y{};
  float angle{}; // radians
  descriptor desc{};
};

// Bresenham circle of radius 3 (the 16 FAST offsets), clockwise. Matches FastCorners.cpp.
inline constexpr std::array<std::pair<int, int>, 16> kCircle{
    {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
     {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}}};

inline constexpr int kFastN = 9; // contiguous arc length for FAST-9

// The BRIEF test pattern: 256 pairs of (x,y) offsets within a patch. Generated
// deterministically from an isotropic Gaussian-like spread inside a radius-15 patch so the
// header stays free of a giant literal table while remaining stable across builds.
struct BriefPattern
{
  std::array<std::array<std::int8_t, 4>, 256> tests; // {x1,y1,x2,y2}

  constexpr BriefPattern()
      : tests{}
  {
    // Simple deterministic LCG; values land in [-13, 13] to keep the patch radius small
    // enough that we can sample after rotation without leaving a guarded border.
    std::uint32_t s = 0x9e3779b9u;
    auto next = [&s]() -> int {
      s = s * 1664525u + 1013904223u;
      // map to [-13, 13]
      return static_cast<int>((s >> 8) % 27u) - 13;
    };
    for(int i = 0; i < 256; ++i)
    {
      tests[static_cast<std::size_t>(i)][0] = static_cast<std::int8_t>(next());
      tests[static_cast<std::size_t>(i)][1] = static_cast<std::int8_t>(next());
      tests[static_cast<std::size_t>(i)][2] = static_cast<std::int8_t>(next());
      tests[static_cast<std::size_t>(i)][3] = static_cast<std::int8_t>(next());
    }
  }
};

inline constexpr BriefPattern kBrief{};

// Patch half-size that bounds every sample after rotation. The largest unrotated BRIEF
// offset component is 13, but rotating a point (13,13) keeps its *magnitude* — up to
// 13*sqrt(2) ~= 18.39 — and that magnitude can land entirely on one axis after steering. So a
// sample can sit up to ceil(13*sqrt(2)) = 19 pixels from the centre on either axis. We must
// guard keypoints by at least that, otherwise describe()'s clamp would actually fire on
// emitted keypoints and corrupt their descriptors near the border. kPatchRadius therefore
// bounds *every* rotated sample, making the clamp in describe()/orientation() purely
// defensive (it never triggers for an emitted keypoint).
inline constexpr int kPatchRadius = 19; // ceil(13 * sqrt(2))

// Intensity-centroid orientation for a corner at (cx,cy) in a float grayscale image.
// gray: row-major width*height, values in [0,1]. Returns angle in radians.
[[nodiscard]] inline float orientation(
    const float* gray, int W, int H, int cx, int cy, int radius = 12) noexcept
{
  double m01 = 0.0, m10 = 0.0;
  const int r2 = radius * radius;
  for(int dy = -radius; dy <= radius; ++dy)
  {
    int yy = cy + dy;
    yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy); // clamp to border
    for(int dx = -radius; dx <= radius; ++dx)
    {
      if(dx * dx + dy * dy > r2)
        continue;
      int xx = cx + dx;
      xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
      const float v = gray[static_cast<std::size_t>(yy) * W + xx];
      m10 += static_cast<double>(dx) * v;
      m01 += static_cast<double>(dy) * v;
    }
  }
  return static_cast<float>(std::atan2(m01, m10));
}

// Compute the rotated-BRIEF descriptor for a corner at (cx,cy) with the given angle.
// Steering the patch can push a sample up to 13*sqrt(2) ~= 18.39 px from the centre; the
// caller's border guard (kPatchRadius = ceil(13*sqrt(2)) = 19) already keeps every emitted
// keypoint that far from the edge, so the clamp below NEVER fires for an emitted keypoint and
// is purely defensive (e.g. if describe() is ever called directly on a near-edge point).
[[nodiscard]] inline descriptor describe(
    const float* gray, int W, int H, int cx, int cy, float angle) noexcept
{
  const float ca = std::cos(angle);
  const float sa = std::sin(angle);
  auto sample = [&](float ox, float oy) -> float {
    int px = cx + static_cast<int>(std::lround(ox));
    int py = cy + static_cast<int>(std::lround(oy));
    px = px < 0 ? 0 : (px >= W ? W - 1 : px);
    py = py < 0 ? 0 : (py >= H ? H - 1 : py);
    return gray[static_cast<std::size_t>(py) * W + px];
  };
  descriptor d{};
  for(int i = 0; i < 256; ++i)
  {
    const auto& t = kBrief.tests[static_cast<std::size_t>(i)];
    // Rotate both sample points by the corner orientation (steered BRIEF).
    const float i1 = sample(t[0] * ca - t[1] * sa, t[0] * sa + t[1] * ca);
    const float i2 = sample(t[2] * ca - t[3] * sa, t[2] * sa + t[3] * ca);

    if(i1 < i2)
      d[static_cast<std::size_t>(i >> 5)] |= (1u << (i & 31));
  }
  return d;
}

// FAST-9 segment test at (x,y). Returns the corner score (sum of abs diffs over the arc) or
// 0 if not a corner. gray row-major, t = threshold in [0,1].
[[nodiscard]] inline float fastScore(
    const float* gray, int W, int x, int y, float t) noexcept
{
  auto at = [&](int xx, int yy) -> float {
    return gray[static_cast<std::size_t>(yy) * W + xx];
  };
  const float Ip = at(x, y);
  const float hi = Ip + t;
  const float lo = Ip - t;

  int brighter = 0, darker = 0;
  for(int k = 0; k < 16; k += 4)
  {
    const float v = at(x + kCircle[k].first, y + kCircle[k].second);
    if(v > hi)
      ++brighter;
    else if(v < lo)
      ++darker;
  }
  if(brighter < 3 && darker < 3)
    return 0.f;

  std::array<int, 16> cls;
  for(int k = 0; k < 16; ++k)
  {
    const float v = at(x + kCircle[k].first, y + kCircle[k].second);
    cls[static_cast<std::size_t>(k)] = (v > hi) ? 1 : (v < lo) ? -1 : 0;
  }

  auto hasArc = [&](int want) {
    int run = 0, best = 0;
    for(int k = 0; k < 16 + kFastN; ++k)
    {
      if(cls[static_cast<std::size_t>(k % 16)] == want)
      {
        ++run;
        best = std::max(best, run);
      }
      else
        run = 0;
    }
    return best >= kFastN;
  };

  if(!hasArc(1) && !hasArc(-1))
    return 0.f;

  float score = 0.f;
  for(int k = 0; k < 16; ++k)
    score += std::abs(at(x + kCircle[k].first, y + kCircle[k].second) - Ip);
  return score;
}

// Detect oriented FAST corners and compute rotated-BRIEF descriptors over a whole frame.
// rgba: tightly-packed RGBA8, W*H. Fills `out` (cleared first), capped at maxFeatures, kept
// to the strongest by score. Returns nothing; `out` holds pixel-space keypoints.
inline void detectAndDescribe(
    const std::uint8_t* rgba, int W, int H, float threshold, int maxFeatures,
    std::vector<float>& grayScratch, std::vector<kp>& out)
{
  const std::size_t N = static_cast<std::size_t>(W) * H;
  grayScratch.resize(N);
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t* p = rgba + i * 4;
    grayScratch[i]
        = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }
  const float* gray = grayScratch.data();

  // Score every pixel, keep candidates well inside the patch border so BRIEF never samples
  // outside the image.
  struct Cand
  {
    int x, y;
    float score;
  };
  std::vector<Cand> cands;
  const int border = kPatchRadius + 1;
  for(int y = border; y < H - border; ++y)
  {
    for(int x = border; x < W - border; ++x)
    {
      const float s = fastScore(gray, W, x, y, threshold);
      if(s > 0.f)
        cands.push_back({x, y, s});
    }
  }

  // Keep the strongest maxFeatures corners.
  if(maxFeatures > 0 && static_cast<int>(cands.size()) > maxFeatures)
  {
    std::nth_element(
        cands.begin(), cands.begin() + maxFeatures, cands.end(),
        [](const Cand& a, const Cand& b) { return a.score > b.score; });
    cands.resize(static_cast<std::size_t>(maxFeatures));
  }

  out.clear();
  out.reserve(cands.size());
  for(const auto& c : cands)
  {
    const float angle = orientation(gray, W, H, c.x, c.y);
    kp k;
    k.x = static_cast<float>(c.x);
    k.y = static_cast<float>(c.y);
    k.angle = angle;
    k.desc = describe(gray, W, H, c.x, c.y, angle);
    out.push_back(k);
  }
}

// Lowe's ratio test brute-force match. For each `cur` descriptor, find the two nearest in
// `prev`; accept if best < ratio * secondBest. Returns indices (curIdx, prevIdx, dist).
struct match_result
{
  int cur, prev, dist;
};

inline void matchRatio(
    const std::vector<kp>& cur, const std::vector<kp>& prev, float ratio,
    std::vector<match_result>& out)
{
  out.clear();
  if(prev.size() < 2)
    return;
  for(int i = 0; i < static_cast<int>(cur.size()); ++i)
  {
    int best = 1 << 30, second = 1 << 30, bestJ = -1;
    for(int j = 0; j < static_cast<int>(prev.size()); ++j)
    {
      const int d = hamming(cur[static_cast<std::size_t>(i)].desc,
                            prev[static_cast<std::size_t>(j)].desc);
      if(d < best)
      {
        second = best;
        best = d;
        bestJ = j;
      }
      else if(d < second)
      {
        second = d;
      }
    }
    if(bestJ >= 0 && static_cast<float>(best) < ratio * static_cast<float>(second))
      out.push_back({i, bestJ, best});
  }
}

} // namespace cv_support
