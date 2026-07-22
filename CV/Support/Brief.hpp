#pragma once

/* score-addon-cv — ORB-style oriented FAST + rotated BRIEF helpers (OpenCV-free).
 *
 * Self-contained, header-only. Used by OrbFeatures and FeatureMatch.
 *  - FAST-9 corner detection on a float grayscale buffer; CV/Cpu/FastCorners.cpp calls the
 *    very same fastScore(), so the two cannot drift apart.
 *  - A halving image pyramid (octaves) so detection is scale-invariant; keypoints are
 *    reported in BASE-image coordinates with the octave they were found at.
 *  - Intensity-centroid orientation angle per corner (the "oriented FAST" of ORB).
 *  - A 256-bit rotated-BRIEF descriptor sampled from a fixed test pattern, steered by the
 *    corner orientation, packed as std::array<uint32_t, 8>.
 *  - Brute-force Hamming matching (std::popcount over the 8 words) with Lowe's ratio test.
 *
 * DIFFERENCES FROM cv.jit (deliberate, see CONTRIBUTING_AGENTS.md "port vocabulary"):
 *  - cv.jit.keypoints uses OpenCV's BRISK: a 512-bit descriptor over a very different
 *    sampling pattern. Ours is a 256-bit steered BRIEF. The two are binary-incompatible
 *    either way, so no attempt is made to reproduce BRISK bit-for-bit; only the *interface*
 *    (octaves, and the 6 keypoint fields x/y/size/angle/response/octave) is ported.
 *  - cv.jit emits `angle` in DEGREES (OpenCV's cv::KeyPoint convention). This port emits
 *    RADIANS everywhere, matching the rest of score-addon-cv (FastCorners, HoughLines,
 *    CartoPol, ...) and the atan2 that produces it. Converting is a `* 180/pi` downstream.
 *  - cv.jit.keypoints.match uses a FLANN matcher. FLANN is an approximate nearest-neighbour
 *    index built for float descriptors; for BINARY descriptors exhaustive Hamming is both
 *    correct and, at the feature counts involved here, faster. We therefore brute-force.
 */

#include <halp/controls.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
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

// A detected & described keypoint, in *pixel* coordinates of the BASE (octave 0) image;
// the caller normalises for ports. Field set mirrors cv.jit's 6-plane keypoint row
// (x, y, size, angle, response, octave) plus the descriptor.
struct kp
{
  float x{}, y{};   // base-image pixel position
  float size{};     // diameter of the meaningful neighbourhood, in base-image pixels
  float angle{};    // RADIANS (cv.jit/OpenCV use degrees — see the header comment)
  float response{}; // detector strength: the FAST arc score at the detection octave
  int octave{};     // pyramid level the corner was detected at (0 = base image)
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

// ---------------------------------------------------------------- BRIEF pre-smoothing
// BRIEF is DEFINED on a smoothed patch, not on raw pixels. OpenCV's ORB runs
//     GaussianBlur(image, blurred, Size(7,7), 2, 2, BORDER_REFLECT_101)
// on each pyramid level before computing descriptors (modules/features2d/src/orb.cpp), and
// Calonder's original BRIEF paper smooths with a 9x9 Gaussian of sigma ~2. The reason is that
// a raw single-pixel comparison is noise-dominated: whenever the two sampled pixels are close
// in value, a +-1 LSB sensor wobble flips the bit, so descriptors of the same physical corner
// in two consecutive camera frames differ in a large number of bits and the Lowe ratio test
// throws the match away. Synthetic test images (hash noise, hard-edged blobs) hide this
// completely because they have no noise at all.
//
// This is applied UNCONDITIONALLY rather than behind a control: unsmoothed BRIEF is not an
// alternative flavour of the descriptor, it is simply not BRIEF, and OpenCV's ORB exposes no
// switch for it either. Only describe() reads the smoothed buffer; orientation() keeps
// reading the raw level, exactly as ORB does (the intensity centroid already integrates over
// a radius-12 disc, so pre-blurring it would only throw away resolution).
//
// Kernel = cv::getGaussianKernel(7, 2), i.e. w[i] = exp(-(i-3)^2 / (2 * 2^2)) normalised:
//   exp(0) = 1, exp(-1/8) = 0.88249690, exp(-1/2) = 0.60653066, exp(-9/8) = 0.32465247
//   sum    = 1 + 2*(0.88249690 + 0.60653066 + 0.32465247) = 4.62736006
// giving the seven weights below (they sum to exactly 1).
inline constexpr std::array<float, 7> kBriefBlurKernel{
    0.07015933f, 0.13107488f, 0.19071281f, 0.21610596f,
    0.19071281f, 0.13107488f, 0.07015933f};

// BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba — mirror WITHOUT repeating the edge sample.
[[nodiscard]] inline int reflect101(int i, int n) noexcept
{
  if(n <= 1)
    return 0;
  while(i < 0 || i >= n)
    i = (i < 0) ? -i : 2 * (n - 1) - i;
  return i;
}

// Separable 7x7 sigma-2 Gaussian, BORDER_REFLECT_101. `dst` and `tmp` are resized as needed
// and reused across calls.
inline void blurForBrief(
    const float* src, int W, int H, std::vector<float>& dst, std::vector<float>& tmp)
{
  const std::size_t N = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
  dst.resize(N);
  tmp.resize(N);
  if(N == 0)
    return;
  const auto& k = kBriefBlurKernel;

  for(int y = 0; y < H; ++y) // horizontal pass: src -> tmp
  {
    const float* row = src + static_cast<std::size_t>(y) * W;
    float* o = tmp.data() + static_cast<std::size_t>(y) * W;
    for(int x = 0; x < W; ++x)
    {
      float acc = 0.f;
      for(int i = -3; i <= 3; ++i)
        acc += k[static_cast<std::size_t>(i + 3)]
               * row[reflect101(x + i, W)];
      o[x] = acc;
    }
  }
  for(int y = 0; y < H; ++y) // vertical pass: tmp -> dst
  {
    float* o = dst.data() + static_cast<std::size_t>(y) * W;
    for(int x = 0; x < W; ++x)
    {
      float acc = 0.f;
      for(int i = -3; i <= 3; ++i)
        acc += k[static_cast<std::size_t>(i + 3)]
               * tmp[static_cast<std::size_t>(reflect101(y + i, H)) * W + x];
      o[x] = acc;
    }
  }
}

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
// `gray` must be the SMOOTHED level (see blurForBrief above) — BRIEF on raw pixels is
// noise-dominated. detectAndDescribeMulti() takes care of that for its callers.
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

// Classification of one circle sample against the two thresholds, as a 2-BIT MASK:
//   kFastBrighter (2) = v > Ip + t,  kFastDarker (1) = v < Ip - t,  0 = neither.
// The mask encoding is what turns the high-speed rejection below into a couple of ANDs.
inline constexpr int kFastDarker = 1;
inline constexpr int kFastBrighter = 2;

[[nodiscard]] inline int fastClassify(float v, float lo, float hi) noexcept
{
  return (v > hi) ? kFastBrighter : (v < lo) ? kFastDarker : 0;
}

// High-speed rejection test for FAST-9 on the 16-point circle. `at(k)` yields circle sample k.
//
// The familiar "at least 3 of the 4 compass points 0/4/8/12" test is the high-speed test for
// FAST-*12*, and is WRONG for N = 9: a run of 9 consecutive indices mod 16 contains only TWO
// multiples of 4 unless it happens to start on one, so 12 of the 16 arc phases are rejected
// outright. (Measured on a 256x256 frame at threshold 0.08: 6.0% of corners lost on hash
// noise, 29% on band-limited noise with 4 px structure, 66% with 8 px structure. The smoother
// the image -- i.e. the more it looks like a camera frame -- the worse it gets.)
//
// The test that IS valid for N = 9 is OpenCV's (modules/features2d/src/fast.cpp): AND together
// the DIAMETRICALLY-OPPOSITE pairs (k, k+8). Any 9 consecutive indices mod 16 must contain at
// least one member of every such pair, because the complement is only 7 consecutive indices
// and two indices 8 apart can never both fit inside a 7-wide window. Hence "neither member of
// some pair is brighter" proves no brighter 9-arc exists, and likewise for darker. All eight
// pairs are individually valid, so all eight are used, as OpenCV does.
//
// This is NOT a speed optimisation: it costs 8-16 samples where the (wrong) compass test cost
// 4, and it lets ~3x more pixels through to the arc test and the score sum, so detection gets
// measurably slower. It is chosen because it is the correct N = 9 rejection; the minimal
// "< 2 of 4 compass points" patch would also be sound but is a much weaker filter (it lets
// through everything with 2 of 4, most of which is not a corner) and it has no textbook
// provenance, so it would invite exactly the same "fix" again later.
//
// Returns the mask of arc polarities still possible; 0 means "definitely not a corner".
// Evaluated in two stages (even pairs, then odd pairs) so most flat pixels cost 8 samples.
template <typename At>
[[nodiscard]] inline int fastQuickReject(At&& at, float lo, float hi) noexcept
{
  int d = fastClassify(at(0), lo, hi) | fastClassify(at(8), lo, hi);
  if(d == 0)
    return 0;
  d &= fastClassify(at(2), lo, hi) | fastClassify(at(10), lo, hi);
  d &= fastClassify(at(4), lo, hi) | fastClassify(at(12), lo, hi);
  d &= fastClassify(at(6), lo, hi) | fastClassify(at(14), lo, hi);
  if(d == 0)
    return 0;
  d &= fastClassify(at(1), lo, hi) | fastClassify(at(9), lo, hi);
  d &= fastClassify(at(3), lo, hi) | fastClassify(at(11), lo, hi);
  d &= fastClassify(at(5), lo, hi) | fastClassify(at(13), lo, hi);
  d &= fastClassify(at(7), lo, hi) | fastClassify(at(15), lo, hi);
  return d;
}

// FAST-9 segment test at (x,y). Returns the corner score (sum of abs diffs over the whole
// circle) or 0 if not a corner. gray row-major, t = threshold in [0,1].
//
// This is THE FAST-9 implementation of the addon: CV/Cpu/FastCorners.cpp calls it too, so the
// detector the FastCorners object exposes and the one ORB/FeatureMatch run on cannot drift
// apart (they previously carried two hand-copied versions of the same test, and the same bug).
[[nodiscard]] inline float fastScore(
    const float* gray, int W, int x, int y, float t) noexcept
{
  auto at = [&](int xx, int yy) -> float {
    return gray[static_cast<std::size_t>(yy) * W + xx];
  };
  const float Ip = at(x, y);
  const float hi = Ip + t;
  const float lo = Ip - t;
  auto circle = [&](int k) -> float {
    return at(x + kCircle[static_cast<std::size_t>(k)].first,
              y + kCircle[static_cast<std::size_t>(k)].second);
  };

  const int possible = fastQuickReject(circle, lo, hi);
  if(possible == 0)
    return 0.f;

  std::array<int, 16> cls;
  for(int k = 0; k < 16; ++k)
    cls[static_cast<std::size_t>(k)] = fastClassify(circle(k), lo, hi);

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

  const bool bright = (possible & kFastBrighter) != 0 && hasArc(kFastBrighter);
  const bool dark = !bright && (possible & kFastDarker) != 0 && hasArc(kFastDarker);
  if(!bright && !dark)
    return 0.f;

  float score = 0.f;
  for(int k = 0; k < 16; ++k)
    score += std::abs(circle(k) - Ip);
  return score;
}

// Nominal keypoint diameter at octave 0, in pixels. ORB's convention (patchSize = 31); it is
// what `size` scales up from, so a corner found at octave o reports 31 * 2^o. This is the
// analogue of cv.jit's KEYPOINT_SIZE plane (BRISK reports its own patch size there).
inline constexpr float kBaseKeypointSize = 31.f;

// Detect oriented FAST corners and compute rotated-BRIEF descriptors over a whole frame,
// across `octaves` levels of a halving image pyramid (octaves = 1 => single scale, exactly
// the pre-pyramid behaviour).
//
// rgba: tightly-packed RGBA8, W*H. Fills `out` (cleared first), capped at maxFeatures, kept
// to the strongest by score *across all octaves*. `out` holds keypoints whose x/y are in
// BASE-image pixels, with `octave` recording where they were found and `size` scaled
// accordingly. Descriptors are always computed on the octave's own (downsampled) image, so
// the same physical feature seen at two different scales yields the same descriptor — that
// is what makes matching scale-invariant.
//
// grayScratch holds the octave-0 grayscale image and is reused across calls; the downsampled
// levels are allocated locally (they are 1/4, 1/16, ... the size, so this is cheap).
inline void detectAndDescribeMulti(
    const std::uint8_t* rgba, int W, int H, float threshold, int maxFeatures, int octaves,
    std::vector<float>& grayScratch, std::vector<kp>& out)
{
  out.clear();
  if(W <= 0 || H <= 0)
    return;
  // Clamp so `1 << level` below can never approach UB, whatever a patch sends in. 16 is far
  // beyond reach anyway: each level must still be >= 2*border+1 = 41 px wide.
  octaves = std::clamp(octaves, 1, 16);

  const std::size_t N = static_cast<std::size_t>(W) * H;
  grayScratch.resize(N);
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t* p = rgba + i * 4;
    grayScratch[i]
        = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }

  // A corner must sit at least kPatchRadius+1 px from every edge OF ITS OWN LEVEL, so that
  // the steered BRIEF pattern (and the intensity-centroid window) never samples outside that
  // level's buffer. This is the border guard that a previous heap-overflow bug came from;
  // it is applied per level, not just at level 0.
  const int border = kPatchRadius + 1;

  // Build the pyramid. Level l has dimensions floor(W/2^l) x floor(H/2^l), produced by a 2x2
  // box average of level l-1 (exactly invertible against a 2x nearest-neighbour upscale,
  // which is what makes the scale-invariance test exact). Stop early once a level is too
  // small to contain a single valid candidate.
  std::vector<int> lw{W}, lh{H};
  std::vector<std::vector<float>> extra; // extra[l - 1] is level l (level 0 = grayScratch)
  for(int l = 1; l < octaves; ++l)
  {
    const int pw = lw[static_cast<std::size_t>(l) - 1];
    const int ph = lh[static_cast<std::size_t>(l) - 1];
    const int nw = pw / 2;
    const int nh = ph / 2;
    if(nw < 2 * border + 1 || nh < 2 * border + 1)
      break;
    const float* src
        = (l == 1) ? grayScratch.data() : extra[static_cast<std::size_t>(l) - 2].data();
    std::vector<float> img(static_cast<std::size_t>(nw) * nh);
    for(int y = 0; y < nh; ++y)
    {
      const float* r0 = src + static_cast<std::size_t>(2 * y) * pw;
      const float* r1 = src + static_cast<std::size_t>(2 * y + 1) * pw;
      float* d = img.data() + static_cast<std::size_t>(y) * nw;
      for(int x = 0; x < nw; ++x)
        d[x] = 0.25f * (r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1]);
    }
    extra.push_back(std::move(img));
    lw.push_back(nw);
    lh.push_back(nh);
  }
  const int levels = static_cast<int>(lw.size());

  auto levelData = [&](int l) -> const float* {
    return (l == 0) ? grayScratch.data() : extra[static_cast<std::size_t>(l) - 1].data();
  };

  // Score every pixel of every level, keeping candidates well inside that level's border.
  struct Cand
  {
    int x, y;
    float score;
    int level;
  };
  std::vector<Cand> cands;
  for(int l = 0; l < levels; ++l)
  {
    const float* gray = levelData(l);
    const int Wl = lw[static_cast<std::size_t>(l)];
    const int Hl = lh[static_cast<std::size_t>(l)];
    for(int y = border; y < Hl - border; ++y)
    {
      for(int x = border; x < Wl - border; ++x)
      {
        const float s = fastScore(gray, Wl, x, y, threshold);
        if(s > 0.f)
          cands.push_back({x, y, s, l});
      }
    }
  }

  // Keep the strongest maxFeatures corners (across octaves).
  if(maxFeatures > 0 && static_cast<int>(cands.size()) > maxFeatures)
  {
    std::nth_element(
        cands.begin(), cands.begin() + maxFeatures, cands.end(),
        [](const Cand& a, const Cand& b) { return a.score > b.score; });
    cands.resize(static_cast<std::size_t>(maxFeatures));
  }

  // Descriptors are sampled from a SMOOTHED copy of the level (OpenCV ORB's 7x7 sigma-2
  // pre-filter, see blurForBrief); orientation stays on the raw level, as in ORB. The blur is
  // computed at most once per level, and only for levels that actually produced a candidate.
  std::vector<std::vector<float>> blurred(static_cast<std::size_t>(levels));
  std::vector<float> blurTmp;
  auto blurredData = [&](int l) -> const float* {
    auto& b = blurred[static_cast<std::size_t>(l)];
    if(b.empty())
      blurForBrief(
          levelData(l), lw[static_cast<std::size_t>(l)], lh[static_cast<std::size_t>(l)], b,
          blurTmp);
    return b.data();
  };

  out.reserve(cands.size());
  for(const auto& c : cands)
  {
    const float* gray = levelData(c.level);
    const int Wl = lw[static_cast<std::size_t>(c.level)];
    const int Hl = lh[static_cast<std::size_t>(c.level)];
    const float angle = orientation(gray, Wl, Hl, c.x, c.y);
    // Level pixel (x) covers base pixels [x*s, (x+1)*s), whose centre is (x+0.5)*s - 0.5.
    // At level 0 (s = 1) this is exactly x, so octaves == 1 is bit-identical to the old
    // single-scale path.
    const float s = static_cast<float>(1 << c.level);
    kp k;
    k.x = (static_cast<float>(c.x) + 0.5f) * s - 0.5f;
    k.y = (static_cast<float>(c.y) + 0.5f) * s - 0.5f;
    k.size = kBaseKeypointSize * s;
    k.angle = angle;
    k.response = c.score;
    k.octave = c.level;
    k.desc = describe(blurredData(c.level), Wl, Hl, c.x, c.y, angle);
    out.push_back(k);
  }
}

// Single-scale convenience overload — the historical entry point. Exactly equivalent to
// detectAndDescribeMulti(..., octaves = 1, ...).
inline void detectAndDescribe(
    const std::uint8_t* rgba, int W, int H, float threshold, int maxFeatures,
    std::vector<float>& grayScratch, std::vector<kp>& out)
{
  detectAndDescribeMulti(rgba, W, H, threshold, maxFeatures, 1, grayScratch, out);
}

// Lowe's ratio test brute-force match. For each `query` descriptor, find the two nearest in
// `train`; accept if best < ratio * secondBest. Returns indices (queryIdx, trainIdx, dist).
//
// cv.jit.keypoints.match hands the same job to cv::DescriptorMatcher::knnMatch(k = 2) and
// then dereferences match[1] unconditionally. When the train set holds fewer than 2
// descriptors knnMatch returns a shorter inner vector and cv.jit reads out of bounds — a
// real crash, reachable just by wiring up a 1-feature template. We do NOT reproduce that:
// `train.size() < 2` means "no second-nearest exists", so no ratio test can pass and we
// return an empty result. (cv.jit's *other* tolerances — silently emitting nothing when a
// set is empty or when keypoint/descriptor counts disagree — ARE reproduced, by the caller.)
struct match_result
{
  int query, train, dist;
};

// Generic form: any two random-access ranges plus accessors yielding a `descriptor`.
template <typename QRange, typename TRange, typename QGet, typename TGet>
inline void matchRatioBy(
    const QRange& query, const TRange& train, QGet&& qdesc, TGet&& tdesc, float ratio,
    std::vector<match_result>& out)
{
  out.clear();
  if(train.size() < 2) // guard: cv.jit crashes here, see above
    return;
  for(int i = 0; i < static_cast<int>(query.size()); ++i)
  {
    const descriptor& q = qdesc(query[static_cast<std::size_t>(i)]);
    int best = 1 << 30, second = 1 << 30, bestJ = -1;
    for(int j = 0; j < static_cast<int>(train.size()); ++j)
    {
      const int d = hamming(q, tdesc(train[static_cast<std::size_t>(j)]));
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

inline void matchRatio(
    const std::vector<kp>& query, const std::vector<kp>& train, float ratio,
    std::vector<match_result>& out)
{
  constexpr auto get = [](const kp& k) -> const descriptor& { return k.desc; };
  matchRatioBy(query, train, get, get, ratio, out);
}

} // namespace cv_support

namespace cv
{
// A keypoint as carried on score list ports. Shared by OrbFeatures (output) and FeatureMatch
// (the two set inputs), so the two objects chain directly with no adapter.
//
// Field order mirrors cv.jit's 6-plane keypoint row (x, y, size, angle, response, octave),
// with the descriptor appended. `position` is normalised [0,1] against the BASE image, so a
// downstream object never needs to know the resolution the detector ran at.
struct keypoint
{
  halp::xy_type<float> position; // normalised [0,1] in the base image
  float size{};                  // diameter in base-image pixels (scales with the octave)
  float angle{};                 // RADIANS (cv.jit emits degrees; see header comment)
  float response{};              // detector strength (FAST arc score at the detection octave)
  int octave{};                  // pyramid level, 0 = base resolution
  cv_support::descriptor desc{}; // 256-bit rotated-BRIEF descriptor, 8 x 32-bit words

  halp_field_names(position, size, angle, response, octave, desc);
};
}
