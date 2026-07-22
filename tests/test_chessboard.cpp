// Tests for the two parity gaps closed in this pass:
//   * ChessboardCorners: sub-pixel corner refinement (cv.jit's cvFindCornerSubPix) with its
//     `window_size` / `zero_zone` parameters, the annotated `Out` texture (cv.jit's first
//     outlet, cvDrawChessboardCorners), and corner emission on partial detections.
//   * OrbFeatures: the `normalize` / `glcoords` coordinate modes of cv.jit.keypoints,
//     including their non-obvious interaction.
//
// ---------------------------------------------------------------------------------------
// SYNTHETIC BOARD — where the "true" corner positions come from
// ---------------------------------------------------------------------------------------
// makeSmoothBoard() renders the analytic, anti-aliased checkerboard
//
//     L(x, y) = 0.5 + 0.5 * tanh(k * sin(pi*(x-x0)/sq)) * tanh(k * sin(pi*(y-y0)/sq))
//
// sampled at pixel CENTRES (pixel i has coordinate i, matching the detector's convention).
// sin(pi*(x-x0)/sq) vanishes exactly at x = x0 + i*sq and changes sign there, so:
//   * the zero set is the full grid of lines x = x0+i*sq and y = y0+j*sq — a checkerboard;
//   * every intersection (x0+i*sq, y0+j*sq) is a saddle: L is ODD-symmetric about it in
//     both axes, so it is the exact, analytically known corner position.
// Choosing a fractional x0 therefore places the ground-truth corners at a KNOWN NON-INTEGER
// position that no integer-pixel detector can reach; that is what the refinement must find.
//
// With W = H = 96, sq = 16, x0 = y0 = 8.3 the in-image junctions are
//     8.3, 24.3, 40.3, 56.3, 72.3, 88.3   (the neighbours at -7.7 and 104.3 fall outside)
// i.e. exactly a 6 x 6 inner-corner grid, each corner 0.3 px away from the nearest integer
// in each axis -> an integer detection is off by sqrt(0.3^2 + 0.3^2) = 0.4243 px.
// The phase is chosen so that NO junction sits within a few pixels of the image border:
// a junction at the very edge leaves a truncated saddle there that the detector picks up
// as a spurious corner, which would pollute the measurement.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/ChessboardCorners.hpp>
#include <CV/Cpu/OrbFeatures.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

using namespace cvtest;
using Catch::Approx;

namespace
{
constexpr float kPi = std::numbers::pi_v<float>;

// Analytic anti-aliased checkerboard; see the derivation at the top of the file.
// `dither` adds a deterministic, spatially varying perturbation (in 8-bit levels) that
// breaks the perfect symmetry of the ideal saddle — used only where a test needs the
// refinement to actually depend on its window.
Image makeSmoothBoard(
    int W, int H, float sq, float x0, float y0, float k = 3.f, int dither = 0)
{
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      const float gx = std::sin(kPi * ((float)x - x0) / sq);
      const float gy = std::sin(kPi * ((float)y - y0) / sq);
      float v = 0.5f + 0.5f * std::tanh(k * gx) * std::tanh(k * gy);
      if(dither != 0)
      {
        // Deterministic hash in [-1,1], scaled to `dither` 8-bit levels.
        const std::uint32_t h
            = (std::uint32_t)(x * 73856093u) ^ (std::uint32_t)(y * 19349663u);
        const float n = (float)((h >> 8) & 0xFF) / 255.f * 2.f - 1.f;
        v += n * (float)dither / 255.f;
      }
      const int q = (int)std::lround(std::clamp(v, 0.f, 1.f) * 255.f);
      img.setGray(x, y, (std::uint8_t)std::clamp(q, 0, 255));
    }
  }
  return img;
}

// Ground-truth corner coordinates of such a board, in pixels. Only junctions inside the
// detector's usable band are listed: the saddle response uses central differences with a
// 2-pixel border, so junctions outside [2, size-3] can never be reported.
std::vector<std::array<float, 2>> trueCorners(int W, int H, float sq, float x0, float y0)
{
  std::vector<float> xs, ys;
  for(float v = x0 - sq * std::ceil(x0 / sq); v <= (float)(W - 3); v += sq)
    if(v >= 2.f)
      xs.push_back(v);
  for(float v = y0 - sq * std::ceil(y0 / sq); v <= (float)(H - 3); v += sq)
    if(v >= 2.f)
      ys.push_back(v);
  std::vector<std::array<float, 2>> out;
  for(float y : ys)
    for(float x : xs)
      out.push_back({x, y});
  return out;
}

struct Run
{
  std::vector<std::array<float, 2>> px; // detected corners, in pixels
  bool found{};
  int count{};
};

Run runBoard(
    Image& img, int cols, int rows, bool subpixel, int win = 11, int zero = -1)
{
  cv::ChessboardCorners obj;
  obj.inputs.cols.value = cols;
  obj.inputs.rows.value = rows;
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.subpixel.value = subpixel;
  obj.inputs.window_size.value = {win, win};
  obj.inputs.zero_zone.value = {zero, zero};
  feed(obj.inputs.image, img);
  obj();

  Run r;
  r.found = obj.outputs.found.value;
  r.count = obj.outputs.count.value;
  for(auto& c : obj.outputs.corners.value)
    r.px.push_back({c.position.x * img.width, c.position.y * img.height});
  return r;
}

// Mean distance from each detected corner to the nearest ground-truth corner. Using the
// nearest truth point (rather than index-matching) makes the metric independent of the
// grid ORDERING, which is a separate, documented best-effort concern.
double meanError(
    const std::vector<std::array<float, 2>>& got,
    const std::vector<std::array<float, 2>>& truth)
{
  if(got.empty() || truth.empty())
    return 1e9;
  double acc = 0;
  for(auto& g : got)
  {
    double best = 1e18;
    for(auto& t : truth)
    {
      const double dx = g[0] - t[0], dy = g[1] - t[1];
      best = std::min(best, std::sqrt(dx * dx + dy * dy));
    }
    acc += best;
  }
  return acc / (double)got.size();
}

// Distance from a value to the nearest integer.
float fracDist(float v)
{
  return std::abs(v - std::round(v));
}
} // namespace

// ======================================================================== sub-pixel
TEST_CASE("ChessboardCorners sub-pixel refinement beats the integer detection", "[chessboard][subpix]")
{
  // Ground truth at 8.3 + 16*i (see the header derivation): 0.3 px off the pixel grid in
  // each axis, so the best any integer detector can do is 0.4243 px of error.
  const int W = 96, H = 96;
  const float sq = 16.f, x0 = 8.3f, y0 = 8.3f;
  Image board = makeSmoothBoard(W, H, sq, x0, y0);
  const auto truth = trueCorners(W, H, sq, x0, y0);
  REQUIRE(truth.size() == 36u); // 6 x 6 inner corners

  const Run raw = runBoard(board, 6, 6, /*subpixel*/ false);
  const Run ref = runBoard(board, 6, 6, /*subpixel*/ true);

  REQUIRE(raw.count == 36);
  REQUIRE(ref.count == 36);
  CHECK(raw.found);
  CHECK(ref.found);

  const double eRaw = meanError(raw.px, truth);
  const double eRef = meanError(ref.px, truth);
  INFO("mean error: integer = " << eRaw << " px, refined = " << eRef << " px");

  // The raw detection is integer-only, hence exactly sqrt(2)*0.3 = 0.4243 px off...
  CHECK(eRaw == Approx(0.42426).margin(1e-3));
  // ...and refinement must strictly improve on it, by a wide margin.
  CHECK(eRef < eRaw);
  CHECK(eRef < 0.05);

  // Every raw corner sits exactly on the pixel grid; at least one refined corner must be
  // genuinely OFF it (a "refinement" that silently rounded back would fail here).
  for(auto& c : raw.px)
  {
    CHECK(fracDist(c[0]) == Approx(0.f).margin(1e-4));
    CHECK(fracDist(c[1]) == Approx(0.f).margin(1e-4));
  }
  float maxFrac = 0.f;
  for(auto& c : ref.px)
    maxFrac = std::max({maxFrac, fracDist(c[0]), fracDist(c[1])});
  INFO("largest off-grid offset among refined corners: " << maxFrac);
  CHECK(maxFrac > 0.1f);
}

TEST_CASE("ChessboardCorners sub-pixel refinement leaves an exact-integer corner alone", "[chessboard][subpix]")
{
  // Same board, phase 8.0: the analytic corners now land exactly ON pixel centres
  // (8, 24, 40, 56, 72, 88). Refinement has nothing to correct and must not wander.
  const int W = 96, H = 96;
  Image board = makeSmoothBoard(W, H, 16.f, 8.f, 8.f);
  const auto truth = trueCorners(W, H, 16.f, 8.f, 8.f);
  REQUIRE(truth.size() == 36u);

  const Run raw = runBoard(board, 6, 6, false);
  const Run ref = runBoard(board, 6, 6, true);
  REQUIRE(raw.count == 36);
  REQUIRE(ref.count == 36);

  const double eRaw = meanError(raw.px, truth);
  const double eRef = meanError(ref.px, truth);
  INFO("integer = " << eRaw << " px, refined = " << eRef << " px");
  CHECK(eRaw == Approx(0.0).margin(1e-3)); // the detection is already exact
  CHECK(eRef < 0.1);                       // refinement does not move it appreciably
  // Per-corner: nothing moved by more than a tenth of a pixel.
  for(std::size_t i = 0; i < raw.px.size(); ++i)
  {
    CHECK(ref.px[i][0] == Approx(raw.px[i][0]).margin(0.1));
    CHECK(ref.px[i][1] == Approx(raw.px[i][1]).margin(0.1));
  }
}

TEST_CASE("ChessboardCorners window_size changes the refinement result", "[chessboard][subpix]")
{
  // On a PERFECTLY symmetric saddle the least-squares solution is the true corner for any
  // window, so window size would be unobservable. A small deterministic dither breaks that
  // symmetry: the two windows then see different data and must disagree.
  const int W = 96, H = 96;
  Image board = makeSmoothBoard(W, H, 16.f, 8.3f, 8.3f, 3.f, /*dither*/ 10);

  const Run small = runBoard(board, 6, 6, true, /*win*/ 2);
  const Run big = runBoard(board, 6, 6, true, /*win*/ 11);
  REQUIRE(small.count == big.count);
  REQUIRE(small.count > 0);

  // Same image and same detection -> corners correspond index-wise; only the window differs.
  float maxDiff = 0.f;
  for(std::size_t i = 0; i < small.px.size(); ++i)
    maxDiff = std::max(
        {maxDiff, std::abs(small.px[i][0] - big.px[i][0]),
         std::abs(small.px[i][1] - big.px[i][1])});
  INFO("largest disagreement between window 2 and window 11: " << maxDiff << " px");
  CHECK(maxDiff > 1e-3f);
  // Both remain sane refinements, not divergence.
  CHECK(maxDiff < 2.f);
}

TEST_CASE("ChessboardCorners zero_zone changes the refinement result", "[chessboard][subpix]")
{
  // zero_zone blanks the weights of the central block of the window. On the dithered board
  // that measurably changes the solution; with the cv.jit default (-1 -1) it is inactive.
  const int W = 96, H = 96;
  Image board = makeSmoothBoard(W, H, 16.f, 8.3f, 8.3f, 3.f, /*dither*/ 10);

  const Run none = runBoard(board, 6, 6, true, 11, /*zero*/ -1);
  const Run dead = runBoard(board, 6, 6, true, 11, /*zero*/ 3);
  REQUIRE(none.count == dead.count);
  REQUIRE(none.count > 0);

  float maxDiff = 0.f;
  for(std::size_t i = 0; i < none.px.size(); ++i)
    maxDiff = std::max(
        {maxDiff, std::abs(none.px[i][0] - dead.px[i][0]),
         std::abs(none.px[i][1] - dead.px[i][1])});
  INFO("zero_zone -1 vs 3 disagreement: " << maxDiff << " px");
  CHECK(maxDiff > 1e-4f);
}

TEST_CASE("ChessboardCorners refines corners hugging the image border safely", "[chessboard][subpix]")
{
  // Junctions at 3.4 + 8*i on a 40x40 image: the first is 3 px from the edge while the
  // refinement half-window is 11 px, so the window extends ~9 px outside the image. Sampling
  // must clamp (BORDER_REPLICATE) instead of reading out of bounds — ASan is the real
  // assertion here; the checks below guard against NaNs and runaway positions.
  const int W = 40, H = 40;
  Image board = makeSmoothBoard(W, H, 8.f, 3.4f, 3.4f);

  const Run ref = runBoard(board, 5, 5, true, /*win*/ 11);
  REQUIRE(ref.count > 0);
  for(auto& c : ref.px)
  {
    CHECK(std::isfinite(c[0]));
    CHECK(std::isfinite(c[1]));
    CHECK(c[0] >= 0.f);
    CHECK(c[1] >= 0.f);
    CHECK(c[0] <= (float)W);
    CHECK(c[1] <= (float)H);
  }

  // A 1x1 and a tiny image must also survive the refinement path. A CONSTANT image has zero
  // gradient everywhere, so its saddle response is identically zero and no corner can pass
  // the gate: the answer is exactly 0, not merely "some non-negative number". (`count >= 0`
  // is unfalsifiable for an int count and asserted nothing.)
  for(int n : {1, 2, 5})
  {
    Image tiny(n, n, 128);
    Run r = runBoard(tiny, 2, 2, true, 11);
    INFO("uniform " << n << "x" << n);
    CHECK(r.count == 0);
    CHECK_FALSE(r.found);
  }
}

// ==================================================================== annotated output
namespace
{
struct Annotated
{
  std::vector<std::uint8_t> px;
  int W{}, H{};
  bool found{};
  int count{};

  int countColor(int r, int g, int b) const
  {
    int n = 0;
    for(std::size_t i = 0; i + 3 < px.size(); i += 4)
      if(px[i] == r && px[i + 1] == g && px[i + 2] == b)
        ++n;
    return n;
  }
};

Annotated annotate(Image& img, int cols, int rows)
{
  cv::ChessboardCorners obj;
  obj.inputs.cols.value = cols;
  obj.inputs.rows.value = rows;
  obj.inputs.threshold.value = 0.5f;
  feed(obj.inputs.image, img);
  obj();

  Annotated a;
  a.found = obj.outputs.found.value;
  a.count = obj.outputs.count.value;
  a.W = obj.outputs.image.texture.width;
  a.H = obj.outputs.image.texture.height;
  REQUIRE(obj.outputs.image.texture.bytes != nullptr);
  a.px.assign(
      obj.outputs.image.texture.bytes,
      obj.outputs.image.texture.bytes + (std::size_t)a.W * a.H * 4);
  CHECK(obj.outputs.image.texture.changed);
  return a;
}
} // namespace

TEST_CASE("ChessboardCorners writes an annotated texture", "[chessboard][draw]")
{
  const int W = 96, H = 96;
  Image board = makeSmoothBoard(W, H, 16.f, 8.3f, 8.3f);

  // Found: the real 6x6 grid.
  Annotated ok = annotate(board, 6, 6);
  REQUIRE(ok.found);
  CHECK(ok.W == W);
  CHECK(ok.H == H);
  REQUIRE(ok.px.size() == (std::size_t)W * H * 4);

  // The texture is written: it is neither blank nor a bare copy of the input.
  bool allBlank = true, sameAsInput = true;
  for(std::size_t i = 0; i < ok.px.size(); ++i)
  {
    if(ok.px[i] != 0)
      allBlank = false;
    if(ok.px[i] != board.px[i])
      sameAsInput = false;
  }
  CHECK_FALSE(allBlank);
  CHECK_FALSE(sameAsInput); // corners were actually drawn on top

  // Not found: ask for a 9x9 grid the board does not contain.
  Annotated bad = annotate(board, 9, 9);
  REQUIRE_FALSE(bad.found);

  // The two annotations differ.
  CHECK(ok.px != bad.px);

  // The found case draws the ordered polyline with cv.jit's per-row palette (row 3 is pure
  // green); the not-found case draws red circles only. Green is therefore present in one and
  // absent from the other — a check that survives any change of marker geometry.
  const int okGreen = ok.countColor(0, 255, 0);
  const int badGreen = bad.countColor(0, 255, 0);
  const int badRed = bad.countColor(255, 0, 0);
  INFO("green px found=" << okGreen << " notfound=" << badGreen << ", red notfound=" << badRed);
  CHECK(okGreen > 0);
  CHECK(badGreen == 0);
  CHECK(badRed > 0);
}

TEST_CASE("ChessboardCorners emits corners on a partial detection", "[chessboard]")
{
  // A board with only a 3x3 junction grid, queried as 5x5. cv.jit hands its corner outlet
  // whatever cvFindChessboardCorners produced regardless of the `found` flag, and refines
  // those points too; reproduce that instead of reporting nothing.
  const int W = 64, H = 64;
  Image board = makeSmoothBoard(W, H, 16.f, 8.3f, 8.3f);
  REQUIRE(trueCorners(W, H, 16.f, 8.3f, 8.3f).size() == 4u * 4u);

  const Run r = runBoard(board, 6, 6, true);
  CHECK_FALSE(r.found);  // not the requested grid...
  CHECK(r.count > 0);    // ...but the partial detection is still reported
  CHECK(r.count < 6 * 6);
  CHECK((int)r.px.size() == r.count);

  // And those partial corners are sub-pixel refined, not integers.
  const Run rawPartial = runBoard(board, 6, 6, false);
  float maxFrac = 0.f;
  for(auto& c : r.px)
    maxFrac = std::max({maxFrac, fracDist(c[0]), fracDist(c[1])});
  INFO("partial corners: raw=" << rawPartial.count << " refined off-grid=" << maxFrac);
  CHECK(maxFrac > 0.1f);
}

// ================================================================ OrbFeatures coordinates
namespace
{
// A deterministic, spatially unique texture so the detector finds a stable set of corners.
Image makeOrbTexture(int W, int H)
{
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      std::uint64_t h = (std::uint64_t)(std::uint32_t)(x + 1000) * 374761u
                        + (std::uint64_t)(std::uint32_t)(y + 1000) * 668265u;
      h = ((h ^ (h >> 13)) & 0xFFFFFFu) * 12743u;
      img.setGray(x, y, (std::uint8_t)((h >> 5) & 0xFF));
    }
  return img;
}

std::vector<cv::keypoint>
runOrb(Image& img, cv::Coordinates mode, bool glCentering)
{
  cv::OrbFeatures obj;
  obj.inputs.threshold.value = 0.08f;
  obj.inputs.max_features.value = 256;
  obj.inputs.octaves.value = 1; // single scale -> every keypoint is at octave 0
  obj.inputs.coordinates.value = mode;
  obj.inputs.gl_centering.value = glCentering;
  feed(obj.inputs.image, img);
  obj();
  return obj.outputs.keypoints.value;
}
} // namespace

TEST_CASE("OrbFeatures coordinate modes reproduce cv.jit.keypoints exactly", "[orb][coords]")
{
  const int W = 96, H = 64; // deliberately NON-square: sx != sy, so max(sx,sy) is observable
  Image img = makeOrbTexture(W, H);

  const auto pix = runOrb(img, cv::Coordinates::Pixels, false);
  const auto norm = runOrb(img, cv::Coordinates::Normalized, false);
  const auto gl = runOrb(img, cv::Coordinates::GL, false);
  const auto both = runOrb(img, cv::Coordinates::Normalized, true);
  const auto pixGl = runOrb(img, cv::Coordinates::Pixels, true);

  REQUIRE(pix.size() > 4u);
  REQUIRE(norm.size() == pix.size());
  REQUIRE(gl.size() == pix.size());
  REQUIRE(both.size() == pix.size());
  REQUIRE(pixGl.size() == pix.size());

  // Pixels mode really is pixels: FAST reports integer corner positions and, at octave 0,
  // Brief.hpp's mapping (c.x + 0.5)*1 - 0.5 leaves them integral.
  for(auto& k : pix)
  {
    REQUIRE(k.octave == 0);
    CHECK(k.position.x == Approx(std::round(k.position.x)).margin(1e-4));
    CHECK(k.position.y == Approx(std::round(k.position.y)).margin(1e-4));
    CHECK(k.position.x >= 0.f);
    CHECK(k.position.x < (float)W);
    CHECK(k.position.y >= 0.f);
    CHECK(k.position.y < (float)H);
  }

  // cv.jit's scales, spelled out.
  const float nsx = 1.f / W, nsy = 1.f / H;      // @normalize 1
  const float gsx = 2.f / W, gsy = 2.f / H;      // @glcoords 1, @normalize 0
  const float cx = W * 0.5f, cy = H * 0.5f;

  bool sawAboveCentre = false, sawBelowCentre = false;
  for(std::size_t i = 0; i < pix.size(); ++i)
  {
    const float px = pix[i].position.x;
    const float py = pix[i].position.y;

    // @normalize 1 @glcoords 0 -> px/W, py/H. This is the historical default of the object.
    CHECK(norm[i].position.x == Approx(px * nsx));
    CHECK(norm[i].position.y == Approx(py * nsy));
    CHECK(norm[i].position.x >= 0.f);
    CHECK(norm[i].position.x <= 1.f);

    // @normalize 0 @glcoords 1 -> centred, scaled by 2/W and 2/H, Y flipped.
    CHECK(gl[i].position.x == Approx((px - cx) * gsx).margin(1e-6));
    CHECK(gl[i].position.y == Approx((cy - py) * gsy).margin(1e-6));
    CHECK(gl[i].position.x >= -1.f);
    CHECK(gl[i].position.x <= 1.f);

    // @normalize 1 @glcoords 1 -> `normalize` wins for the SCALE (1/W, not 2/W) while
    // `glcoords` still applies the centring and the Y flip: the range is [-0.5, 0.5].
    CHECK(both[i].position.x == Approx((px - cx) * nsx).margin(1e-6));
    CHECK(both[i].position.y == Approx((cy - py) * nsy).margin(1e-6));
    CHECK(both[i].position.x >= -0.5f);
    CHECK(both[i].position.x <= 0.5f);
    CHECK(both[i].position.y >= -0.5f);
    CHECK(both[i].position.y <= 0.5f);
    // Exactly half of the plain GL value — the "textbook" reading (glcoords => [-1,1]
    // regardless of normalize) would give twice this and fail here.
    CHECK(both[i].position.x == Approx(gl[i].position.x * 0.5f).margin(1e-6));
    CHECK(both[i].position.y == Approx(gl[i].position.y * 0.5f).margin(1e-6));

    // THE Y FLIP, asserted on the sign. Un-flipped centred Y would be (py - cy)/H; the
    // emitted value is its exact negation.
    const float unflipped = (py - cy) * nsy;
    CHECK(both[i].position.y == Approx(-unflipped).margin(1e-6));
    if(py < cy)
    {
      sawAboveCentre = true;
      CHECK(unflipped < 0.f);              // above centre in image coordinates
      CHECK(both[i].position.y > 0.f);     // ...is POSITIVE in GL coordinates
    }
    else if(py > cy)
    {
      sawBelowCentre = true;
      CHECK(unflipped > 0.f);
      CHECK(both[i].position.y < 0.f);
    }

    // @normalize 0 @glcoords 1 is what "Pixels + GL centering" must collapse to.
    CHECK(pixGl[i].position.x == Approx(gl[i].position.x).margin(1e-6));
    CHECK(pixGl[i].position.y == Approx(gl[i].position.y).margin(1e-6));

    // SIZE is scaled by max(scale_x, scale_y). W > H here, so 1/H > 1/W and 2/H > 2/W.
    // The other three are RELATIVE to `pix`, so they must be anchored by an ABSOLUTE value
    // first, otherwise the whole block is satisfied by size == 0 everywhere. Pixels mode has
    // scale 1 and every keypoint here is at octave 0 (asserted above, Octaves == 1), so the
    // reported diameter is exactly ORB's nominal patch size at octave 0:
    // 31 * 2^0 == 31 px — ORB's patchSize convention, spelled out as a literal on purpose
    // (referring to cv_support::kBaseKeypointSize would just be the same self-comparison one
    // level of indirection away; tests/test_match2.cpp's expectedSize() hardcodes 31.f too).
    // (This line previously read `pix[i].size == Approx(pix[i].size)` — a self-comparison
    // that cannot fail, which let the entire size block be satisfied by size == 0.)
    CHECK(pix[i].size == Approx(31.f)); // Pixels mode: scale 1, octave 0
    CHECK(norm[i].size == Approx(pix[i].size * std::max(nsx, nsy)));
    CHECK(gl[i].size == Approx(pix[i].size * std::max(gsx, gsy)));
    CHECK(both[i].size == Approx(pix[i].size * std::max(nsx, nsy)));
  }

  // The sign-flip assertions above are only meaningful if both halves were exercised.
  CHECK(sawAboveCentre);
  CHECK(sawBelowCentre);
}

TEST_CASE("OrbFeatures defaults to the historical normalised behaviour", "[orb][coords]")
{
  // Default-constructed: Coordinates::Normalized, GL centering off. Positions must be the
  // plain [0,1] values the object has always emitted, so tests/test_features.cpp is unaffected.
  const int W = 64, H = 64;
  Image img = makeOrbTexture(W, H);

  cv::OrbFeatures dflt;
  dflt.inputs.threshold.value = 0.08f;
  dflt.inputs.max_features.value = 256;
  dflt.inputs.octaves.value = 1;
  feed(dflt.inputs.image, img);
  dflt();

  CHECK(dflt.inputs.coordinates.value == cv::Coordinates::Normalized);
  CHECK_FALSE(bool(dflt.inputs.gl_centering.value));

  const auto pix = runOrb(img, cv::Coordinates::Pixels, false);
  REQUIRE(dflt.outputs.keypoints.value.size() == pix.size());
  REQUIRE(dflt.outputs.count.value == (int)pix.size());
  for(std::size_t i = 0; i < pix.size(); ++i)
  {
    CHECK(dflt.outputs.keypoints.value[i].position.x == Approx(pix[i].position.x / W));
    CHECK(dflt.outputs.keypoints.value[i].position.y == Approx(pix[i].position.y / H));
  }
}
