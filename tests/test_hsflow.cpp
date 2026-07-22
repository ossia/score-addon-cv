// Tests for CV/Cpu/HornSchunck.hpp — the port of cv.jit.HSflow (dense Horn-Schunck
// optical flow).
//
// Reference: cv.jit/source/projects/cv.jit.HSflow/{cv.jit.HSflow,max.cv.jit.HSflow}.cpp
//   attributes: maxiter (long, 3), lambda (float32, 0.001), threshold (float32, 0.f)
//   call:       cvCalcOpticalFlowHS(prev, cur, /usePrevious/ 0, flowX, flowY, lambda,
//                                   cvTermCriteria(ITER+EPS, maxIter, threshold))
//
// ---------------------------------------------------------------------------- derivations
//
// The object implements the classic 1981 Horn & Schunck iteration
//
//     t = (Ex*ubar + Ey*vbar + Et) / (lambda^2 + Ex^2 + Ey^2)
//     u = ubar - Ex*t ,   v = vbar - Ey*t
//
// with the 1/6 (4-neighbour) + 1/12 (diagonal) averaging kernel, Ex/Ey/Et taken over the
// 2x2x2 forward-difference cube, and luminance normalised to [0,1]. `lambda` is the
// smoothness weight alpha (bigger = smoother) — see the LAMBDA CONVENTION block in the
// header; it is *not* OpenCV's Lagrange multiplier, whose sense is inverted.
//
// The synthetic scene used for the translation tests is
//
//     I(x,y) = 128 + 50*sin(2*pi*x/16) + 50*sin(2*pi*y/16)
//
// sampled at (x - sx, y - sy) so that the content is translated by exactly (+sx, +sy)
// pixels. Two properties make it the right probe:
//   * it is smooth and band-limited (period 16 px), so the brightness-constancy
//     linearisation is valid for a 1 px displacement — the recovered flow must be the true
//     displacement, not a fraction of it;
//   * BOTH Ex and Ey are non-zero almost everywhere, so a *single* HS sweep can only
//     recover the normal flow, i.e. the projection of the true (1,0) onto the local
//     gradient direction, which is strictly shorter than 1. This is exactly the aperture
//     problem HS resolves by *iterating*, and it is what separates a real Horn-Schunck
//     from the single-pass "Horn-Schunck-style" shader CV/Shaders/Filters/DenseFlow.fs.
//     Measured mean u over the interior for a (+1,0) shift:
//        1 sweep 0.485 | 2 0.544 | 3 0.589 | 5 0.658 | 10 0.774 | 20 0.896 | 50 0.989
//     which is where the "iteration count matters" numbers below come from.
//
// All measurements are taken over an interior margin (8 px) so that border replication in
// the averaging kernel does not enter the assertions.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ScoreTextureModel.hpp"
#include "TestImage.hpp"

#include <CV/Cpu/CartoPol.hpp>
#include <CV/Cpu/HornSchunck.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using Catch::Approx;
using namespace cvtest;

namespace
{
constexpr double kPi = 3.14159265358979323846;

// The flow textures are r32f and, per the addon-wide contract (see the block at the top of
// CV/Cpu/CartoPol.hpp), carry [0,1]: the SIGNED px/frame components are bipolar-encoded
// against the `Flow scale` value port. Reading the raw float and calling it a velocity is
// exactly the mistake this suite now guards against, so every accessor decodes.
float fx(cv::HornSchunck& o, int x, int y)
{
  auto& t = o.outputs.dx.texture;
  const float e
      = reinterpret_cast<const float*>(t.bytes)[static_cast<std::size_t>(y) * t.width + x];
  return cv::polar_codec::decode_signed(e, o.outputs.flow_scale.value);
}
float fy(cv::HornSchunck& o, int x, int y)
{
  auto& t = o.outputs.dy.texture;
  const float e
      = reinterpret_cast<const float*>(t.bytes)[static_cast<std::size_t>(y) * t.width + x];
  return cv::polar_codec::decode_signed(e, o.outputs.flow_scale.value);
}

// I(x,y) = 128 + 50 sin(2pi x/T) + 50 sin(2pi y/T) sampled at (x-sx, y-sy): the pattern
// appears translated by (+sx,+sy) pixels. Sub-pixel shifts are exact (analytic sampling).
Image sinePattern(int W, int H, float sx, float sy, float T = 16.f)
{
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      const double u = x - sx, v = y - sy;
      const double g
          = 128. + 50. * std::sin(2 * kPi * u / T) + 50. * std::sin(2 * kPi * v / T);
      img.setGray(
          x, y, static_cast<std::uint8_t>(std::lround(std::clamp(g, 0., 255.))));
    }
  return img;
}

// Deterministic value noise: a coarse random grid (cell x cell) bilinearly upsampled.
// cell == 1 gives per-pixel white noise.
Image noisePattern(int W, int H, unsigned seed, int cell)
{
  const int gw = W / cell + 3, gh = H / cell + 3;
  std::vector<float> g(static_cast<std::size_t>(gw) * gh);
  unsigned st = seed;
  for(auto& v : g)
  {
    st = st * 1664525u + 1013904223u;
    v = static_cast<float>(st >> 8) / 16777216.f;
  }
  auto G = [&](int a, int b) {
    a = std::clamp(a, 0, gw - 1);
    b = std::clamp(b, 0, gh - 1);
    return g[static_cast<std::size_t>(b) * gw + a];
  };
  Image img(W, H, 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      const float gx = static_cast<float>(x) / cell + 1.f;
      const float gy = static_cast<float>(y) / cell + 1.f;
      const int x0 = static_cast<int>(std::floor(gx)), y0 = static_cast<int>(std::floor(gy));
      const float ax = gx - x0, ay = gy - y0;
      const float v = G(x0, y0) * (1 - ax) * (1 - ay) + G(x0 + 1, y0) * ax * (1 - ay)
                      + G(x0, y0 + 1) * (1 - ax) * ay + G(x0 + 1, y0 + 1) * ax * ay;
      img.setGray(
          x, y, static_cast<std::uint8_t>(std::lround(std::clamp(v * 255.f, 0.f, 255.f))));
    }
  return img;
}

struct Interior
{
  double mean_u{}, mean_v{}, rmse{}, max_err{};
};

Interior interior(cv::HornSchunck& o, int W, int H, int m, float tx, float ty)
{
  Interior r;
  double su = 0, sv = 0, sq = 0;
  int n = 0;
  for(int y = m; y < H - m; ++y)
    for(int x = m; x < W - m; ++x)
    {
      const double du = fx(o, x, y) - tx, dv = fy(o, x, y) - ty;
      su += fx(o, x, y);
      sv += fy(o, x, y);
      sq += du * du + dv * dv;
      r.max_err = std::max(r.max_err, std::hypot(du, dv));
      ++n;
    }
  r.mean_u = su / n;
  r.mean_v = sv / n;
  r.rmse = std::sqrt(sq / n);
  return r;
}

// Total variation of the flow field (L1 of the forward differences of both components)
// and its L1 mass, over an interior margin.
struct Smoothness
{
  double tv{}, l1{};
  [[nodiscard]] double relative() const { return tv / l1; } // scale-invariant roughness
};

Smoothness smoothness(cv::HornSchunck& o, int W, int H, int m)
{
  Smoothness s;
  for(int y = m; y < H - m; ++y)
    for(int x = m; x < W - m; ++x)
    {
      s.tv += std::abs(fx(o, x + 1, y) - fx(o, x, y))
              + std::abs(fx(o, x, y + 1) - fx(o, x, y))
              + std::abs(fy(o, x + 1, y) - fy(o, x, y))
              + std::abs(fy(o, x, y + 1) - fy(o, x, y));
      s.l1 += std::abs(fx(o, x, y)) + std::abs(fy(o, x, y));
    }
  return s;
}

bool allFinite(cv::HornSchunck& o, int W, int H)
{
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
      if(!std::isfinite(fx(o, x, y)) || !std::isfinite(fy(o, x, y)))
        return false;
  return true;
}
}

// ============================================================================ first frame
//
// DELIBERATE DEVIATION from cv.jit, and the reason this test exists: cv.jit.HSflow keeps
// the previous frame in a Jitter matrix that is `_jit_sym_clear`ed in cv_jit_HSflow_new()
// and only filled at the *end* of the first matrix_calc. Its first frame is therefore
// matched against an all-black image and emits a large spurious flow field. Here the first
// frame emits exactly zero and reports zero iterations. A literal transcription of cv.jit
// would fail this test — do not "fix" it back.
TEST_CASE("HSflow: the first frame emits exactly zero flow", "[hsflow]")
{
  const int W = 32, H = 32;
  cv::HornSchunck o;
  // A frame that is *maximally* unlike black, so cv.jit's zero-cleared previous matrix
  // would have produced a huge flow here.
  Image bright = sinePattern(W, H, 0, 0);
  feed(o.inputs.image, bright);
  o();

  CHECK(o.outputs.iterations.value == 0);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      REQUIRE(fx(o, x, y) == 0.f); // exactly zero, not approximately
      REQUIRE(fy(o, x, y) == 0.f);
    }
  CHECK(o.outputs.dx.texture.changed);
  CHECK(o.outputs.dy.texture.changed);
}

// ====================================================================== translation recovery
//
// 50 sweeps, default lambda (0.001). Measured over the 8 px interior margin:
//   (+1, 0): mean (0.9888, 0.0000), max pointwise vector error 0.060
//   ( 0,+1): mean (0.0000, 0.9888)   (symmetric by construction)
//   (+1,+1): mean (0.9897, 0.9897)
//   (-1, 0): mean (-0.9887, 0.0000)
// Stated tolerance: 0.12 per pixel on the vector error, i.e. 2x the worst measured value.
TEST_CASE("HSflow: recovers a known pure translation with the right sign", "[hsflow]")
{
  const int W = 64, H = 64, M = 8;
  Image ref = sinePattern(W, H, 0, 0);

  auto run = [&](float sx, float sy) {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 50;
    Image shifted = sinePattern(W, H, sx, sy);
    feed(o.inputs.image, ref);
    o();
    feed(o.inputs.image, shifted);
    o();
    CHECK(o.outputs.iterations.value == 50);
    auto s = interior(o, W, H, M, sx, sy);
    // Per-pixel bound, not just the mean: the field must be right everywhere inside.
    for(int y = M; y < H - M; ++y)
      for(int x = M; x < W - M; ++x)
        REQUIRE(std::hypot(fx(o, x, y) - sx, fy(o, x, y) - sy) < 0.12);
    return s;
  };

  SECTION("horizontal, +1 px")
  {
    auto s = run(1.f, 0.f);
    CHECK(s.mean_u > 0.9); // right sign and near-unit magnitude
    CHECK(s.mean_u < 1.05);
    CHECK(std::abs(s.mean_v) < 0.02); // no cross-talk into dy
  }
  SECTION("horizontal, -1 px (sign must flip; an unsigned 8-bit encoding could not)")
  {
    auto s = run(-1.f, 0.f);
    CHECK(s.mean_u < -0.9);
    CHECK(s.mean_u > -1.05);
    CHECK(std::abs(s.mean_v) < 0.02);
  }
  SECTION("vertical, +1 px")
  {
    auto s = run(0.f, 1.f);
    CHECK(s.mean_v > 0.9);
    CHECK(s.mean_v < 1.05);
    CHECK(std::abs(s.mean_u) < 0.02);
  }
  SECTION("diagonal, (+1,+1) px")
  {
    auto s = run(1.f, 1.f);
    CHECK(s.mean_u > 0.9);
    CHECK(s.mean_u < 1.05);
    CHECK(s.mean_v > 0.9);
    CHECK(s.mean_v < 1.05);
  }
  SECTION("sub-pixel: a (+0.5, 0) shift is resolved, which r8 output could not carry")
  {
    auto s = run(0.5f, 0.f);
    CHECK(s.mean_u > 0.45);
    CHECK(s.mean_u < 0.55);
    CHECK(std::abs(s.mean_v) < 0.02);
    CHECK(std::abs(s.mean_u - 1.0) > 0.4); // genuinely distinct from both 0 and 1
  }
}

// ==================================================================== iteration count matters
//
// THE discriminator against a single-pass "flow" filter. One HS sweep can only produce the
// normal flow (the projection of the true motion onto the local gradient); information is
// propagated across the image solely by iterating the averaging step. The error must
// therefore decrease strictly and substantially with maxiter.
TEST_CASE("HSflow: maxiter strictly improves the estimate", "[hsflow]")
{
  const int W = 64, H = 64, M = 8;
  Image a = sinePattern(W, H, 0, 0);
  Image b = sinePattern(W, H, 1, 0);

  const int iters[] = {1, 2, 3, 5, 10, 20, 50};
  double prev_rmse = 1e30;
  double first_rmse = 0, last_rmse = 0;
  for(std::size_t k = 0; k < std::size(iters); ++k)
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = iters[k];
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    CHECK(o.outputs.iterations.value == iters[k]);
    const double e = interior(o, W, H, M, 1.f, 0.f).rmse;
    INFO("maxiter = " << iters[k] << " rmse = " << e);
    REQUIRE(e < prev_rmse); // strictly decreasing
    prev_rmse = e;
    if(k == 0)
      first_rmse = e;
    last_rmse = e;
  }
  // Measured: rmse 0.7177 at maxiter = 1, 0.0171 at maxiter = 50 — a factor of 42.
  INFO("rmse(1) = " << first_rmse << "  rmse(50) = " << last_rmse);
  CHECK(first_rmse > 0.6);
  CHECK(last_rmse < 0.05);
  CHECK(first_rmse > 10.0 * last_rmse);
}

// The same point in its purest form: a solid 20x20 white square translated by 1 px on a
// black background. Its interior is completely untextured (Ex = Ey = Et = 0), so the data
// term says nothing there; flow can only arrive from the edges, one averaging-kernel step
// per sweep. At the centre (10 px from every edge) it is still zero after 10 sweeps and
// clearly non-zero after 100. A single-pass filter can never do this.
TEST_CASE("HSflow: flow fills an untextured interior only through iteration", "[hsflow]")
{
  const int W = 64, H = 64;
  Image p1(W, H, 0);
  p1.fillRect(20, 20, 20, 20, 255);
  Image p2(W, H, 0);
  p2.fillRect(21, 20, 20, 20, 255); // same square, moved +1 px in x

  auto centre = [&](int iters) {
    cv::HornSchunck o;
    o.inputs.maxiter.value = iters;
    o.inputs.lambda.value = 0.01f;
    feed(o.inputs.image, p1);
    o();
    feed(o.inputs.image, p2);
    o();
    return static_cast<double>(fx(o, 30, 30));
  };

  CHECK(std::abs(centre(1)) < 1e-6);
  CHECK(std::abs(centre(3)) < 1e-6);
  CHECK(std::abs(centre(10)) < 1e-3);
  CHECK(centre(100) > 0.3); // measured 0.447, and positive: same sign as the motion
}

// ======================================================================== lambda matters
//
// Two *independent* noise frames: there is no true motion, so the data term is pure
// contradiction and the field is whatever the regulariser makes of it. In this port's
// convention lambda is the smoothness weight (see the header), so raising it must produce
// a visibly smoother field. Asserted two ways:
//   * raw total variation strictly decreases;
//   * TV / L1 (a scale-invariant roughness) strictly decreases too, so the effect is real
//     smoothing and not merely a shrinking field.
// Measured at 20 sweeps, 64x64, cell-3 noise:
//   lambda  0.001    0.01     0.05     0.1     0.2      0.5     1.0
//   TV      37870    30519    16273    9981    5034     1370    397
//   TV/L1   1.453    1.370    1.107    0.944   0.787    0.679   0.656
TEST_CASE("HSflow: a larger lambda produces a smoother field", "[hsflow]")
{
  const int W = 64, H = 64, M = 2;
  Image a = noisePattern(W, H, 111u, 3);
  Image b = noisePattern(W, H, 999u, 3);

  const float lambdas[] = {0.001f, 0.01f, 0.05f, 0.1f, 0.2f, 0.5f, 1.f};
  double prev_tv = 1e30, prev_rel = 1e30;
  double tv_lo = 0, tv_hi = 0;
  for(std::size_t k = 0; k < std::size(lambdas); ++k)
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 20;
    o.inputs.lambda.value = lambdas[k];
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    REQUIRE(allFinite(o, W, H));
    const auto s = smoothness(o, W, H, M);
    INFO("lambda = " << lambdas[k] << " tv = " << s.tv << " tv/l1 = " << s.relative());
    REQUIRE(s.tv < prev_tv);
    REQUIRE(s.relative() < prev_rel);
    prev_tv = s.tv;
    prev_rel = s.relative();
    if(k == 0)
      tv_lo = s.tv;
    tv_hi = s.tv;
  }
  CHECK(tv_lo > 20.0 * tv_hi); // measured 37870 -> 397, ~95x
}

// =============================================== threshold (CV_TERMCRIT_EPS) terminates early
//
// cv.jit passes `threshold` as the epsilon of cvTermCriteria(ITER + EPS, maxIter,
// threshold). threshold == 0 (its default) means the epsilon test never fires, so all
// maxiter sweeps run. Measured on the (+1,0) sine transition with maxiter = 200:
//   threshold  1e-2 -> 26 sweeps | 1e-3 -> 60 | 1e-4 -> 96 | 1e-5 -> 133 | 0 -> 200
// and the 1e-3 result is within 0.017 (max pointwise) of the fully-iterated one.
TEST_CASE("HSflow: threshold terminates the iteration early", "[hsflow]")
{
  const int W = 64, H = 64, M = 8;
  Image a = sinePattern(W, H, 0, 0);
  Image b = sinePattern(W, H, 1, 0);

  auto run = [&](float th) {
    auto o = std::make_unique<cv::HornSchunck>();
    o->inputs.maxiter.value = 200;
    o->inputs.threshold.value = th;
    feed(o->inputs.image, a);
    (*o)();
    feed(o->inputs.image, b);
    (*o)();
    return o;
  };

  auto ref = run(0.f);
  CHECK(ref->outputs.iterations.value == 200); // threshold 0 == no epsilon test (cv.jit)

  auto early = run(1e-3f);
  CHECK(early->outputs.iterations.value > 1);
  CHECK(early->outputs.iterations.value < 200); // measured 60

  // ... and the early-terminated field is close to the fully-iterated one.
  double maxdiff = 0;
  for(int y = M; y < H - M; ++y)
    for(int x = M; x < W - M; ++x)
      maxdiff = std::max(
          maxdiff,
          std::hypot(
              static_cast<double>(fx(*early, x, y)) - fx(*ref, x, y),
              static_cast<double>(fy(*early, x, y)) - fy(*ref, x, y)));
  INFO("max |early - converged| = " << maxdiff);
  CHECK(maxdiff < 0.05);

  // A tighter epsilon must run longer (but still stop before maxiter).
  auto tighter = run(1e-5f);
  CHECK(tighter->outputs.iterations.value > early->outputs.iterations.value);
  CHECK(tighter->outputs.iterations.value < 200);
}

// =========================================================================== degenerate inputs
TEST_CASE("HSflow: zero motion yields exactly zero flow", "[hsflow]")
{
  const int W = 32, H = 32;
  Image a = sinePattern(W, H, 0, 0);
  cv::HornSchunck o;
  o.inputs.maxiter.value = 30;
  feed(o.inputs.image, a);
  o();
  feed(o.inputs.image, a); // identical frame: Et == 0 everywhere
  o();
  CHECK(o.outputs.iterations.value == 30);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      REQUIRE(fx(o, x, y) == 0.f);
      REQUIRE(fy(o, x, y) == 0.f);
    }
}

// Ex == Ey == 0 makes the HS denominator lambda^2, which is itself 0 when lambda is 0.
// Every such case must stay finite.
TEST_CASE("HSflow: a uniform image never produces NaN or inf", "[hsflow]")
{
  const int W = 16, H = 16;

  SECTION("uniform, no change, lambda = 0")
  {
    Image a(W, H, 128), b(W, H, 128);
    cv::HornSchunck o;
    o.inputs.lambda.value = 0.f;
    o.inputs.maxiter.value = 10;
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    REQUIRE(allFinite(o, W, H));
    for(int y = 0; y < H; ++y)
      for(int x = 0; x < W; ++x)
        REQUIRE(fx(o, x, y) == 0.f);
  }

  SECTION("uniform, global brightness flash, lambda = 0 (Ex = Ey = 0 but Et != 0)")
  {
    Image a(W, H, 60), b(W, H, 200);
    cv::HornSchunck o;
    o.inputs.lambda.value = 0.f;
    o.inputs.maxiter.value = 10;
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    REQUIRE(allFinite(o, W, H));
    // No spatial gradient anywhere -> the data term cannot say anything, the field stays 0.
    for(int y = 0; y < H; ++y)
      for(int x = 0; x < W; ++x)
        REQUIRE(fx(o, x, y) == 0.f);
  }

  SECTION("white-noise pair with lambda = 0 stays finite")
  {
    Image a = noisePattern(48, 48, 3u, 1);
    Image b = noisePattern(48, 48, 4u, 1);
    cv::HornSchunck o;
    o.inputs.lambda.value = 0.f;
    o.inputs.maxiter.value = 20;
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    REQUIRE(allFinite(o, 48, 48));
  }
}

TEST_CASE("HSflow: degenerate image sizes", "[hsflow]")
{
  for(int S : {1, 2, 3})
  {
    Image a(S, S, 120), b(S, S, 200);
    cv::HornSchunck o;
    feed(o.inputs.image, a);
    o();
    CHECK(o.outputs.iterations.value == 0);
    CHECK(fx(o, 0, 0) == 0.f);
    feed(o.inputs.image, b);
    o();
    CHECK(o.outputs.iterations.value == 3); // the default maxiter
    REQUIRE(allFinite(o, S, S));
  }
}

TEST_CASE("HSflow: an unchanged texture leaves the outputs alone", "[hsflow]")
{
  cv::HornSchunck o;
  Image a = sinePattern(32, 32, 0, 0);
  feed(o.inputs.image, a);
  o();
  o.outputs.iterations = -1;
  o.inputs.image.texture.changed = false;
  o(); // must be a no-op
  CHECK(o.outputs.iterations.value == -1);
}

// ======================================================================== dimension change
TEST_CASE("HSflow: a mid-stream dimension change resets cleanly", "[hsflow]")
{
  cv::HornSchunck o;
  o.inputs.maxiter.value = 50;

  Image a = sinePattern(64, 64, 0, 0);
  Image b = sinePattern(64, 64, 1, 0);
  feed(o.inputs.image, a);
  o();
  feed(o.inputs.image, b);
  o();
  CHECK(interior(o, 64, 64, 8, 1.f, 0.f).mean_u > 0.9);

  // Now a different size: treated like a first frame -> zero flow, zero iterations, and the
  // output textures must have been resized (no stale 64x64 read).
  Image c = sinePattern(32, 32, 0, 0);
  feed(o.inputs.image, c);
  o();
  CHECK(o.outputs.dx.texture.width == 32);
  CHECK(o.outputs.dx.texture.height == 32);
  CHECK(o.outputs.dy.texture.width == 32);
  CHECK(o.outputs.iterations.value == 0);
  for(int y = 0; y < 32; ++y)
    for(int x = 0; x < 32; ++x)
    {
      REQUIRE(fx(o, x, y) == 0.f);
      REQUIRE(fy(o, x, y) == 0.f);
    }

  // ... and the next frame at the new size works normally.
  Image d = sinePattern(32, 32, 0, 1);
  feed(o.inputs.image, d);
  o();
  CHECK(o.outputs.iterations.value == 50);
  auto s = interior(o, 32, 32, 8, 0.f, 1.f);
  CHECK(s.mean_v > 0.85);
  CHECK(std::abs(s.mean_u) < 0.05);

  // A non-square size too (catches a W/H swap in the buffer bookkeeping).
  Image e = sinePattern(48, 24, 0, 0);
  feed(o.inputs.image, e);
  o();
  CHECK(o.outputs.dx.texture.width == 48);
  CHECK(o.outputs.dx.texture.height == 24);
  CHECK(o.outputs.iterations.value == 0);
  Image f = sinePattern(48, 24, 1, 0);
  feed(o.inputs.image, f);
  o();
  REQUIRE(allFinite(o, 48, 24));
  CHECK(interior(o, 48, 24, 6, 1.f, 0.f).mean_u > 0.85);
}

// ============================================================ usePrevious == 0 (cv.jit quirk)
//
// cv.jit hardcodes cvCalcOpticalFlowHS's `usePrevious` argument to 0, so every frame
// restarts the iteration from a zero flow field — the previous frame's *flow* is never a
// warm start (only the previous frame's *image* is kept). On a constant-velocity sequence
// with a low maxiter this is directly observable: with the warm start off, every frame of
// the sequence returns the SAME under-converged estimate forever; with it on, the estimate
// climbs towards the true displacement. A "helpful" implementation that silently
// warm-started would fail the first section.
TEST_CASE("HSflow: no warm start by default (cv.jit usePrevious = 0)", "[hsflow]")
{
  const int W = 64, H = 64, M = 8;
  std::vector<Image> seq;
  for(int k = 0; k < 6; ++k)
    seq.push_back(sinePattern(W, H, static_cast<float>(k), 0)); // +1 px per frame

  SECTION("default: identical result on every frame of a constant-velocity sequence")
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 5;
    CHECK(o.inputs.use_previous.value == false); // cv.jit parity is the default
    feed(o.inputs.image, seq[0]);
    o();
    double first = 0;
    for(int k = 1; k < 6; ++k)
    {
      feed(o.inputs.image, seq[k]);
      o();
      const double m = interior(o, W, H, M, 1.f, 0.f).mean_u;
      if(k == 1)
        first = m;
      else
        REQUIRE(m == Approx(first).epsilon(1e-6)); // no accumulation whatsoever
    }
    // 5 sweeps from zero. Measured 0.658 here, but that is a converged floating-point
    // measurement of an iterative solver: the exact figure moves with the compiler, the
    // optimisation level and the FMA contraction settings, so a +/-1.5% REQUIRE on it is a
    // trap rather than a specification. What the test actually needs to pin down is the
    // TREND — 5 sweeps must be well past the single-sweep normal flow (0.485) and still
    // clearly short of the true displacement (1.0), because that gap is the whole point of
    // iterating. The per-sweep progression itself is asserted in "maxiter strictly
    // improves the estimate".
    CHECK(first > 0.55);
    CHECK(first < 0.75);
  }

  SECTION("Use previous: the estimate improves frame after frame")
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 5;
    o.inputs.use_previous.value = true;
    feed(o.inputs.image, seq[0]);
    o();
    double prev = -1e30;
    for(int k = 1; k < 6; ++k)
    {
      feed(o.inputs.image, seq[k]);
      o();
      const double m = interior(o, W, H, M, 1.f, 0.f).mean_u;
      REQUIRE(m > prev); // strictly improving towards 1.0
      prev = m;
    }
    CHECK(prev > 0.9); // measured 0.937 at frame 5
    CHECK(prev < 1.02);
  }
}

// ================================================================ the r32f [0,1] contract
//
// score converts every r32f texture output to RGBA8 by interpreting the float as if it were
// already in [0,1] (Crousti/TextureConversion.hpp, case QRhiTexture::R32F:
// `gray = qBound(0, int(v*255.f), 255)`). A flow field written raw would therefore arrive at
// the next object with every leftward/upward component crushed to black and everything past
// 1 px/frame saturated to white. These tests pin the encoding that fixes it.
TEST_CASE("HSflow: the flow textures obey the r32f [0,1] contract", "[hsflow][contract]")
{
  const int W = 48, H = 48;
  cv::HornSchunck o;
  o.inputs.maxiter.value = 20;

  SECTION("first frame: zero flow is exactly the middle of the bipolar swing")
  {
    Image a = sinePattern(W, H, 0, 0);
    feed(o.inputs.image, a);
    o();
    REQUIRE(r32f_in_unit_range(o.outputs.dx));
    REQUIRE(r32f_in_unit_range(o.outputs.dy));
    const float* ex = reinterpret_cast<const float*>(o.outputs.dx.texture.bytes);
    for(int i = 0; i < W * H; ++i)
      REQUIRE(ex[i] == 0.5f); // not 0.0 — 0.0 means "-Flow scale"
    CHECK(o.outputs.flow_scale.value > 0.f);
    CHECK(o.outputs.clipped.value == false);
  }

  SECTION("a real field stays in range and both signs survive")
  {
    Image a = sinePattern(W, H, 0, 0);
    Image b = sinePattern(W, H, -1, 1); // left and down: u < 0, v > 0
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    REQUIRE(r32f_in_unit_range(o.outputs.dx));
    REQUIRE(r32f_in_unit_range(o.outputs.dy));
    CHECK(o.outputs.clipped.value == false); // auto scale never clips

    auto s = interior(o, W, H, 8, -1.f, 1.f);
    CHECK(s.mean_u < -0.5); // a negative component that actually came back negative
    CHECK(s.mean_v > 0.5);

    // Raw encoded values: negative u sits BELOW 0.5, positive v ABOVE it. Under the old
    // "write raw px/frame" behaviour the u plane was < 0 and arrived as pure black.
    const float* ex = reinterpret_cast<const float*>(o.outputs.dx.texture.bytes);
    const float* ey = reinterpret_cast<const float*>(o.outputs.dy.texture.bytes);
    const std::size_t c = static_cast<std::size_t>(H / 2) * W + W / 2;
    CHECK(ex[c] < 0.5f);
    CHECK(ex[c] > 0.0f);
    CHECK(ey[c] > 0.5f);
  }
}

TEST_CASE("HSflow: a fixed Flow scale clips observably", "[hsflow][contract]")
{
  const int W = 48, H = 48;
  Image a = sinePattern(W, H, 0, 0);
  Image b = sinePattern(W, H, 1, 0);

  SECTION("auto (default) never clips")
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 50;
    CHECK(o.inputs.flow_scale.value == 0.f); // auto is the default
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    CHECK(o.outputs.clipped.value == false);
    // Auto scale == the peak component, so at least one pixel sits exactly at an end.
    const float* ex = reinterpret_cast<const float*>(o.outputs.dx.texture.bytes);
    const float* ey = reinterpret_cast<const float*>(o.outputs.dy.texture.bytes);
    float lo = 1.f, hi = 0.f;
    for(int i = 0; i < W * H; ++i)
    {
      lo = std::min({lo, ex[i], ey[i]});
      hi = std::max({hi, ex[i], ey[i]});
    }
    CHECK((lo == Approx(0.f).margin(1e-6) || hi == Approx(1.f).margin(1e-6)));
  }

  SECTION("a scale far below the real motion clips and says so")
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 50;
    o.inputs.flow_scale.value = 0.05f; // the true flow is ~1 px/frame
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    CHECK(o.outputs.clipped.value == true);
    CHECK(o.outputs.flow_scale.value == Approx(0.05f));
    REQUIRE(r32f_in_unit_range(o.outputs.dx)); // still contract-clean, just saturated
    // Everything past +/-0.05 pins to the ends, so the decoded interior is +0.05, not 1.
    CHECK(interior(o, W, H, 8, 1.f, 0.f).mean_u == Approx(0.05).margin(1e-3));
  }

  SECTION("a generous fixed scale does not clip and measures correctly")
  {
    cv::HornSchunck o;
    o.inputs.maxiter.value = 50;
    o.inputs.flow_scale.value = 8.f;
    feed(o.inputs.image, a);
    o();
    feed(o.inputs.image, b);
    o();
    CHECK(o.outputs.clipped.value == false);
    CHECK(o.outputs.flow_scale.value == Approx(8.f));
    CHECK(interior(o, W, H, 8, 1.f, 0.f).mean_u > 0.95);
  }
}

// ======================================================= HSflow -> CartoPol, the real patch
//
// cv.jit.HSflow -> cv.jit.cartopol is THE documented use of these two objects, and it was
// completely broken: HSflow wrote raw signed px/frame into an r32f port while CartoPol read
// an RGBA8 bipolar field, so score's conversion clamped every negative component to black
// (== -Range) and saturated anything past 1 px/frame. Every hop below goes through
// cvtest::connect_r32f, i.e. score's real conversion — no hand-encoding anywhere.
TEST_CASE("HSflow -> CartoPol recovers the direction of motion", "[hsflow][cartopol]")
{
  const int W = 64, H = 64, M = 12;
  Image ref = sinePattern(W, H, 0, 0);

  // (dx, dy) -> expected phase of atan2(v, u). The two NEGATIVE cases are the ones the old
  // code could not represent at all.
  struct Case
  {
    float sx, sy, phase;
    const char* name;
  };
  const Case cases[] = {
      {1.f, 0.f, 0.f, "right"},
      {0.f, 1.f, float(kPi / 2), "down"},
      {-1.f, 0.f, float(kPi), "left"},   // |phase| == pi, sign of atan2 is immaterial
      {0.f, -1.f, float(-kPi / 2), "up"},
  };

  for(const auto& c : cases)
  {
    INFO(c.name);
    cv::HornSchunck hs;
    hs.inputs.maxiter.value = 50;
    Image shifted = sinePattern(W, H, c.sx, c.sy);
    feed(hs.inputs.image, ref);
    hs();
    feed(hs.inputs.image, shifted);
    hs();

    cv::CartoPol cp;
    Image gx, gy;
    connect_r32f(hs.outputs.dx, cp.inputs.x, gx);
    connect_r32f(hs.outputs.dy, cp.inputs.y, gy);
    // The one wire that makes the chain agree: HSflow's Flow scale -> CartoPol's Range.
    cp.inputs.range.value = hs.outputs.flow_scale.value;
    cp.inputs.is_signed.value = true;
    cp();
    REQUIRE(cp.outputs.amplitude.texture.bytes != nullptr);
    REQUIRE(r32f_in_unit_range(cp.outputs.amplitude));
    REQUIRE(r32f_in_unit_range(cp.outputs.phase));

    // Mean amplitude and mean phase over the interior, in world units.
    double sa = 0, sfx = 0, sfy = 0;
    int n = 0;
    for(int y = M; y < H - M; ++y)
      for(int x = M; x < W - M; ++x)
      {
        const std::size_t i = static_cast<std::size_t>(y) * W + x;
        const double a = cp.outputs.amplitude.texture.bytes[i]
                         * cp.outputs.amplitude_scale.value;
        const double p
            = cv::polar_codec::decode_phase(cp.outputs.phase.texture.bytes[i]);
        sa += a;
        sfx += a * std::cos(p); // average the vectors, not the angles (no wrap issues)
        sfy += a * std::sin(p);
        ++n;
      }
    const double mean_a = sa / n;
    const double mean_phase = std::atan2(sfy / n, sfx / n);

    // 50 sweeps recover ~0.99 px/frame; one 8-bit hop costs at most 2*scale/255.
    CHECK(mean_a == Approx(1.0).margin(0.08));
    // Compare angles modulo 2pi.
    double d = mean_phase - c.phase;
    while(d > kPi)
      d -= 2 * kPi;
    while(d < -kPi)
      d += 2 * kPi;
    CHECK(std::abs(d) < 0.06);
  }
}
