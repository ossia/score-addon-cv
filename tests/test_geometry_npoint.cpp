// Tests for the arbitrary-N (list-input) geometry solvers: Homography and SolvePnP.
//
// test_geometry.cpp covers the legacy fixed-spinbox behaviour and MUST keep passing
// unchanged; everything here exercises the new "Points" list inputs, the RANSAC modes and
// the axis-angle / rotation-list outputs. Test names are prefixed "N-point" because
// tests/run_standalone.sh links every test TU into a single Catch2 binary, where test-case
// names have to be globally unique.
//
// ---------------------------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------------------------
// Homography. A projective transform is applied as
//     [u' v' w']^T = G * [x y 1]^T,   (u, v) = (u'/w', v'/w')
// with G row-major and G(2,2) normalised to 1. The object normalises its own output the
// same way, so for noise-free correspondences in general position the recovered matrix must
// equal G element-by-element, not merely up to scale. Every "exact recovery" case below
// therefore asserts on the 9 coefficients directly, which a scale-sloppy implementation
// would fail.
//
// The ground truth used throughout the exact cases is
//     G = [ 1.2   0.1   0.3
//          -0.2   0.9   0.4
//           0.05 -0.03  1.0 ]
// whose third row keeps w' = 1 + 0.05x - 0.03y >= 0.94 > 0 over every point set used, so
// no point is mapped through or past the horizon line.
//
// SolvePnP. Ground truth poses are generated as R = Rz(g) * Ry(b) * Rx(a) with a
// translation that puts the whole object at positive depth, and image points are produced
// with the object's own pinhole model
//     u = fx * (R X + t).x / (R X + t).z + cx,     v = fy * ... + cy.
// Intrinsics are deliberately realistic (fx = fy = 800, cx = 320, cy = 240) so that the
// RANSAC "reprojection error" threshold keeps its documented pixel meaning (cv.jit hard
// codes 8.0 pixels).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <CV/Cpu/Homography.hpp>
#include <CV/Cpu/SolvePnP.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using Catch::Approx;

namespace
{
// ------------------------------------------------------------------ homography helpers
struct Mat3
{
  std::array<double, 9> m; // row-major

  std::pair<double, double> apply(double x, double y) const
  {
    const double u = m[0] * x + m[1] * y + m[2];
    const double v = m[3] * x + m[4] * y + m[5];
    const double w = m[6] * x + m[7] * y + m[8];
    return {u / w, v / w};
  }
};

const Mat3 G_exact{{1.2, 0.1, 0.3, -0.2, 0.9, 0.4, 0.05, -0.03, 1.0}};

// Build the object's list input from a set of source points mapped through `g`.
std::vector<cv::homography_correspondence>
makeCorr(const Mat3& g, const std::vector<std::pair<double, double>>& src)
{
  std::vector<cv::homography_correspondence> out;
  out.reserve(src.size());
  for(auto& s : src)
  {
    auto d = g.apply(s.first, s.second);
    out.push_back(
        {static_cast<float>(s.first), static_cast<float>(s.second),
         static_cast<float>(d.first), static_cast<float>(d.second)});
  }
  return out;
}

Mat3 asMat3(const std::array<float, 9>& a)
{
  Mat3 r{};
  for(int i = 0; i < 9; ++i)
    r.m[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)];
  return r;
}

// Largest forward transfer error ||H*s - d|| over a correspondence set.
double maxTransfer(const Mat3& h, const std::vector<cv::homography_correspondence>& c)
{
  double worst = 0.0;
  for(auto& p : c)
  {
    auto [x, y] = h.apply(p.sx, p.sy);
    worst = std::max(worst, std::hypot(x - p.dx, y - p.dy));
  }
  return worst;
}

double rmsTransfer(const Mat3& h, const std::vector<cv::homography_correspondence>& c)
{
  double s = 0.0;
  for(auto& p : c)
  {
    auto [x, y] = h.apply(p.sx, p.sy);
    s += (x - p.dx) * (x - p.dx) + (y - p.dy) * (y - p.dy);
  }
  return std::sqrt(s / static_cast<double>(c.size()));
}

// The three point sets used by the exact-recovery cases. All are in general position
// (no three points collinear, no repeats).
const std::vector<std::pair<double, double>> pts4{{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};

const std::vector<std::pair<double, double>> pts6{
    {0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}, {0.3, 0.7}, {0.8, 0.25}};

const std::vector<std::pair<double, double>> pts12{
    {0., 0.},    {1., 0.},    {2., 0.1},  {3., 0.2},  {0.1, 1.},  {1.2, 1.1},
    {2.3, 0.9},  {3.1, 1.2},  {0.2, 2.},  {1.1, 2.2}, {2.4, 1.9}, {3.3, 2.1}};

// ------------------------------------------------------------------- solvePnP helpers
struct Pose
{
  std::array<double, 9> R; // row-major
  std::array<double, 3> t;
};

std::array<double, 9> mul(const std::array<double, 9>& a, const std::array<double, 9>& b)
{
  std::array<double, 9> r{};
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
    {
      double s = 0;
      for(int k = 0; k < 3; ++k)
        s += a[static_cast<std::size_t>(i * 3 + k)] * b[static_cast<std::size_t>(k * 3 + j)];
      r[static_cast<std::size_t>(i * 3 + j)] = s;
    }
  return r;
}

std::array<double, 9> rotX(double a)
{
  return {1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a)};
}
std::array<double, 9> rotY(double a)
{
  return {std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)};
}
std::array<double, 9> rotZ(double a)
{
  return {std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1};
}

std::array<double, 3> applyR(const std::array<double, 9>& R, const std::array<double, 3>& v)
{
  return {
      R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
      R[6] * v[0] + R[7] * v[1] + R[8] * v[2]};
}

struct Cam
{
  double fx, fy, cx, cy;
};

const Cam cam{800., 800., 320., 240.};

// Non-coplanar object clouds. 4 points = a tetrahedron; the larger sets extend it while
// keeping every point well away from any common plane.
const std::vector<std::array<double, 3>> obj4{
    {0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};

const std::vector<std::array<double, 3>> obj6{
    {0., 0., 0.},   {1., 0., 0.},   {0., 1., 0.},
    {0., 0., 1.},   {1., 1., 0.4},  {0.4, 0.6, 1.2}};

const std::vector<std::array<double, 3>> obj10{
    {0., 0., 0.},      {1., 0., 0.},      {0., 1., 0.},    {0., 0., 1.},
    {1., 1., 0.4},     {0.4, 0.6, 1.2},   {-0.5, 0.3, 0.8}, {0.7, -0.4, 0.5},
    {1.3, 0.8, -0.3},  {-0.2, 1.1, 0.15}};

std::vector<cv::pnp_correspondence>
project(const std::vector<std::array<double, 3>>& obj, const Pose& p)
{
  std::vector<cv::pnp_correspondence> out;
  out.reserve(obj.size());
  for(auto& X : obj)
  {
    auto Pc = applyR(p.R, X);
    Pc[0] += p.t[0];
    Pc[1] += p.t[1];
    Pc[2] += p.t[2];
    REQUIRE(Pc[2] > 0.1); // sanity: the generator must keep the object in front
    out.push_back(
        {static_cast<float>(X[0]), static_cast<float>(X[1]), static_cast<float>(X[2]),
         static_cast<float>(cam.fx * Pc[0] / Pc[2] + cam.cx),
         static_cast<float>(cam.fy * Pc[1] / Pc[2] + cam.cy)});
  }
  return out;
}

void setIntrinsics(cv::SolvePnP& p)
{
  p.inputs.fx.value = static_cast<float>(cam.fx);
  p.inputs.fy.value = static_cast<float>(cam.fy);
  p.inputs.cx.value = static_cast<float>(cam.cx);
  p.inputs.cy.value = static_cast<float>(cam.cy);
}

// Rotation matrix (row-major) from a unit quaternion (x, y, z, w).
std::array<double, 9> fromQuat(const std::array<float, 4>& q)
{
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  return {
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
      2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
}

// Rotation matrix (row-major) from the object's axis-angle output
// (theta in DEGREES, ax, ay, az) via Rodrigues.
std::array<double, 9> fromAxisAngle(const std::array<float, 4>& aa)
{
  const double th = aa[0] * cv::cv_pi / 180.0;
  const double x = aa[1], y = aa[2], z = aa[3];
  const double c = std::cos(th), s = std::sin(th), C = 1 - c;
  return {
      x * x * C + c,     x * y * C - z * s, x * z * C + y * s,
      y * x * C + z * s, y * y * C + c,     y * z * C - x * s,
      z * x * C - y * s, z * y * C + x * s, z * z * C + c};
}

double maxAbsDiff(const std::array<double, 9>& a, const std::array<double, 9>& b)
{
  double d = 0;
  for(int i = 0; i < 9; ++i)
    d = std::max(
        d, std::abs(a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]));
  return d;
}
}

// =========================================================================== Homography

TEST_CASE("N-point homography: exact recovery from 4, 6 and 12 correspondences", "[npoint][homography]")
{
  for(const auto* pts : {&pts4, &pts6, &pts12})
  {
    cv::Homography h;
    h.inputs.points.value = makeCorr(G_exact, *pts);
    h();

    INFO("N = " << pts->size());
    REQUIRE(h.outputs.valid.value);

    // The matrix itself, not just its action: both are normalised to (2,2) == 1.
    for(int i = 0; i < 9; ++i)
      CHECK(
          h.outputs.matrix.value[static_cast<std::size_t>(i)]
          == Approx(G_exact.m[static_cast<std::size_t>(i)]).margin(1e-4));

    // Every correspondence is reproduced.
    CHECK(maxTransfer(asMat3(h.outputs.matrix.value), h.inputs.points.value) < 1e-4);

    // Plain least-squares reports the whole set as inliers.
    CHECK(h.outputs.inliers.value == static_cast<int>(pts->size()));
    REQUIRE(h.outputs.inlier_mask.value.size() == pts->size());
    for(int v : h.outputs.inlier_mask.value)
      CHECK(v == 1);
  }
}

TEST_CASE("N-point homography: over-determined noisy set converges to least squares", "[npoint][homography]")
{
  // 12 correspondences, each destination perturbed by a deterministic +/-0.01 pattern.
  // Bound: the noise is at most 0.01 per coordinate on destinations spanning ~4 units,
  // and the DLT is a (normalised) least-squares fit, so
  //   (a) the fitted matrix must stay within 0.02 of ground truth element-wise, and
  //   (b) its RMS transfer residual over the NOISY data must not exceed the ground-truth
  //       matrix's own RMS residual by more than 1% -- i.e. it really is the better fit,
  //       which is exactly what a non-least-squares (e.g. "use the first 4 points")
  //       implementation would fail.
  auto corr = makeCorr(G_exact, pts12);
  const double eps = 0.01;
  for(std::size_t i = 0; i < corr.size(); ++i)
  {
    const double sx = ((i % 3) - 1.0);       // -1, 0, +1 ...
    const double sy = (((i / 3) % 3) - 1.0); // ... deterministic, zero-mean-ish
    corr[i].dx += static_cast<float>(eps * sx);
    corr[i].dy += static_cast<float>(eps * sy);
  }

  cv::Homography h;
  h.inputs.points.value = corr;
  h();
  REQUIRE(h.outputs.valid.value);

  const Mat3 fit = asMat3(h.outputs.matrix.value);
  for(int i = 0; i < 9; ++i)
    CHECK(
        fit.m[static_cast<std::size_t>(i)]
        == Approx(G_exact.m[static_cast<std::size_t>(i)]).margin(0.02));

  const double rmsFit = rmsTransfer(fit, corr);
  const double rmsTruth = rmsTransfer(G_exact, corr);
  INFO("rms fit = " << rmsFit << ", rms truth = " << rmsTruth);
  CHECK(rmsFit <= rmsTruth * 1.01);
  // And it does not simply interpolate the noise away: residuals are of the noise order.
  CHECK(rmsFit > 0.0);
  CHECK(rmsFit < 0.05);
}

TEST_CASE("N-point homography: degenerate configurations are rejected", "[npoint][homography]")
{
  SECTION("collinear sources")
  {
    // 6 source points on the line y = 2x; destinations in general position.
    // No homography is determined by this, and a textbook "just take the smallest
    // singular vector" implementation happily returns garbage instead of failing.
    std::vector<cv::homography_correspondence> c;
    const std::array<std::pair<float, float>, 6> dst{
        {{0.f, 0.f}, {2.f, 0.3f}, {3.f, 1.7f}, {1.f, 2.4f}, {2.5f, 1.f}, {0.4f, 1.9f}}};
    for(int i = 0; i < 6; ++i)
    {
      const float x = 0.5f * static_cast<float>(i);
      c.push_back(
          {x, 2.f * x, dst[static_cast<std::size_t>(i)].first,
           dst[static_cast<std::size_t>(i)].second});
    }

    cv::Homography h;
    h.inputs.points.value = c;
    h();
    CHECK_FALSE(h.outputs.valid.value);
    CHECK(h.outputs.inliers.value == 0);
  }

  SECTION("all source points identical")
  {
    std::vector<cv::homography_correspondence> c{
        {1.f, 1.f, 0.f, 0.f},
        {1.f, 1.f, 1.f, 0.f},
        {1.f, 1.f, 1.f, 1.f},
        {1.f, 1.f, 0.f, 1.f}};
    cv::Homography h;
    h.inputs.points.value = c;
    h();
    CHECK_FALSE(h.outputs.valid.value);
  }

  SECTION("fewer than 4 correspondences")
  {
    cv::Homography h;
    h.inputs.points.value = makeCorr(G_exact, {{0., 0.}, {1., 0.}, {1., 1.}});
    h();
    CHECK_FALSE(h.outputs.valid.value);
    CHECK(h.outputs.inliers.value == 0);
  }
}

TEST_CASE("N-point homography: RANSAC survives 30% gross outliers where plain LS fails", "[npoint][homography]")
{
  // 20 correspondences in pixel units, 6 of them (30%) with the destination displaced by
  // 60-140 px. Ground truth is a mild projective warp of a 400x300 frame.
  const Mat3 Gpx{{1.05, 0.08, 12.0, -0.06, 0.98, -7.0, 0.0002, -0.0001, 1.0}};

  std::vector<std::pair<double, double>> src;
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 4; ++j)
      src.push_back({20.0 + 90.0 * i + 7.0 * j, 15.0 + 90.0 * j - 5.0 * i});
  REQUIRE(src.size() == 20);

  auto clean = makeCorr(Gpx, src);
  auto corrupted = clean;

  // Planted outliers: indices spread through the set, offsets in different directions so
  // they cannot cancel out in a least-squares fit.
  const std::array<int, 6> bad{1, 4, 8, 11, 15, 18};
  const std::array<std::pair<float, float>, 6> off{
      {{110.f, -70.f}, {-95.f, 80.f}, {130.f, 60.f}, {-120.f, -60.f}, {70.f, 125.f},
       {-140.f, 45.f}}};
  for(std::size_t k = 0; k < bad.size(); ++k)
  {
    auto& c = corrupted[static_cast<std::size_t>(bad[k])];
    c.dx += off[k].first;
    c.dy += off[k].second;
  }

  // --- plain least squares: demonstrably wrong.
  cv::Homography ls;
  ls.inputs.points.value = corrupted;
  ls.inputs.method.value = cv::HomographyMethod::LeastSquares;
  ls();
  REQUIRE(ls.outputs.valid.value);
  const double lsWorst = maxTransfer(asMat3(ls.outputs.matrix.value), clean);
  INFO("plain LS worst transfer error on the clean data: " << lsWorst << " px");
  CHECK(lsWorst > 5.0); // dragged far off by the outliers
  CHECK(ls.outputs.inliers.value == 20); // LS mode never marks anything an outlier

  // --- RANSAC: recovers the truth and identifies exactly the planted outliers.
  cv::Homography rs;
  rs.inputs.points.value = corrupted;
  rs.inputs.method.value = cv::HomographyMethod::RANSAC;
  rs.inputs.ransac_threshold.value = 2.0f;
  rs.inputs.ransac_iterations.value = 2000;
  rs();
  REQUIRE(rs.outputs.valid.value);

  const Mat3 rh = asMat3(rs.outputs.matrix.value);
  const double rsWorst = maxTransfer(rh, clean);
  INFO("RANSAC worst transfer error on the clean data: " << rsWorst << " px");
  CHECK(rsWorst < 0.5);
  CHECK(rsWorst < lsWorst / 10.0); // both directions of the comparison asserted

  for(int i = 0; i < 9; ++i)
    CHECK(
        rh.m[static_cast<std::size_t>(i)]
        == Approx(Gpx.m[static_cast<std::size_t>(i)]).margin(1e-3));

  CHECK(rs.outputs.inliers.value == 14);
  REQUIRE(rs.outputs.inlier_mask.value.size() == 20);
  for(int i = 0; i < 20; ++i)
  {
    const bool planted
        = std::find(bad.begin(), bad.end(), i) != bad.end();
    INFO("point " << i);
    CHECK(rs.outputs.inlier_mask.value[static_cast<std::size_t>(i)] == (planted ? 0 : 1));
  }

  // Determinism: the RNG is seeded per call, so a second evaluation is identical.
  const auto first = rs.outputs.matrix.value;
  rs();
  CHECK(rs.outputs.matrix.value == first);
}

TEST_CASE("N-point homography: an empty list falls back to the spinboxes", "[npoint][homography]")
{
  // Exactly the configuration of "Homography maps the 4 correspondences (affine)" in
  // test_geometry.cpp: the fallback must reproduce it bit for bit.
  cv::Homography spin;
  spin.inputs.src0.value = {0.f, 0.f};
  spin.inputs.src1.value = {1.f, 0.f};
  spin.inputs.src2.value = {1.f, 1.f};
  spin.inputs.src3.value = {0.f, 1.f};
  spin.inputs.dst0.value = {2.f, 3.f};
  spin.inputs.dst1.value = {4.f, 3.f};
  spin.inputs.dst2.value = {4.f, 7.f};
  spin.inputs.dst3.value = {0.f, 7.f};
  REQUIRE(spin.inputs.points.value.empty());
  spin();

  REQUIRE(spin.outputs.valid.value);
  const Mat3 hm = asMat3(spin.outputs.matrix.value);
  {
    auto [x, y] = hm.apply(0., 0.);
    CHECK(x == Approx(2.).margin(1e-3));
    CHECK(y == Approx(3.).margin(1e-3));
  }
  {
    auto [x, y] = hm.apply(1., 1.);
    CHECK(x == Approx(4.).margin(1e-3));
    CHECK(y == Approx(7.).margin(1e-3));
  }
  CHECK(spin.outputs.inliers.value == 4);

  // The same 4 correspondences through the list input give the same matrix.
  cv::Homography lst;
  lst.inputs.points.value
      = {{0.f, 0.f, 2.f, 3.f},
         {1.f, 0.f, 4.f, 3.f},
         {1.f, 1.f, 4.f, 7.f},
         {0.f, 1.f, 0.f, 7.f}};
  lst();
  REQUIRE(lst.outputs.valid.value);
  for(int i = 0; i < 9; ++i)
    CHECK(
        lst.outputs.matrix.value[static_cast<std::size_t>(i)]
        == Approx(spin.outputs.matrix.value[static_cast<std::size_t>(i)]).margin(1e-5));

  // A non-empty list wins over the spinboxes (the toggles are ignored entirely).
  cv::Homography mix = spin;
  mix.inputs.use4.value = true;
  mix.inputs.src4.value = {5.f, 5.f};
  mix.inputs.dst4.value = {99.f, 99.f}; // wildly inconsistent, must be ignored
  mix.inputs.points.value = makeCorr(G_exact, pts6);
  mix();
  REQUIRE(mix.outputs.valid.value);
  for(int i = 0; i < 9; ++i)
    CHECK(
        mix.outputs.matrix.value[static_cast<std::size_t>(i)]
        == Approx(G_exact.m[static_cast<std::size_t>(i)]).margin(1e-4));
}

// ============================================================================= SolvePnP

TEST_CASE("N-point PnP: known pose recovered from 4, 6 and 10 correspondences", "[npoint][pnp]")
{
  Pose gt;
  gt.R = mul(mul(rotZ(0.26), rotY(0.35)), rotX(-0.17));
  gt.t = {0.15, -0.22, 4.0};

  for(const auto* o : {&obj4, &obj6, &obj10})
  {
    cv::SolvePnP p;
    setIntrinsics(p);
    p.inputs.points.value = project(*o, gt);
    p();

    INFO("N = " << o->size());
    REQUIRE(p.outputs.valid.value);

    CHECK(p.outputs.translation.value.x == Approx(gt.t[0]).margin(2e-3));
    CHECK(p.outputs.translation.value.y == Approx(gt.t[1]).margin(2e-3));
    CHECK(p.outputs.translation.value.z == Approx(gt.t[2]).margin(5e-3));

    std::array<double, 9> got{};
    for(int i = 0; i < 9; ++i)
      got[static_cast<std::size_t>(i)]
          = p.outputs.matrix.value[static_cast<std::size_t>(i)];
    INFO("max |R - R_gt| = " << maxAbsDiff(got, gt.R));
    CHECK(maxAbsDiff(got, gt.R) < 2e-3);

    CHECK(p.outputs.inliers.value == static_cast<int>(o->size()));
    REQUIRE(p.outputs.inlier_mask.value.size() == o->size());
    for(int v : p.outputs.inlier_mask.value)
      CHECK(v == 1);
  }
}

TEST_CASE("N-point PnP: rotation matrix is orthonormal and matches the quaternion", "[npoint][pnp]")
{
  Pose gt;
  gt.R = mul(mul(rotZ(-0.4), rotY(0.6)), rotX(0.25));
  gt.t = {-0.3, 0.4, 5.0};

  cv::SolvePnP p;
  setIntrinsics(p);
  p.inputs.points.value = project(obj10, gt);
  p();
  REQUIRE(p.outputs.valid.value);

  std::array<double, 9> R{};
  for(int i = 0; i < 9; ++i)
    R[static_cast<std::size_t>(i)] = p.outputs.matrix.value[static_cast<std::size_t>(i)];

  // R^T R == I (row-major indexing: R(r,c) = R[r*3+c]).
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
    {
      double s = 0;
      for(int k = 0; k < 3; ++k)
        s += R[static_cast<std::size_t>(k * 3 + i)] * R[static_cast<std::size_t>(k * 3 + j)];
      CHECK(s == Approx(i == j ? 1.0 : 0.0).margin(1e-5));
    }

  const double det = R[0] * (R[4] * R[8] - R[5] * R[7])
                     - R[1] * (R[3] * R[8] - R[5] * R[6])
                     + R[2] * (R[3] * R[7] - R[4] * R[6]);
  CHECK(det == Approx(1.0).margin(1e-5));

  // Consistent with the quaternion output.
  CHECK(maxAbsDiff(fromQuat(p.outputs.quat.value), R) < 1e-5);
  // And it is the ground-truth rotation.
  CHECK(maxAbsDiff(R, gt.R) < 2e-3);

  // The "Rotation" list port with Format = Matrix is cv.jit's COLUMN-major ordering,
  // i.e. the transpose of the row-major "Rotation matrix" port. Asserting both here is
  // what pins the two conventions apart.
  p.inputs.format.value = cv::PnpRotationFormat::Matrix;
  p();
  REQUIRE(p.outputs.rotation.value.size() == 9);
  for(int col = 0; col < 3; ++col)
    for(int row = 0; row < 3; ++row)
      CHECK(
          p.outputs.rotation.value[static_cast<std::size_t>(col * 3 + row)]
          == Approx(p.outputs.matrix.value[static_cast<std::size_t>(row * 3 + col)])
                 .margin(1e-6));
}

TEST_CASE("N-point PnP: rotation list port is sized per the format selector", "[npoint][pnp]")
{
  Pose gt;
  gt.R = mul(rotY(0.5), rotX(0.1));
  gt.t = {0.1, 0.1, 4.5};

  cv::SolvePnP p;
  setIntrinsics(p);
  p.inputs.points.value = project(obj6, gt);

  // Default is Euler, exactly like cv.jit.unproject.
  p();
  REQUIRE(p.outputs.valid.value);
  REQUIRE(p.outputs.rotation.value.size() == 3);
  CHECK(p.outputs.rotation.value[0] == Approx(p.outputs.euler.value.x).margin(1e-5));
  CHECK(p.outputs.rotation.value[1] == Approx(p.outputs.euler.value.y).margin(1e-5));
  CHECK(p.outputs.rotation.value[2] == Approx(p.outputs.euler.value.z).margin(1e-5));

  p.inputs.format.value = cv::PnpRotationFormat::Quaternion;
  p();
  REQUIRE(p.outputs.rotation.value.size() == 4);
  for(int i = 0; i < 4; ++i)
    CHECK(
        p.outputs.rotation.value[static_cast<std::size_t>(i)]
        == Approx(p.outputs.quat.value[static_cast<std::size_t>(i)]).margin(1e-6));

  p.inputs.format.value = cv::PnpRotationFormat::AxisAngle;
  p();
  REQUIRE(p.outputs.rotation.value.size() == 4);
  for(int i = 0; i < 4; ++i)
    CHECK(
        p.outputs.rotation.value[static_cast<std::size_t>(i)]
        == Approx(p.outputs.axis_angle.value[static_cast<std::size_t>(i)]).margin(1e-6));

  p.inputs.format.value = cv::PnpRotationFormat::Matrix;
  p();
  CHECK(p.outputs.rotation.value.size() == 9);
}

TEST_CASE("N-point PnP: axis-angle round-trips to the quaternion rotation", "[npoint][pnp]")
{
  // Several poses, including a near-identity one where sqrt(1 - w*w) underflows and
  // cv.jit's `fpclassify(theta) != FP_NORMAL` guard is the only thing standing between
  // the axis and a division by (almost) zero.
  struct Case
  {
    const char* name;
    std::array<double, 9> R;
    bool appreciable; // is the rotation large enough for the axis to be meaningful?
  };
  const std::array<Case, 5> cases{
      {{"30 deg about Z", rotZ(30.0 * cv::cv_pi / 180.0), true},
       {"120 deg about (1,1,1)", mul(mul(rotZ(1.1), rotY(0.9)), rotX(1.3)), true},
       {"180-ish about Y", rotY(179.0 * cv::cv_pi / 180.0), true},
       {"near identity (1e-6 rad)", rotY(1e-6), false},
       {"exact identity", rotX(0.0), false}}};

  for(const auto& c : cases)
  {
    Pose gt;
    gt.R = c.R;
    gt.t = {0.05, -0.05, 4.0};

    cv::SolvePnP p;
    setIntrinsics(p);
    p.inputs.points.value = project(obj10, gt);
    p();

    INFO(c.name);
    REQUIRE(p.outputs.valid.value);

    const auto& aa = p.outputs.axis_angle.value;
    // No NaN / inf may ever reach the port -- this is the whole point of cv.jit's
    // `fpclassify(theta) != FP_NORMAL` guard, and of the extra isfinite(axis) guard for
    // the case where theta stays normal but sqrt(1 - w*w) has already collapsed.
    for(int i = 0; i < 4; ++i)
      REQUIRE(std::isfinite(aa[static_cast<std::size_t>(i)]));

    // cv.jit reports theta in degrees over [0, 360).
    CHECK(aa[0] >= 0.f);
    CHECK(aa[0] <= 360.f);

    const double an = std::sqrt(
        double(aa[1]) * aa[1] + double(aa[2]) * aa[2] + double(aa[3]) * aa[3]);
    if(c.appreciable)
    {
      // For a real rotation the axis is a unit vector.
      CHECK(an == Approx(1.0).margin(1e-5));
    }
    else
    {
      // CV.JIT QUIRK, deliberately reproduced: for a near-identity rotation cv.jit still
      // divides (x,y,z) by sqrt(1 - w*w) whenever theta happens to stay FP_NORMAL. Both
      // numerator and denominator are then at the noise floor, so the emitted axis is NOT
      // unit-length (observed norms here: ~0.94 and ~1.00007). It is harmless -- theta is
      // ~1e-6 deg so the rotation round-trips regardless -- but a "fixed" implementation
      // that renormalised the axis would no longer match cv.jit. All that is guaranteed is
      // that it stays finite and bounded.
      CHECK(std::isfinite(an));
      CHECK(an < 10.0);
      CHECK(aa[0] < 1e-2f); // theta really is ~0 degrees
    }

    // Round trip: Rodrigues(axis-angle) must be the same rotation as the quaternion.
    const auto Raa = fromAxisAngle(aa);
    const auto Rq = fromQuat(p.outputs.quat.value);
    INFO("max |R_axisangle - R_quat| = " << maxAbsDiff(Raa, Rq));
    CHECK(maxAbsDiff(Raa, Rq) < 1e-4);
  }
}

TEST_CASE("N-point PnP: RANSAC rejects planted outliers", "[npoint][pnp]")
{
  Pose gt;
  gt.R = mul(mul(rotZ(0.2), rotY(-0.3)), rotX(0.12));
  gt.t = {0.2, 0.1, 4.5};

  auto clean = project(obj10, gt);
  // Two extra points so the outlier fraction stays comfortably below 50%.
  std::vector<std::array<double, 3>> obj12 = obj10;
  obj12.push_back({0.9, -0.8, 0.9});
  obj12.push_back({-0.7, -0.5, 0.2});
  clean = project(obj12, gt);
  REQUIRE(clean.size() == 12);

  auto corrupted = clean;
  const std::array<int, 3> bad{2, 5, 9};
  const std::array<std::pair<float, float>, 3> off{
      {{45.f, -38.f}, {-52.f, 41.f}, {60.f, 55.f}}};
  for(std::size_t k = 0; k < bad.size(); ++k)
  {
    auto& c = corrupted[static_cast<std::size_t>(bad[k])];
    c.ix += off[k].first;
    c.iy += off[k].second;
  }

  // Plain least squares: the outliers drag the pose away.
  cv::SolvePnP ls;
  setIntrinsics(ls);
  ls.inputs.points.value = corrupted;
  ls.inputs.method.value = cv::PnpMethod::LeastSquares;
  ls();
  REQUIRE(ls.outputs.valid.value);
  const double lsErr = std::hypot(
      std::hypot(
          ls.outputs.translation.value.x - gt.t[0],
          ls.outputs.translation.value.y - gt.t[1]),
      ls.outputs.translation.value.z - gt.t[2]);
  INFO("plain LS translation error: " << lsErr);
  CHECK(lsErr > 0.02);
  CHECK(ls.outputs.inliers.value == 12);

  // RANSAC with cv.jit.unproject's exact hard-coded parameters.
  cv::SolvePnP rs;
  setIntrinsics(rs);
  rs.inputs.points.value = corrupted;
  rs.inputs.method.value = cv::PnpMethod::RANSAC;
  CHECK(rs.inputs.iterations_count.value == 100);
  CHECK(rs.inputs.reprojection_error.value == Approx(8.0f));
  CHECK(rs.inputs.confidence.value == Approx(0.99f));
  rs();
  REQUIRE(rs.outputs.valid.value);

  CHECK(rs.outputs.inliers.value == 9);
  REQUIRE(rs.outputs.inlier_mask.value.size() == 12);
  for(int i = 0; i < 12; ++i)
  {
    const bool planted = std::find(bad.begin(), bad.end(), i) != bad.end();
    INFO("point " << i);
    CHECK(rs.outputs.inlier_mask.value[static_cast<std::size_t>(i)] == (planted ? 0 : 1));
  }

  const double rsErr = std::hypot(
      std::hypot(
          rs.outputs.translation.value.x - gt.t[0],
          rs.outputs.translation.value.y - gt.t[1]),
      rs.outputs.translation.value.z - gt.t[2]);
  INFO("RANSAC translation error: " << rsErr);
  CHECK(rsErr < 5e-3);
  CHECK(rsErr < lsErr / 4.0);

  std::array<double, 9> R{};
  for(int i = 0; i < 9; ++i)
    R[static_cast<std::size_t>(i)] = rs.outputs.matrix.value[static_cast<std::size_t>(i)];
  CHECK(maxAbsDiff(R, gt.R) < 2e-3);
}

TEST_CASE("N-point PnP: an empty list falls back to the 4 spinbox sets", "[npoint][pnp]")
{
  Pose gt;
  gt.R = mul(rotY(0.3), rotX(-0.15));
  gt.t = {0.2, -0.1, 5.0};
  const auto c = project(obj4, gt);
  REQUIRE(c.size() == 4);

  cv::SolvePnP spin;
  setIntrinsics(spin);
  spin.inputs.obj0.value = {c[0].ox, c[0].oy, c[0].oz};
  spin.inputs.obj1.value = {c[1].ox, c[1].oy, c[1].oz};
  spin.inputs.obj2.value = {c[2].ox, c[2].oy, c[2].oz};
  spin.inputs.obj3.value = {c[3].ox, c[3].oy, c[3].oz};
  spin.inputs.img0.value = {c[0].ix, c[0].iy};
  spin.inputs.img1.value = {c[1].ix, c[1].iy};
  spin.inputs.img2.value = {c[2].ix, c[2].iy};
  spin.inputs.img3.value = {c[3].ix, c[3].iy};
  REQUIRE(spin.inputs.points.value.empty());
  spin();
  REQUIRE(spin.outputs.valid.value);

  CHECK(spin.outputs.translation.value.x == Approx(gt.t[0]).margin(2e-3));
  CHECK(spin.outputs.translation.value.y == Approx(gt.t[1]).margin(2e-3));
  CHECK(spin.outputs.translation.value.z == Approx(gt.t[2]).margin(5e-3));
  CHECK(spin.outputs.inliers.value == 4);

  // Same data through the list input gives the same pose.
  cv::SolvePnP lst;
  setIntrinsics(lst);
  lst.inputs.points.value = c;
  lst();
  REQUIRE(lst.outputs.valid.value);
  for(int i = 0; i < 9; ++i)
    CHECK(
        lst.outputs.matrix.value[static_cast<std::size_t>(i)]
        == Approx(spin.outputs.matrix.value[static_cast<std::size_t>(i)]).margin(1e-5));
  CHECK(lst.outputs.translation.value.x == Approx(spin.outputs.translation.value.x).margin(1e-5));
  CHECK(lst.outputs.translation.value.y == Approx(spin.outputs.translation.value.y).margin(1e-5));
  CHECK(lst.outputs.translation.value.z == Approx(spin.outputs.translation.value.z).margin(1e-5));

  // A short list (< 4) is rejected rather than silently padded from the spinboxes.
  cv::SolvePnP shortList;
  setIntrinsics(shortList);
  shortList.inputs.points.value = {c[0], c[1], c[2]};
  shortList();
  CHECK_FALSE(shortList.outputs.valid.value);
}
