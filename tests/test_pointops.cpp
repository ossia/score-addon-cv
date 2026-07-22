// Tests for the pure-index / pure-arithmetic point objects:
//   PerspectivePoints  (cv.jit.perspective)
//   FaceParts          (cv.jit.face.parts)
//   FaceRigidPoints    (cv.jit.face.rigidpoints)
//
// Test-case names are prefixed "Point-ops" because tests/run_standalone.sh links every
// test TU into one Catch2 binary, where test-case names must be globally unique.
//
// ---------------------------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------------------------
// PerspectivePoints applies the FORWARD projective map
//     [x' y' w']^T = H . [x y 1]^T,     out = (x'/w', y'/w')
// with H row-major, H[r*3+c] = H(r, c).
//
// (A) Pure scale + translation.  H_aff = [ 2 0 10 ; 0 3 -5 ; 0 0 1 ], so w' == 1 for
//     every point and the result is exactly (2x + 10, 3y - 5) with no division at all.
//     Every value below is exactly representable in binary32, so these are asserted with
//     == rather than Approx; an implementation that divided by a w' it had rounded would
//     fail them.
//         (1, 2)  -> (2*1+10, 3*2-5)   = (12,  1)
//         (0, 0)  -> (10, -5)
//         (-3, 4) -> (2*-3+10, 3*4-5)  = ( 4,  7)
//         (0.5, -1.5) -> (11, -9.5)
//
// (B) Genuinely projective.  H_proj = [ 2 0 10 ; 0 3 -5 ; 0.1 0.2 1 ]: same first two
//     rows as (A), but now w' = 0.1x + 0.2y + 1 varies, so the two cases differ exactly
//     by the perspective division. Hand-computed:
//         (1, 2):    x' = 2 + 0 + 10 = 12,  y' = 0 + 6 - 5 = 1,
//                    w' = 0.1 + 0.4 + 1 = 1.5      -> (12/1.5, 1/1.5) = (8, 0.6666667)
//         (0, 0):    x' = 10, y' = -5, w' = 1      -> (10, -5)
//         (-3, 4):   x' = -6 + 10 = 4,  y' = 12 - 5 = 7,
//                    w' = -0.3 + 0.8 + 1 = 1.5     -> (4/1.5, 7/1.5)
//                                                  = (2.6666667, 4.6666667)
//         (5, -1):   x' = 10 + 10 = 20, y' = -3 - 5 = -8,
//                    w' = 0.5 - 0.2 + 1 = 1.3      -> (20/1.3, -8/1.3)
//                                                  = (15.3846154, -6.1538462)
//     Note (0, 0) lands on the same value in (A) and (B) because w' = 1 there; that is
//     the only point of the set where the two agree, which is itself a check that the
//     division is really happening.
//
// (C) The horizon line.  H_hor = [ 1 0 0 ; 0 1 0 ; 1 0 0 ] gives w' = x, so the whole
//     line x = 0 is the horizon: those points have no image in the affine plane. OpenCV's
//     perspectiveTransform emits (0, 0) for them and the object does the same. Off the
//     horizon it is the ordinary map (x, y) -> (x/x, y/x) = (1, y/x):
//         (2, 3)   -> (1, 1.5)
//         (0, 5)   -> guarded, (0, 0)
//         (1e-12, 7) -> |w'| = 1e-12 < 1e-8, guarded, (0, 0)   [1e-12/1e-12 would be 1,
//                       but 7/1e-12 = 7e12 -- and with x = 0 exactly it would be inf]
//         (-4, 8)  -> (1, -2)
//
// ---------------------------------------------------------------------------------------
// FaceParts / FaceRigidPoints use synthetic landmarks  L(i) = (i, 100 + i)  so that every
// point carries its own index in both coordinates and a mis-slice by even one position is
// visible. The group boundaries come from cv.jit/misc/lbfmodel.yaml.map:
//     jaw 0 17 | brow-l 17 5 | brow-r 22 5 | nose-ridge 27 4 | nose-base 31 5
//     eye-l 36 6 | eye-r 42 6 | lips 48 12 | mouth 60 8            (sum = 68)
// so the expected first/last indices per group are
//     jaw 0..16, brow-l 17..21, brow-r 22..26, nose-ridge 27..30, nose-base 31..35,
//     eye-l 36..41, eye-r 42..47, lips 48..59, mouth 60..67.
//
// FaceRigidPoints emits (27, 33, 36, 45, 2, 14) in that order; see the derivation of each
// index from the map in CV/Cpu/FaceParts.hpp.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <CV/Cpu/FaceParts.hpp>
#include <CV/Cpu/Homography.hpp>
#include <CV/Cpu/PerspectivePoints.hpp>

#include <array>
#include <cmath>
#include <vector>

using Catch::Approx;

namespace
{
// Row-major 3x3 as the object takes it.
constexpr std::array<float, 9> H_aff{2.f, 0.f, 10.f, 0.f, 3.f, -5.f, 0.f, 0.f, 1.f};
constexpr std::array<float, 9> H_proj{2.f, 0.f, 10.f, 0.f, 3.f, -5.f, 0.1f, 0.2f, 1.f};
constexpr std::array<float, 9> H_hor{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f};

std::vector<cv::point2> pts(std::initializer_list<std::pair<float, float>> l)
{
  std::vector<cv::point2> v;
  v.reserve(l.size());
  for(auto& p : l)
    v.push_back({p.first, p.second});
  return v;
}

// L(i) = (i, 100 + i)
std::vector<cv::point2> landmarks(int n)
{
  std::vector<cv::point2> v;
  v.reserve(static_cast<std::size_t>(n));
  for(int i = 0; i < n; ++i)
    v.push_back({static_cast<float>(i), static_cast<float>(100 + i)});
  return v;
}

bool isLandmark(const cv::point2& p, int i)
{
  return p.x == static_cast<float>(i) && p.y == static_cast<float>(100 + i);
}

bool allFinite(const std::vector<cv::point2>& v)
{
  for(auto& p : v)
    if(!std::isfinite(p.x) || !std::isfinite(p.y))
      return false;
  return true;
}
}

// =======================================================================================
// PerspectivePoints
// =======================================================================================

TEST_CASE("Point-ops: perspective identity is a pass-through", "[perspective_points]")
{
  cv::PerspectivePoints obj;

  // A freshly constructed object must already be the identity: max.cv.jit.perspective.cpp
  // pre-fills its matrix inlet with the identity so the object does nothing until fed.
  REQUIRE(obj.inputs.matrix.value == cv::identity3);

  const auto in = pts({{0.f, 0.f}, {1.f, 2.f}, {-3.5f, 4.25f}, {1000.f, -1000.f}});
  obj.inputs.points.value = in;
  obj();

  const auto& out = obj.outputs.points.value;
  REQUIRE(out.size() == in.size());
  for(std::size_t i = 0; i < in.size(); ++i)
  {
    // Exact: the identity performs no arithmetic that can round.
    CHECK(out[i].x == in[i].x);
    CHECK(out[i].y == in[i].y);
  }

  // Explicitly setting the identity must behave the same.
  obj.inputs.matrix.value = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
  obj();
  REQUIRE(obj.outputs.points.value.size() == in.size());
  for(std::size_t i = 0; i < in.size(); ++i)
  {
    CHECK(obj.outputs.points.value[i].x == in[i].x);
    CHECK(obj.outputs.points.value[i].y == in[i].y);
  }
}

TEST_CASE("Point-ops: perspective translation/scale is exact", "[perspective_points]")
{
  // Derivation (A): out = (2x + 10, 3y - 5), w' == 1 everywhere.
  cv::PerspectivePoints obj;
  obj.inputs.matrix.value = H_aff;
  obj.inputs.points.value = pts({{1.f, 2.f}, {0.f, 0.f}, {-3.f, 4.f}, {0.5f, -1.5f}});
  obj();

  const auto& out = obj.outputs.points.value;
  REQUIRE(out.size() == 4);
  CHECK(out[0].x == 12.f);
  CHECK(out[0].y == 1.f);
  CHECK(out[1].x == 10.f);
  CHECK(out[1].y == -5.f);
  CHECK(out[2].x == 4.f);
  CHECK(out[2].y == 7.f);
  CHECK(out[3].x == 11.f);
  CHECK(out[3].y == -9.5f);
}

TEST_CASE("Point-ops: perspective projective mapping", "[perspective_points]")
{
  // Derivation (B).
  cv::PerspectivePoints obj;
  obj.inputs.matrix.value = H_proj;
  obj.inputs.points.value = pts({{1.f, 2.f}, {0.f, 0.f}, {-3.f, 4.f}, {5.f, -1.f}});
  obj();

  const auto& out = obj.outputs.points.value;
  REQUIRE(out.size() == 4);

  CHECK(out[0].x == Approx(8.0).epsilon(1e-6));
  CHECK(out[0].y == Approx(1.0 / 1.5).epsilon(1e-6));

  CHECK(out[1].x == Approx(10.0).epsilon(1e-6));
  CHECK(out[1].y == Approx(-5.0).epsilon(1e-6));

  CHECK(out[2].x == Approx(4.0 / 1.5).epsilon(1e-6));
  CHECK(out[2].y == Approx(7.0 / 1.5).epsilon(1e-6));

  CHECK(out[3].x == Approx(20.0 / 1.3).epsilon(1e-6));
  CHECK(out[3].y == Approx(-8.0 / 1.3).epsilon(1e-6));

  // The perspective division is really applied: only the point with w' = 1 agrees with
  // the affine matrix that shares H_proj's first two rows.
  cv::PerspectivePoints aff;
  aff.inputs.matrix.value = H_aff;
  aff.inputs.points.value = obj.inputs.points.value;
  aff();
  CHECK(aff.outputs.points.value[1].x == Approx(out[1].x));
  CHECK(aff.outputs.points.value[1].y == Approx(out[1].y));
  CHECK(aff.outputs.points.value[0].x != Approx(out[0].x));
  CHECK(aff.outputs.points.value[2].y != Approx(out[2].y));
}

TEST_CASE("Point-ops: perspective guards the horizon line", "[perspective_points]")
{
  // Derivation (C): w' = x, so x = 0 (and anything with |x| < 1e-8) is guarded to (0, 0).
  cv::PerspectivePoints obj;
  obj.inputs.matrix.value = H_hor;
  obj.inputs.points.value
      = pts({{2.f, 3.f}, {0.f, 5.f}, {1e-12f, 7.f}, {-4.f, 8.f}, {-1e-9f, -2.f}});
  obj();

  const auto& out = obj.outputs.points.value;
  REQUIRE(out.size() == 5);
  REQUIRE(allFinite(out));

  CHECK(out[0].x == Approx(1.0));
  CHECK(out[0].y == Approx(1.5));

  CHECK(out[1].x == 0.f); // exactly on the horizon
  CHECK(out[1].y == 0.f);

  CHECK(out[2].x == 0.f); // |w'| = 1e-12 < 1e-8
  CHECK(out[2].y == 0.f);

  CHECK(out[3].x == Approx(1.0));
  CHECK(out[3].y == Approx(-2.0));

  CHECK(out[4].x == 0.f); // negative but tiny w'
  CHECK(out[4].y == 0.f);

  // A matrix whose last row is entirely zero puts *every* point on the horizon.
  obj.inputs.matrix.value = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f};
  obj();
  REQUIRE(obj.outputs.points.value.size() == 5);
  REQUIRE(allFinite(obj.outputs.points.value));
  for(auto& p : obj.outputs.points.value)
  {
    CHECK(p.x == 0.f);
    CHECK(p.y == 0.f);
  }
}

TEST_CASE("Point-ops: perspective point count follows the input", "[perspective_points]")
{
  cv::PerspectivePoints obj;
  obj.inputs.matrix.value = H_proj;

  // Empty in, empty out.
  obj.inputs.points.value.clear();
  obj();
  CHECK(obj.outputs.points.value.empty());

  // Grow.
  obj.inputs.points.value = pts({{1.f, 2.f}, {0.f, 0.f}, {-3.f, 4.f}});
  obj();
  CHECK(obj.outputs.points.value.size() == 3);

  // Shrink: the object must not keep stale points from the previous tick.
  obj.inputs.points.value = pts({{0.f, 0.f}});
  obj();
  REQUIRE(obj.outputs.points.value.size() == 1);
  CHECK(obj.outputs.points.value[0].x == Approx(10.0));
  CHECK(obj.outputs.points.value[0].y == Approx(-5.0));

  // And back to empty.
  obj.inputs.points.value.clear();
  obj();
  CHECK(obj.outputs.points.value.empty());
}

TEST_CASE("Point-ops: Homography chains into PerspectivePoints", "[perspective_points]")
{
  // Build H from 4 correspondences with the addon's own estimator, feed the resulting
  // Matrix port straight into PerspectivePoints (same std::array<float, 9> row-major
  // type, no adapter), and check that the source points land on the destination points.
  //
  // The correspondence set is a unit square mapped to a trapezoid, i.e. a genuinely
  // projective transform (a parallelogram would only need an affine one).
  const std::vector<std::pair<float, float>> src{
      {0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
  const std::vector<std::pair<float, float>> dst{
      {10.f, 20.f}, {110.f, 30.f}, {90.f, 130.f}, {20.f, 110.f}};

  cv::Homography h;
  for(std::size_t i = 0; i < 4; ++i)
    h.inputs.points.value.push_back(
        {src[i].first, src[i].second, dst[i].first, dst[i].second});
  h.inputs.method.value = cv::HomographyMethod::LeastSquares;
  h();
  REQUIRE(h.outputs.valid.value);

  cv::PerspectivePoints p;
  p.inputs.matrix.value = h.outputs.matrix.value; // the chain
  for(auto& s : src)
    p.inputs.points.value.push_back({s.first, s.second});
  p();

  REQUIRE(p.outputs.points.value.size() == 4);
  for(std::size_t i = 0; i < 4; ++i)
  {
    CHECK(p.outputs.points.value[i].x == Approx(dst[i].first).margin(1e-3));
    CHECK(p.outputs.points.value[i].y == Approx(dst[i].second).margin(1e-3));
  }

  // The same H must also map the square's centre to the trapezoid's projective centre,
  // i.e. to the intersection of its diagonals -- which is NOT the centroid of the four
  // destination corners. Diagonals of dst: (10,20)-(90,130) and (110,30)-(20,110).
  //   line 1: (10,20) + t*(80,110)
  //   line 2: (110,30) + u*(-90,80)
  //   10 + 80t = 110 - 90u  and  20 + 110t = 30 + 80u
  //   => 80t + 90u = 100 ; 110t - 80u = 10
  //   From the second: u = (110t - 10)/80. Substituting:
  //   80t + 90*(110t - 10)/80 = 100  ->  6400t + 9900t - 900 = 8000
  //   ->  16300t = 8900  ->  t = 89/163
  //   => x = 10 + 80*89/163 = (1630 + 7120)/163 = 8750/163 = 53.680982...
  //      y = 20 + 110*89/163 = (3260 + 9790)/163 = 13050/163 = 80.061350...
  // The centroid of dst is ((10+110+90+20)/4, (20+30+130+110)/4) = (57.5, 72.5), well
  // away from that, so an affine-only implementation would fail this.
  p.inputs.points.value = pts({{0.5f, 0.5f}});
  p();
  REQUIRE(p.outputs.points.value.size() == 1);
  CHECK(p.outputs.points.value[0].x == Approx(8750.0 / 163.0).margin(1e-3));
  CHECK(p.outputs.points.value[0].y == Approx(13050.0 / 163.0).margin(1e-3));
}

// =======================================================================================
// FaceParts
// =======================================================================================

TEST_CASE("Point-ops: face landmark map matches lbfmodel.yaml.map", "[face_parts]")
{
  // The shipped constexpr table must stay byte-identical to cv.jit/misc/lbfmodel.yaml.map:
  //   jaw, 0 17;  brow-l, 17 5;  brow-r, 22 5;  nose-ridge, 27 4;  nose-base, 31 5;
  //   eye-l, 36 6;  eye-r, 42 6;  lips, 48 12;  mouth, 60 8;
  const auto& m = cv::face_landmark_map;
  REQUIRE(m.size() == 9);

  CHECK(m[0].name == "jaw");
  CHECK(m[0].start == 0);
  CHECK(m[0].count == 17);
  CHECK(m[1].name == "brow-l");
  CHECK(m[1].start == 17);
  CHECK(m[1].count == 5);
  CHECK(m[2].name == "brow-r");
  CHECK(m[2].start == 22);
  CHECK(m[2].count == 5);
  CHECK(m[3].name == "nose-ridge");
  CHECK(m[3].start == 27);
  CHECK(m[3].count == 4);
  CHECK(m[4].name == "nose-base");
  CHECK(m[4].start == 31);
  CHECK(m[4].count == 5);
  CHECK(m[5].name == "eye-l");
  CHECK(m[5].start == 36);
  CHECK(m[5].count == 6);
  CHECK(m[6].name == "eye-r");
  CHECK(m[6].start == 42);
  CHECK(m[6].count == 6);
  CHECK(m[7].name == "lips");
  CHECK(m[7].start == 48);
  CHECK(m[7].count == 12);
  CHECK(m[8].name == "mouth");
  CHECK(m[8].start == 60);
  CHECK(m[8].count == 8);

  // The groups tile 0..67 with no gap and no overlap, and cover exactly 68 landmarks.
  int total = 0;
  int next = 0;
  for(auto& g : m)
  {
    CHECK(g.start == next);
    next = g.start + g.count;
    total += g.count;
  }
  CHECK(total == cv::face_landmark_count);
  CHECK(next == 68);
}

TEST_CASE("Point-ops: face parts slices a 68-point list", "[face_parts]")
{
  cv::FaceParts obj;
  obj.inputs.landmarks.value = landmarks(68);
  obj();

  const auto& o = obj.outputs;

  // All 9 group sizes.
  REQUIRE(o.jaw.value.size() == 17);
  REQUIRE(o.brow_l.value.size() == 5);
  REQUIRE(o.brow_r.value.size() == 5);
  REQUIRE(o.nose_ridge.value.size() == 4);
  REQUIRE(o.nose_base.value.size() == 5);
  REQUIRE(o.eye_l.value.size() == 6);
  REQUIRE(o.eye_r.value.size() == 6);
  REQUIRE(o.lips.value.size() == 12);
  REQUIRE(o.mouth.value.size() == 8);

  // First and last element of each group, by index.
  CHECK(isLandmark(o.jaw.value.front(), 0));
  CHECK(isLandmark(o.jaw.value.back(), 16));
  CHECK(isLandmark(o.brow_l.value.front(), 17));
  CHECK(isLandmark(o.brow_l.value.back(), 21));
  CHECK(isLandmark(o.brow_r.value.front(), 22));
  CHECK(isLandmark(o.brow_r.value.back(), 26));
  CHECK(isLandmark(o.nose_ridge.value.front(), 27));
  CHECK(isLandmark(o.nose_ridge.value.back(), 30));
  CHECK(isLandmark(o.nose_base.value.front(), 31));
  CHECK(isLandmark(o.nose_base.value.back(), 35));
  CHECK(isLandmark(o.eye_l.value.front(), 36));
  CHECK(isLandmark(o.eye_l.value.back(), 41));
  CHECK(isLandmark(o.eye_r.value.front(), 42));
  CHECK(isLandmark(o.eye_r.value.back(), 47));
  CHECK(isLandmark(o.lips.value.front(), 48));
  CHECK(isLandmark(o.lips.value.back(), 59));
  CHECK(isLandmark(o.mouth.value.front(), 60));
  CHECK(isLandmark(o.mouth.value.back(), 67));

  // Every element, not just the endpoints: the groups concatenated in outlet order must
  // reproduce 0..67 exactly (which also proves there is no gap or duplicate).
  std::vector<cv::point2> all;
  for(const std::vector<cv::point2>* g :
      {&o.jaw.value, &o.brow_l.value, &o.brow_r.value, &o.nose_ridge.value,
       &o.nose_base.value, &o.eye_l.value, &o.eye_r.value, &o.lips.value, &o.mouth.value})
    all.insert(all.end(), g->begin(), g->end());
  REQUIRE(all.size() == 68);
  for(int i = 0; i < 68; ++i)
    CHECK(isLandmark(all[static_cast<std::size_t>(i)], i));
}

TEST_CASE("Point-ops: face parts handles short and empty input", "[face_parts]")
{
  cv::FaceParts obj;

  // Empty: every group empty, nothing dereferenced.
  obj.inputs.landmarks.value.clear();
  obj();
  CHECK(obj.outputs.jaw.value.empty());
  CHECK(obj.outputs.brow_l.value.empty());
  CHECK(obj.outputs.brow_r.value.empty());
  CHECK(obj.outputs.nose_ridge.value.empty());
  CHECK(obj.outputs.nose_base.value.empty());
  CHECK(obj.outputs.eye_l.value.empty());
  CHECK(obj.outputs.eye_r.value.empty());
  CHECK(obj.outputs.lips.value.empty());
  CHECK(obj.outputs.mouth.value.empty());

  // 30 points: jaw (0..16), brow-l (17..21) and brow-r (22..26) fit; nose-ridge needs
  // index 30 inclusive, i.e. 31 landmarks, so it and everything after it come out empty.
  // Under ASan a clamped-instead-of-emptied implementation would also be caught here.
  obj.inputs.landmarks.value = landmarks(30);
  obj();
  REQUIRE(obj.outputs.jaw.value.size() == 17);
  REQUIRE(obj.outputs.brow_l.value.size() == 5);
  REQUIRE(obj.outputs.brow_r.value.size() == 5);
  CHECK(isLandmark(obj.outputs.brow_r.value.back(), 26));
  CHECK(obj.outputs.nose_ridge.value.empty());
  CHECK(obj.outputs.nose_base.value.empty());
  CHECK(obj.outputs.eye_l.value.empty());
  CHECK(obj.outputs.eye_r.value.empty());
  CHECK(obj.outputs.lips.value.empty());
  CHECK(obj.outputs.mouth.value.empty());

  // Exactly one landmark short of a full nose-ridge (30 needed for 27..30 -> 31 points).
  obj.inputs.landmarks.value = landmarks(31);
  obj();
  REQUIRE(obj.outputs.nose_ridge.value.size() == 4);
  CHECK(isLandmark(obj.outputs.nose_ridge.value.back(), 30));
  CHECK(obj.outputs.nose_base.value.empty());

  // Longer than 68: the extra landmarks are simply ignored.
  obj.inputs.landmarks.value = landmarks(100);
  obj();
  REQUIRE(obj.outputs.mouth.value.size() == 8);
  CHECK(isLandmark(obj.outputs.mouth.value.back(), 67));

  // Back to empty: no stale groups left over from the previous tick.
  obj.inputs.landmarks.value.clear();
  obj();
  CHECK(obj.outputs.jaw.value.empty());
  CHECK(obj.outputs.mouth.value.empty());
}

TEST_CASE("Point-ops: face parts groups chain into PerspectivePoints", "[face_parts]")
{
  // Shared element type: a FaceParts outlet plugs into PerspectivePoints with no adapter.
  cv::FaceParts fp;
  fp.inputs.landmarks.value = landmarks(68);
  fp();

  cv::PerspectivePoints pp;
  pp.inputs.matrix.value = H_aff; // (x, y) -> (2x + 10, 3y - 5)
  pp.inputs.points.value = fp.outputs.eye_l.value;
  pp();

  REQUIRE(pp.outputs.points.value.size() == 6);
  // eye-l is landmarks 36..41, i.e. (36, 136) .. (41, 141).
  CHECK(pp.outputs.points.value.front().x == 2.f * 36.f + 10.f);
  CHECK(pp.outputs.points.value.front().y == 3.f * 136.f - 5.f);
  CHECK(pp.outputs.points.value.back().x == 2.f * 41.f + 10.f);
  CHECK(pp.outputs.points.value.back().y == 3.f * 141.f - 5.f);
}

// =======================================================================================
// FaceRigidPoints
// =======================================================================================

TEST_CASE("Point-ops: rigid points are 27 33 36 45 2 14 in order", "[face_rigid]")
{
  cv::FaceRigidPoints obj;
  obj.inputs.landmarks.value = landmarks(68);
  obj();

  const auto& p = obj.outputs.points.value;
  REQUIRE(p.size() == 6);

  // Order matters: jit.glue concatenates gen_offsets' outlets 1..6 in this order.
  CHECK(isLandmark(p[0], 27)); // nose top / bridge          = nose-ridge.start
  CHECK(isLandmark(p[1], 33)); // nose base centre           = 31 + 5/2
  CHECK(isLandmark(p[2], 36)); // left-eye outer corner      = eye-l.start
  CHECK(isLandmark(p[3], 45)); // right-eye outer corner     = 42 + 6/2
  CHECK(isLandmark(p[4], 2));  // left jaw                   = jaw.start + 2
  CHECK(isLandmark(p[5], 14)); // right jaw                  = jaw.start + 17 - 3

  // The table itself, so a reordering is caught even if the object stopped using it.
  REQUIRE(cv::face_rigid_indices.size() == 6);
  CHECK(cv::face_rigid_indices[0] == 27);
  CHECK(cv::face_rigid_indices[1] == 33);
  CHECK(cv::face_rigid_indices[2] == 36);
  CHECK(cv::face_rigid_indices[3] == 45);
  CHECK(cv::face_rigid_indices[4] == 2);
  CHECK(cv::face_rigid_indices[5] == 14);

  // The indices really are derived from the map (integer division included).
  CHECK(cv::face_rigid_indices[0] == cv::face_landmark_map[3].start);
  CHECK(
      cv::face_rigid_indices[1]
      == cv::face_landmark_map[4].start + cv::face_landmark_map[4].count / 2);
  CHECK(cv::face_rigid_indices[2] == cv::face_landmark_map[5].start);
  CHECK(
      cv::face_rigid_indices[3]
      == cv::face_landmark_map[6].start + cv::face_landmark_map[6].count / 2);
  CHECK(cv::face_rigid_indices[4] == cv::face_landmark_map[0].start + 2);
  CHECK(
      cv::face_rigid_indices[5]
      == cv::face_landmark_map[0].start + cv::face_landmark_map[0].count - 3);
}

TEST_CASE("Point-ops: rigid points handle short input", "[face_rigid]")
{
  cv::FaceRigidPoints obj;

  // Empty.
  obj.inputs.landmarks.value.clear();
  obj();
  CHECK(obj.outputs.points.value.empty());
  CHECK(obj.outputs.correspondences.value.empty());

  // Too short: index 45 is needed, so 45 landmarks (0..44) is one short.
  obj.inputs.landmarks.value = landmarks(45);
  obj();
  CHECK(obj.outputs.points.value.empty());
  CHECK(obj.outputs.correspondences.value.empty());

  // Exactly enough: 46 landmarks (0..45).
  REQUIRE(cv::face_rigid_min_landmarks == 46);
  obj.inputs.landmarks.value = landmarks(46);
  obj();
  REQUIRE(obj.outputs.points.value.size() == 6);
  CHECK(isLandmark(obj.outputs.points.value[3], 45));
  CHECK(isLandmark(obj.outputs.points.value[5], 14));

  // ...and back to short: nothing stale is left behind.
  obj.inputs.landmarks.value = landmarks(10);
  obj();
  CHECK(obj.outputs.points.value.empty());
  CHECK(obj.outputs.correspondences.value.empty());
}

TEST_CASE("Point-ops: rigid points emit PnP correspondences", "[face_rigid]")
{
  cv::FaceRigidPoints obj;
  obj.inputs.landmarks.value = landmarks(68);
  obj();

  const auto& c = obj.outputs.correspondences.value;
  REQUIRE(c.size() == 6);

  // The 2D side is the extracted landmark, in the same order as the Points outlet.
  for(std::size_t k = 0; k < 6; ++k)
  {
    const int idx = cv::face_rigid_indices[k];
    CHECK(c[k].ix == static_cast<float>(idx));
    CHECK(c[k].iy == static_cast<float>(100 + idx));
    CHECK(c[k].ix == obj.outputs.points.value[k].x);
    CHECK(c[k].iy == obj.outputs.points.value[k].y);
  }

  // The 3D side is the built-in model, unscaled (default Model scale is 1).
  REQUIRE(obj.inputs.model_scale.value == 1.f);
  for(std::size_t k = 0; k < 6; ++k)
  {
    CHECK(c[k].ox == cv::face_rigid_model[k].x);
    CHECK(c[k].oy == cv::face_rigid_model[k].y);
    CHECK(c[k].oz == cv::face_rigid_model[k].z);
  }

  // The model must be usable by SolvePnP, i.e. not coplanar-degenerate and left/right
  // symmetric: eye-l and eye-r mirror each other, likewise the two jaw points, and the
  // two nose points sit on the symmetry plane x = 0.
  CHECK(cv::face_rigid_model[0].x == 0.f);
  CHECK(cv::face_rigid_model[1].x == 0.f);
  CHECK(cv::face_rigid_model[2].x == -cv::face_rigid_model[3].x);
  CHECK(cv::face_rigid_model[2].y == cv::face_rigid_model[3].y);
  CHECK(cv::face_rigid_model[2].z == cv::face_rigid_model[3].z);
  CHECK(cv::face_rigid_model[4].x == -cv::face_rigid_model[5].x);
  CHECK(cv::face_rigid_model[4].y == cv::face_rigid_model[5].y);
  CHECK(cv::face_rigid_model[4].z == cv::face_rigid_model[5].z);
  // Not all in one plane: the six z values take at least three distinct values.
  CHECK(cv::face_rigid_model[0].z != cv::face_rigid_model[1].z);
  CHECK(cv::face_rigid_model[1].z != cv::face_rigid_model[2].z);
  CHECK(cv::face_rigid_model[2].z != cv::face_rigid_model[4].z);
}

TEST_CASE("Point-ops: rigid points model scale and override", "[face_rigid]")
{
  cv::FaceRigidPoints obj;
  obj.inputs.landmarks.value = landmarks(68);

  // Scale multiplies only the 3D side.
  obj.inputs.model_scale.value = 2.f;
  obj();
  {
    const auto& c = obj.outputs.correspondences.value;
    REQUIRE(c.size() == 6);
    for(std::size_t k = 0; k < 6; ++k)
    {
      CHECK(c[k].ox == 2.f * cv::face_rigid_model[k].x);
      CHECK(c[k].oy == 2.f * cv::face_rigid_model[k].y);
      CHECK(c[k].oz == 2.f * cv::face_rigid_model[k].z);
      CHECK(c[k].ix == static_cast<float>(cv::face_rigid_indices[k]));
    }
  }

  // A 6-point Model list replaces the built-in model (and is scaled too).
  obj.inputs.model_scale.value = 1.f;
  obj.inputs.model.value
      = {{1.f, 2.f, 3.f},    {4.f, 5.f, 6.f},    {7.f, 8.f, 9.f},
         {10.f, 11.f, 12.f}, {13.f, 14.f, 15.f}, {16.f, 17.f, 18.f}};
  obj();
  {
    const auto& c = obj.outputs.correspondences.value;
    REQUIRE(c.size() == 6);
    for(std::size_t k = 0; k < 6; ++k)
    {
      CHECK(c[k].ox == obj.inputs.model.value[k].x);
      CHECK(c[k].oy == obj.inputs.model.value[k].y);
      CHECK(c[k].oz == obj.inputs.model.value[k].z);
    }
  }

  // A Model list of the wrong length is ignored, built-in model restored -- and nothing
  // is read past its end.
  obj.inputs.model.value.resize(3);
  obj();
  {
    const auto& c = obj.outputs.correspondences.value;
    REQUIRE(c.size() == 6);
    for(std::size_t k = 0; k < 6; ++k)
      CHECK(c[k].ox == cv::face_rigid_model[k].x);
  }
}

TEST_CASE("Point-ops: FaceRigidPoints chains into SolvePnP", "[face_rigid]")
{
  // End-to-end head pose: synthesise the image projections of the built-in rigid model
  // under a known pose, feed them in as landmarks 27/33/36/45/2/14, and check that
  // SolvePnP -- fed straight from the Correspondences outlet, no adapter -- recovers it.
  //
  // Ground truth: R = identity (frontal head), t = (0, 0, 700) mm, pinhole with
  // fx = fy = 800, cx = 320, cy = 240. Then for a model point (X, Y, Z),
  //     u = 800 * X / (Z + 700) + 320,   v = 800 * Y / (Z + 700) + 240.
  // Worked out for the six model points (mm):
  //     27 ( 0, -55,  25): u = 320,                v = 240 + 800*(-55)/725
  //     33 ( 0,   0,   0): u = 320,                v = 240
  //     36 (-65,-60,  50): u = 320 + 800*(-65)/750, v = 240 + 800*(-60)/750
  //     45 ( 65,-60,  50): u = 320 + 800*( 65)/750, v = same as 36
  //     2  (-78, 30,  95): u = 320 + 800*(-78)/795, v = 240 + 800*( 30)/795
  //     14 ( 78, 30,  95): u = 320 + 800*( 78)/795, v = same as 2
  constexpr double fx = 800.0, fy = 800.0, cx = 320.0, cy = 240.0, tz = 700.0;

  std::vector<cv::point2> lm(68, cv::point2{0.f, 0.f});
  for(std::size_t k = 0; k < 6; ++k)
  {
    const auto& m = cv::face_rigid_model[k];
    const double Z = m.z + tz;
    lm[static_cast<std::size_t>(cv::face_rigid_indices[k])]
        = {static_cast<float>(fx * m.x / Z + cx), static_cast<float>(fy * m.y / Z + cy)};
  }

  cv::FaceRigidPoints frp;
  frp.inputs.landmarks.value = lm;
  frp();
  REQUIRE(frp.outputs.correspondences.value.size() == 6);

  cv::SolvePnP pnp;
  pnp.inputs.points.value = frp.outputs.correspondences.value; // the chain
  pnp.inputs.fx.value = static_cast<float>(fx);
  pnp.inputs.fy.value = static_cast<float>(fy);
  pnp.inputs.cx.value = static_cast<float>(cx);
  pnp.inputs.cy.value = static_cast<float>(cy);
  pnp.inputs.method.value = cv::PnpMethod::LeastSquares;
  pnp();

  REQUIRE(pnp.outputs.valid.value);
  CHECK(pnp.outputs.translation.value.x == Approx(0.0).margin(1.0));
  CHECK(pnp.outputs.translation.value.y == Approx(0.0).margin(1.0));
  CHECK(pnp.outputs.translation.value.z == Approx(tz).margin(5.0));
  // Identity rotation.
  const auto& R = pnp.outputs.matrix.value;
  CHECK(R[0] == Approx(1.0).margin(1e-2));
  CHECK(R[4] == Approx(1.0).margin(1e-2));
  CHECK(R[8] == Approx(1.0).margin(1e-2));
  CHECK(R[1] == Approx(0.0).margin(1e-2));
  CHECK(R[2] == Approx(0.0).margin(1e-2));
  CHECK(R[3] == Approx(0.0).margin(1e-2));
  CHECK(R[5] == Approx(0.0).margin(1e-2));
  CHECK(R[6] == Approx(0.0).margin(1e-2));
  CHECK(R[7] == Approx(0.0).margin(1e-2));
}
