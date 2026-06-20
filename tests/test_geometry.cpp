// Tests for the control-in / value-out geometry & math objects: Kalman, Homography, SolvePnP.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <CV/Cpu/Kalman.hpp>
#include <CV/Cpu/Homography.hpp>
#include <CV/Cpu/SolvePnP.hpp>

#include <cmath>

using Catch::Approx;

// ---------------------------------------------------------------------------- Kalman
TEST_CASE("Kalman converges to a static measurement", "[kalman]")
{
  cv::Kalman k;
  k.inputs.q.value = 0.01f;
  k.inputs.r.value = 0.05f;
  k.inputs.measurement.value = {0.3f, 0.7f};

  for(int i = 0; i < 60; ++i)
    k();

  CHECK(k.outputs.estimate.value.x == Approx(0.3f).margin(0.02));
  CHECK(k.outputs.estimate.value.y == Approx(0.7f).margin(0.02));
  // A static target -> velocity should settle near zero.
  CHECK(std::abs(k.outputs.velocity.value.x) < 0.05f);
  CHECK(std::abs(k.outputs.velocity.value.y) < 0.05f);
}

TEST_CASE("Kalman tracks constant velocity and estimates it", "[kalman]")
{
  cv::Kalman k;
  k.inputs.q.value = 0.01f;
  k.inputs.r.value = 0.1f;

  float x = 0.f, y = 0.f;
  const float vx = 0.01f, vy = 0.005f;
  for(int t = 0; t < 50; ++t)
  {
    // noisy measurement of a moving point
    float nx = x + ((t % 3) - 1) * 0.002f;
    float ny = y + ((t % 2) ? 0.001f : -0.001f);
    k.inputs.measurement.value = {nx, ny};
    k();
    x += vx;
    y += vy;
  }

  CHECK(k.outputs.velocity.value.x == Approx(vx).margin(0.004));
  CHECK(k.outputs.velocity.value.y == Approx(vy).margin(0.004));
}

TEST_CASE("Kalman reset reinitialises state", "[kalman]")
{
  cv::Kalman k;
  k.inputs.measurement.value = {0.9f, 0.9f};
  for(int i = 0; i < 30; ++i)
    k();
  REQUIRE(k.outputs.estimate.value.x > 0.5f);

  k.inputs.reset.value = true;
  k.inputs.measurement.value = {0.1f, 0.1f};
  k();
  // After reset the estimate jumps toward the new (low) measurement.
  CHECK(k.outputs.estimate.value.x < 0.5f);
}

TEST_CASE("Kalman seeds from the first measurement (no origin transient)", "[kalman]")
{
  cv::Kalman k;
  k.inputs.q.value = 0.01f;
  k.inputs.r.value = 0.1f;
  // Far-from-origin measurement: a slow pull from (0,0) would be obvious.
  k.inputs.measurement.value = {5.f, -3.f};

  // The FIRST update must already sit essentially on the measurement.
  k();
  CHECK(k.outputs.estimate.value.x == Approx(5.f).margin(1e-4));
  CHECK(k.outputs.estimate.value.y == Approx(-3.f).margin(1e-4));
  CHECK(k.outputs.velocity.value.x == Approx(0.f).margin(1e-4));
  CHECK(k.outputs.velocity.value.y == Approx(0.f).margin(1e-4));

  // Same after an explicit reset to a different point.
  k.inputs.reset.value = true;
  k.inputs.measurement.value = {-2.f, 8.f};
  k();
  CHECK(k.outputs.estimate.value.x == Approx(-2.f).margin(1e-4));
  CHECK(k.outputs.estimate.value.y == Approx(8.f).margin(1e-4));
}

TEST_CASE("Kalman control input shifts the prediction", "[kalman]")
{
  // Two filters driven by the same static measurement; one gets a control input.
  cv::Kalman a, b;
  for(auto* k : {&a, &b})
  {
    k->inputs.q.value = 0.01f;
    k->inputs.r.value = 0.1f;
    k->inputs.measurement.value = {1.f, 1.f};
  }

  // Seed both from the first measurement.
  a();
  b();

  // Zero control => identical evolution to the no-control case.
  a.inputs.control.value = {0.f, 0.f};
  b.inputs.control.value = {0.5f, -0.25f};
  for(int i = 0; i < 5; ++i)
  {
    a();
    b();
  }

  // The control input pushes b's estimate the same direction as the control vector,
  // away from a's (which only sees the measurement).
  CHECK(b.outputs.estimate.value.x > a.outputs.estimate.value.x);
  CHECK(b.outputs.estimate.value.y < a.outputs.estimate.value.y);
}

TEST_CASE("Kalman zero control reproduces the static-convergence behaviour", "[kalman]")
{
  cv::Kalman k;
  k.inputs.q.value = 0.01f;
  k.inputs.r.value = 0.05f;
  k.inputs.measurement.value = {0.3f, 0.7f};
  k.inputs.control.value = {0.f, 0.f};

  for(int i = 0; i < 60; ++i)
    k();

  CHECK(k.outputs.estimate.value.x == Approx(0.3f).margin(0.02));
  CHECK(k.outputs.estimate.value.y == Approx(0.7f).margin(0.02));
  CHECK(std::abs(k.outputs.velocity.value.x) < 0.05f);
  CHECK(std::abs(k.outputs.velocity.value.y) < 0.05f);
}

// ------------------------------------------------------------------------ Homography
namespace
{
struct H
{
  std::array<float, 9> m;
  std::pair<float, float> apply(float x, float y) const
  {
    float u = m[0] * x + m[1] * y + m[2];
    float v = m[3] * x + m[4] * y + m[5];
    float w = m[6] * x + m[7] * y + m[8];
    return {u / w, v / w};
  }
};
}

TEST_CASE("Homography maps the 4 correspondences (affine)", "[homography]")
{
  cv::Homography h;
  // unit square -> scaled/translated rectangle
  h.inputs.src0.value = {0.f, 0.f};
  h.inputs.src1.value = {1.f, 0.f};
  h.inputs.src2.value = {1.f, 1.f};
  h.inputs.src3.value = {0.f, 1.f};
  h.inputs.dst0.value = {2.f, 3.f};
  h.inputs.dst1.value = {4.f, 3.f};
  h.inputs.dst2.value = {4.f, 7.f};
  h.inputs.dst3.value = {0.f, 7.f}; // intentionally a general quad

  h();
  REQUIRE(h.outputs.valid.value);

  H hm{h.outputs.matrix.value};
  auto [x0, y0] = hm.apply(0.f, 0.f);
  CHECK(x0 == Approx(2.f).margin(1e-3));
  CHECK(y0 == Approx(3.f).margin(1e-3));
  auto [x2, y2] = hm.apply(1.f, 1.f);
  CHECK(x2 == Approx(4.f).margin(1e-3));
  CHECK(y2 == Approx(7.f).margin(1e-3));
}

TEST_CASE("Homography maps a perspective quad", "[homography]")
{
  cv::Homography h;
  h.inputs.src0.value = {0.f, 0.f};
  h.inputs.src1.value = {1.f, 0.f};
  h.inputs.src2.value = {1.f, 1.f};
  h.inputs.src3.value = {0.f, 1.f};
  h.inputs.dst0.value = {0.0f, 0.0f};
  h.inputs.dst1.value = {1.0f, 0.0f};
  h.inputs.dst2.value = {0.8f, 1.0f};
  h.inputs.dst3.value = {0.2f, 1.0f};
  h();
  REQUIRE(h.outputs.valid.value);

  H hm{h.outputs.matrix.value};
  auto [x2, y2] = hm.apply(1.f, 1.f);
  CHECK(x2 == Approx(0.8f).margin(1e-3));
  CHECK(y2 == Approx(1.0f).margin(1e-3));
}

TEST_CASE("Homography recovers an over-determined consistent set (N>4)", "[homography]")
{
  cv::Homography h;
  // Ground-truth projective transform.
  const std::array<float, 9> G{1.2f, 0.1f, 0.3f, -0.2f, 0.9f, 0.4f, 0.05f, -0.03f, 1.f};
  auto apply = [&](float x, float y) {
    float u = G[0] * x + G[1] * y + G[2];
    float v = G[3] * x + G[4] * y + G[5];
    float w = G[6] * x + G[7] * y + G[8];
    return std::pair<float, float>{u / w, v / w};
  };

  // 6 source points (4 mandatory + 2 extra non-zero pairs).
  std::array<std::pair<float, float>, 6> sp{
      {{1.f, 1.f}, {3.f, 1.f}, {3.f, 4.f}, {1.f, 4.f}, {2.f, 2.5f}, {2.5f, 1.5f}}};

  auto set = [&](auto& sport, auto& dport, std::pair<float, float> s) {
    sport.value = {s.first, s.second};
    auto d = apply(s.first, s.second);
    dport.value = {d.first, d.second};
  };
  set(h.inputs.src0, h.inputs.dst0, sp[0]);
  set(h.inputs.src1, h.inputs.dst1, sp[1]);
  set(h.inputs.src2, h.inputs.dst2, sp[2]);
  set(h.inputs.src3, h.inputs.dst3, sp[3]);
  set(h.inputs.src4, h.inputs.dst4, sp[4]);
  set(h.inputs.src5, h.inputs.dst5, sp[5]);
  h.inputs.use4.value = true;
  h.inputs.use5.value = true;

  h();
  REQUIRE(h.outputs.valid.value);

  H hm{h.outputs.matrix.value};
  for(auto& s : sp)
  {
    auto [ux, uy] = hm.apply(s.first, s.second);
    auto [gx, gy] = apply(s.first, s.second);
    CHECK(ux == Approx(gx).margin(1e-3));
    CHECK(uy == Approx(gy).margin(1e-3));
  }
}

TEST_CASE("Homography least-squares fits a noisy over-determined set", "[homography]")
{
  cv::Homography h;
  const std::array<float, 9> G{1.1f, 0.05f, 0.2f, -0.1f, 0.95f, 0.3f, 0.02f, -0.01f, 1.f};
  auto apply = [&](float x, float y) {
    float u = G[0] * x + G[1] * y + G[2];
    float v = G[3] * x + G[4] * y + G[5];
    float w = G[6] * x + G[7] * y + G[8];
    return std::pair<float, float>{u / w, v / w};
  };

  std::array<std::pair<float, float>, 8> sp{
      {{0.f + 1.f, 1.f}, {3.f, 1.f}, {3.f, 4.f}, {1.f, 4.f},
       {2.f, 2.5f}, {2.5f, 1.5f}, {1.5f, 3.f}, {2.8f, 3.5f}}};

  // Deterministic small "noise" added to the destination points.
  auto noise = [](int i, int c) {
    return ((((i * 7 + c * 13) % 5) - 2) * 0.002f);
  };

  auto& sport = h.inputs.src0;
  (void)sport;
  auto setN = [&](auto& sp_, auto& dp_, std::pair<float, float> s, int i) {
    sp_.value = {s.first, s.second};
    auto d = apply(s.first, s.second);
    dp_.value = {d.first + noise(i, 0), d.second + noise(i, 1)};
  };
  setN(h.inputs.src0, h.inputs.dst0, sp[0], 0);
  setN(h.inputs.src1, h.inputs.dst1, sp[1], 1);
  setN(h.inputs.src2, h.inputs.dst2, sp[2], 2);
  setN(h.inputs.src3, h.inputs.dst3, sp[3], 3);
  setN(h.inputs.src4, h.inputs.dst4, sp[4], 4);
  setN(h.inputs.src5, h.inputs.dst5, sp[5], 5);
  setN(h.inputs.src6, h.inputs.dst6, sp[6], 6);
  setN(h.inputs.src7, h.inputs.dst7, sp[7], 7);
  h.inputs.use4.value = true;
  h.inputs.use5.value = true;
  h.inputs.use6.value = true;
  h.inputs.use7.value = true;

  h();
  REQUIRE(h.outputs.valid.value);

  // The least-squares fit should stay close to ground truth despite the noise.
  H hm{h.outputs.matrix.value};
  for(auto& s : sp)
  {
    auto [ux, uy] = hm.apply(s.first, s.second);
    auto [gx, gy] = apply(s.first, s.second);
    CHECK(ux == Approx(gx).margin(0.02));
    CHECK(uy == Approx(gy).margin(0.02));
  }
}

// --------------------------------------------------------------------------- SolvePnP
namespace
{
// Project a 3D point with a known pose + intrinsics (matches the object's pinhole model).
struct Cam
{
  float fx, fy, cx, cy;
};
}

TEST_CASE("SolvePnP recovers a known pose", "[pnp]")
{
  cv::SolvePnP p;
  // Intrinsics
  p.inputs.fx.value = 1.2f;
  p.inputs.fy.value = 1.2f;
  p.inputs.cx.value = 0.0f;
  p.inputs.cy.value = 0.0f;

  // 4 non-coplanar object points
  std::array<std::array<float, 3>, 4> obj{
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}};

  // Ground-truth pose: small rotation about Y + translation in +Z.
  const float ang = 0.3f; // radians
  const float ca = std::cos(ang), sa = std::sin(ang);
  auto rot = [&](std::array<float, 3> v) {
    return std::array<float, 3>{ca * v[0] + sa * v[2], v[1], -sa * v[0] + ca * v[2]};
  };
  const std::array<float, 3> t{0.2f, -0.1f, 5.f};

  std::array<std::array<float, 2>, 4> img;
  for(int i = 0; i < 4; ++i)
  {
    auto r = rot(obj[i]);
    float X = r[0] + t[0], Y = r[1] + t[1], Z = r[2] + t[2];
    img[i] = {1.2f * X / Z + 0.0f, 1.2f * Y / Z + 0.0f};
  }

  p.inputs.obj0.value = {obj[0][0], obj[0][1], obj[0][2]};
  p.inputs.obj1.value = {obj[1][0], obj[1][1], obj[1][2]};
  p.inputs.obj2.value = {obj[2][0], obj[2][1], obj[2][2]};
  p.inputs.obj3.value = {obj[3][0], obj[3][1], obj[3][2]};
  p.inputs.img0.value = {img[0][0], img[0][1]};
  p.inputs.img1.value = {img[1][0], img[1][1]};
  p.inputs.img2.value = {img[2][0], img[2][1]};
  p.inputs.img3.value = {img[3][0], img[3][1]};

  p();

  REQUIRE(p.outputs.valid.value);
  // Translation should be recovered close to ground truth.
  CHECK(p.outputs.translation.value.x == Approx(t[0]).margin(0.05));
  CHECK(p.outputs.translation.value.y == Approx(t[1]).margin(0.05));
  CHECK(p.outputs.translation.value.z == Approx(t[2]).margin(0.1));
}

TEST_CASE("SolvePnP rotation matrix is a valid rotation consistent with the quat", "[pnp]")
{
  cv::SolvePnP p;
  p.inputs.fx.value = 1.2f;
  p.inputs.fy.value = 1.2f;
  p.inputs.cx.value = 0.0f;
  p.inputs.cy.value = 0.0f;

  std::array<std::array<float, 3>, 4> obj{
      {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}};

  const float ang = 0.3f;
  const float ca = std::cos(ang), sa = std::sin(ang);
  auto rot = [&](std::array<float, 3> v) {
    return std::array<float, 3>{ca * v[0] + sa * v[2], v[1], -sa * v[0] + ca * v[2]};
  };
  const std::array<float, 3> t{0.2f, -0.1f, 5.f};

  std::array<std::array<float, 2>, 4> img;
  for(int i = 0; i < 4; ++i)
  {
    auto r = rot(obj[i]);
    float X = r[0] + t[0], Y = r[1] + t[1], Z = r[2] + t[2];
    img[i] = {1.2f * X / Z, 1.2f * Y / Z};
  }

  p.inputs.obj0.value = {obj[0][0], obj[0][1], obj[0][2]};
  p.inputs.obj1.value = {obj[1][0], obj[1][1], obj[1][2]};
  p.inputs.obj2.value = {obj[2][0], obj[2][1], obj[2][2]};
  p.inputs.obj3.value = {obj[3][0], obj[3][1], obj[3][2]};
  p.inputs.img0.value = {img[0][0], img[0][1]};
  p.inputs.img1.value = {img[1][0], img[1][1]};
  p.inputs.img2.value = {img[2][0], img[2][1]};
  p.inputs.img3.value = {img[3][0], img[3][1]};

  p();
  REQUIRE(p.outputs.valid.value);

  const auto& m = p.outputs.matrix.value;
  // Row-major 3x3. Columns must be orthonormal.
  auto col = [&](int c) {
    return std::array<float, 3>{m[c], m[3 + c], m[6 + c]};
  };
  auto dot = [](std::array<float, 3> a, std::array<float, 3> b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  for(int c = 0; c < 3; ++c)
    CHECK(dot(col(c), col(c)) == Approx(1.0).margin(1e-3));
  CHECK(dot(col(0), col(1)) == Approx(0.0).margin(1e-3));
  CHECK(dot(col(0), col(2)) == Approx(0.0).margin(1e-3));
  CHECK(dot(col(1), col(2)) == Approx(0.0).margin(1e-3));

  // det(R) ~ +1
  const float det
      = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
  CHECK(det == Approx(1.0).margin(1e-3));

  // The recovered matrix should be the known ground-truth rotation (about Y by ang).
  CHECK(m[0] == Approx(ca).margin(1e-2)); // R(0,0)
  CHECK(m[2] == Approx(sa).margin(1e-2)); // R(0,2)
  CHECK(m[8] == Approx(ca).margin(1e-2)); // R(2,2)

  // Consistency with the quaternion (x,y,z,w): rebuild R from the quat and compare.
  const auto& q = p.outputs.quat.value;
  const float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  std::array<float, 9> rq{
      1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
      2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
      2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)};
  for(int i = 0; i < 9; ++i)
    CHECK(rq[i] == Approx(m[i]).margin(1e-2));
}

TEST_CASE("SolvePnP rejects degenerate input", "[pnp]")
{
  cv::SolvePnP p;
  p.inputs.fx.value = 1.f;
  p.inputs.fy.value = 1.f;
  // All object points identical -> degenerate.
  p.inputs.obj0.value = {0.f, 0.f, 0.f};
  p.inputs.obj1.value = {0.f, 0.f, 0.f};
  p.inputs.obj2.value = {0.f, 0.f, 0.f};
  p.inputs.obj3.value = {0.f, 0.f, 0.f};
  p.inputs.img0.value = {0.f, 0.f};
  p.inputs.img1.value = {0.1f, 0.f};
  p.inputs.img2.value = {0.f, 0.1f};
  p.inputs.img3.value = {0.1f, 0.1f};
  p();
  CHECK_FALSE(p.outputs.valid.value);
}
