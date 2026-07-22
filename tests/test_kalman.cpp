// Tests for the N-dimensional cv.jit.kalman port (CV/Cpu/Kalman.hpp).
//
// The fixed-2D fallback path (xy ports) is covered by tests/test_geometry.cpp; everything
// here exercises the cv.jit object proper: the list input, the auto-sized state, the
// asymmetric transition matrix, the two weight attributes and the Prediction/Corrected
// outputs.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <CV/Cpu/Kalman.hpp>

#include <cmath>
#include <vector>

using Catch::Approx;

namespace
{
void feed(cv::Kalman& k, std::vector<float> m)
{
  k.inputs.measurement_list.value = std::move(m);
  k();
}
}

// ---------------------------------------------------------------- transition matrix
//
// cv.jit.kalman rebuilds the transition matrix on every list from
//     weights = { 1, velocity_weight, acceleration_weight }
//     for (i = 0; i < D; i++)
//       for (j = i, k = 0; j < D && k < 3; j += M, k++)
//         A(i, j) = weights[k];
//
// For M = 2 (D = 6), wv = 0.5, wa = 0.25 that unrolls to
//   i=0: A(0,0)=1  A(0,2)=wv A(0,4)=wa
//   i=1: A(1,1)=1  A(1,3)=wv A(1,5)=wa
//   i=2: A(2,2)=1  A(2,4)=wv          (j=6 is out of range)
//   i=3: A(3,3)=1  A(3,5)=wv
//   i=4: A(4,4)=1
//   i=5: A(5,5)=1
//
// i.e. the asymmetric block form
//     [ I   wv*I  wa*I ]
//     [ 0    I    wv*I ]
//     [ 0    0     I   ]
// Note what this is NOT: the acceleration block feeds the velocity block with wv, not wa,
// and the position block gets wa (not the textbook 0.5*dt^2). Every other entry is zero,
// so a symmetric or textbook constant-acceleration A fails this test.
TEST_CASE("Kalman transition matrix is cv.jit's asymmetric block form", "[kalman][nd]")
{
  cv::Kalman k;
  k.inputs.vel_weight.value = 0.5f;
  k.inputs.accel_weight.value = 0.25f;

  feed(k, {1.f, 2.f});

  REQUIRE(k.measurement_dims() == 2);
  REQUIRE(k.state_dims() == 6);

  const float expected[6][6] = {
      {1.f, 0.f, 0.5f, 0.f, 0.25f, 0.f},
      {0.f, 1.f, 0.f, 0.5f, 0.f, 0.25f},
      {0.f, 0.f, 1.f, 0.f, 0.5f, 0.f},
      {0.f, 0.f, 0.f, 1.f, 0.f, 0.5f},
      {0.f, 0.f, 0.f, 0.f, 1.f, 0.f},
      {0.f, 0.f, 0.f, 0.f, 0.f, 1.f},
  };

  const auto& A = k.transition_matrix();
  REQUIRE(A.rows() == 6);
  REQUIRE(A.cols() == 6);
  for(int i = 0; i < 6; i++)
  {
    for(int j = 0; j < 6; j++)
    {
      INFO("A(" << i << "," << j << ")");
      CHECK(A(i, j) == Approx(expected[i][j]).margin(1e-7));
    }
  }

  // Explicitly: the acceleration -> velocity coupling uses wv, and the matrix is not
  // symmetric.
  CHECK(A(2, 4) == Approx(0.5f).margin(1e-7)); // wv, not wa
  CHECK(A(4, 2) == Approx(0.f).margin(1e-7));
  CHECK(A(0, 4) == Approx(0.25f).margin(1e-7));

  // measurementMatrix = setIdentity() on the M x D matrix: only the positions are observed.
  const auto& H = k.measurement_matrix();
  REQUIRE(H.rows() == 2);
  REQUIRE(H.cols() == 6);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 6; j++)
      CHECK(H(i, j) == Approx(i == j ? 1.f : 0.f).margin(1e-7));
}

// ------------------------------------------------------------------- N-D auto-sizing
TEST_CASE("Kalman sizes itself from the measurement list length", "[kalman][nd]")
{
  SECTION("3 elements")
  {
    cv::Kalman k;
    feed(k, {1.f, 2.f, 3.f});
    CHECK(k.measurement_dims() == 3);
    CHECK(k.state_dims() == 9);
    REQUIRE(k.outputs.state.value.size() == 3u);
  }

  SECTION("5 elements")
  {
    cv::Kalman k;
    feed(k, {1.f, 2.f, 3.f, 4.f, 5.f});
    CHECK(k.measurement_dims() == 5);
    CHECK(k.state_dims() == 15);
    REQUIRE(k.outputs.state.value.size() == 5u);
  }
}

// The first list after an init seeds the position block exactly. (cv.jit writes the seed to
// statePre, where predict() immediately discards it; this port writes it to statePost too so
// it survives -- see the header's "deliberate deviations".)
TEST_CASE("Kalman seeds the position block from the first list", "[kalman][nd]")
{
  const std::vector<float> z{5.f, -3.f, 7.25f, 0.5f, -100.f};

  for(auto mode : {cv::KalmanOutput::Corrected, cv::KalmanOutput::Prediction})
  {
    cv::Kalman k;
    k.inputs.output.value = mode;
    feed(k, z);

    REQUIRE(k.outputs.state.value.size() == 5u);
    for(int i = 0; i < 5; i++)
      CHECK(k.outputs.state.value[i] == Approx(z[i]).margin(1e-6));

    // Position block seeded, velocity + acceleration blocks still exactly zero.
    const auto& s = k.corrected_state();
    REQUIRE(s.size() == 15);
    for(int i = 0; i < 5; i++)
      CHECK(s(i) == Approx(z[i]).margin(1e-6));
    for(int i = 5; i < 15; i++)
      CHECK(s(i) == Approx(0.f).margin(1e-6));
  }
}

TEST_CASE("Kalman re-initialises when the list length changes mid-stream", "[kalman][nd]")
{
  cv::Kalman k;

  // Run a 3-D stream for a while so there is real state to become stale.
  for(int t = 0; t < 20; t++)
    feed(k, {1.f + 0.1f * t, 2.f + 0.2f * t, 3.f - 0.05f * t});
  REQUIRE(k.measurement_dims() == 3);
  REQUIRE(k.state_dims() == 9);

  // Switch to a 5-element list: full re-init, so the new list is a seed again and the
  // output is exactly the measurement (no trace of the 3-D run).
  const std::vector<float> z5{-8.f, 0.f, 42.f, 0.125f, -0.5f};
  feed(k, z5);

  CHECK(k.measurement_dims() == 5);
  CHECK(k.state_dims() == 15);
  REQUIRE(k.outputs.state.value.size() == 5u);
  for(int i = 0; i < 5; i++)
    CHECK(k.outputs.state.value[i] == Approx(z5[i]).margin(1e-6));
  for(int i = 5; i < 15; i++)
    CHECK(k.corrected_state()(i) == Approx(0.f).margin(1e-6));

  // And back down to 2 elements.
  const std::vector<float> z2{3.f, -3.f};
  feed(k, z2);
  CHECK(k.measurement_dims() == 2);
  CHECK(k.state_dims() == 6);
  REQUIRE(k.outputs.state.value.size() == 2u);
  for(int i = 0; i < 2; i++)
    CHECK(k.outputs.state.value[i] == Approx(z2[i]).margin(1e-6));

  // Growing again after shrinking must not read stale memory either.
  for(int t = 0; t < 5; t++)
    feed(k, {3.f, -3.f});
  feed(k, {1.f, 2.f, 3.f, 4.f});
  CHECK(k.measurement_dims() == 4);
  REQUIRE(k.outputs.state.value.size() == 4u);
  CHECK(k.outputs.state.value[3] == Approx(4.f).margin(1e-6));
}

// --------------------------------------------------------------------- weight semantics
//
// weights = {1, 0, 0} makes A the identity: a constant-signal model. Seeded from the first
// list, a constant input produces a zero innovation forever, so the output is *exactly* the
// input at every step, whatever the noise settings.
TEST_CASE("Kalman with zero weights is a constant-signal model", "[kalman][nd]")
{
  cv::Kalman k;
  k.inputs.vel_weight.value = 0.f;
  k.inputs.accel_weight.value = 0.f;

  const std::vector<float> z{2.5f, -1.75f, 0.f};
  for(int t = 0; t < 50; t++)
  {
    feed(k, z);
    REQUIRE(k.outputs.state.value.size() == 3u);
    for(int i = 0; i < 3; i++)
      CHECK(k.outputs.state.value[i] == Approx(z[i]).margin(1e-6));
  }

  // A = I exactly.
  const auto& A = k.transition_matrix();
  for(int i = 0; i < 9; i++)
    for(int j = 0; j < 9; j++)
      CHECK(A(i, j) == Approx(i == j ? 1.f : 0.f).margin(1e-7));

  // The velocity and acceleration blocks never move away from zero: nothing drives them.
  for(int i = 3; i < 9; i++)
    CHECK(std::abs(k.corrected_state()(i)) < 1e-5f);
}

// acceleration_weight = 0 (with velocity_weight = 1, i.e. one step of velocity per tick) is a
// constant-velocity model and must follow a linear ramp far better than the constant-signal
// model does.
//
// Derivation of the constant-signal model's lag. With A = I the position state decouples from
// the rest (P stays diagonal, K has a single non-zero component), so it reduces to the scalar
// filter P- = P+ + q, K = P-/(P- + r), P+ = (1-K)P-. In steady state K*P- = q and
// P-^2 - q*P- - q*r = 0; with q = 0.01, r = 0.1 that gives P- = 0.037015 and K = 0.27016.
// A scalar filter chasing a ramp of slope s lags by s*(1-K)/K = 0.1 * 0.72984/0.27016 = 0.270.
// The constant-velocity model has no such structural lag: its velocity state converges to the
// slope and the residual error decays to ~0.
TEST_CASE("Kalman with acceleration_weight = 0 is constant-velocity", "[kalman][nd]")
{
  cv::Kalman cvel, cpos;
  for(auto* k : {&cvel, &cpos})
  {
    k->inputs.q.value = 0.01f;
    k->inputs.r.value = 0.1f;
    k->inputs.accel_weight.value = 0.f;
    // The lag derived above is that of the corrected estimate.
    k->inputs.output.value = cv::KalmanOutput::Corrected;
  }
  cvel.inputs.vel_weight.value = 1.f;  // constant velocity
  cpos.inputs.vel_weight.value = 0.f;  // constant signal (A = I)

  const float slope = 0.1f;
  float last = 0.f;
  for(int t = 0; t < 120; t++)
  {
    last = slope * t;
    feed(cvel, {last});
    feed(cpos, {last});
  }

  const float e_vel = std::abs(cvel.outputs.state.value[0] - last);
  const float e_pos = std::abs(cpos.outputs.state.value[0] - last);

  INFO("constant-velocity error " << e_vel << ", constant-signal error " << e_pos);
  // The constant-signal model sits at its predicted structural lag of ~0.27.
  CHECK(e_pos == Approx(0.270f).margin(0.02));
  // The constant-velocity model tracks the ramp essentially exactly.
  CHECK(e_vel < 0.005f);
  CHECK(e_vel < 0.05f * e_pos);

  // Its velocity state has converged onto the ramp slope.
  CHECK(cvel.corrected_state()(1) == Approx(slope).margin(0.005));
}

// ------------------------------------------------------------- Prediction vs Corrected
//
// Hand-computed one-step case. M = 1 (D = 3), velocity_weight = acceleration_weight = 0 so
// A = I, q = 0.01, r = 0.1, errorCovPost starts at I.
//
// List 1, z = 2: seeds the position block, so statePost = [2,0,0].
//   predict : statePre = A*statePost = [2,0,0]; P- = P+ + q*I = 1.01*I
//   correct : S = 1.01 + 0.1 = 1.11, K0 = 1.01/1.11 = 0.90990991
//             innovation = 2 - 2 = 0  =>  statePost stays [2,0,0]
//             P+(0,0) = (1 - K0)*1.01 = (0.1/1.11)*1.01 = 0.101/1.11 = 0.09099099
//             P is diagonal throughout, so K has a single non-zero component.
//
// List 2, z = 6:
//   predict : statePre = A*statePost = [2,0,0]      <-- the Prediction output: exactly 2
//             P-(0,0) = 0.09099099 + 0.01 = 0.10099099
//   correct : S = 0.10099099 + 0.1 = 0.20099099
//             K0 = 0.10099099/0.20099099 = 0.50246516
//             statePost(0) = 2 + K0*(6 - 2) = 2 + 2.00986064 = 4.00986064
//                                              <-- the Corrected output
TEST_CASE("Kalman Prediction is the pre-update estimate", "[kalman][nd]")
{
  cv::Kalman pred, corr;
  for(auto* k : {&pred, &corr})
  {
    k->inputs.q.value = 0.01f;
    k->inputs.r.value = 0.1f;
    k->inputs.vel_weight.value = 0.f;
    k->inputs.accel_weight.value = 0.f;
  }
  pred.inputs.output.value = cv::KalmanOutput::Prediction;
  corr.inputs.output.value = cv::KalmanOutput::Corrected;

  feed(pred, {2.f});
  feed(corr, {2.f});
  REQUIRE(pred.outputs.state.value[0] == Approx(2.f).margin(1e-6));
  REQUIRE(corr.outputs.state.value[0] == Approx(2.f).margin(1e-6));

  feed(pred, {6.f});
  feed(corr, {6.f});

  CHECK(pred.outputs.state.value[0] == Approx(2.f).margin(1e-5));
  CHECK(corr.outputs.state.value[0] == Approx(4.00986064f).margin(1e-4));

  // The two modes really do differ, and both instances hold the same internal state.
  CHECK(pred.outputs.state.value[0] != Approx(corr.outputs.state.value[0]).margin(1e-3));
  CHECK(pred.predicted_state()(0) == Approx(2.f).margin(1e-5));
  CHECK(pred.corrected_state()(0) == Approx(4.00986064f).margin(1e-4));
  CHECK(corr.predicted_state()(0) == Approx(2.f).margin(1e-5));

  // Prediction is what cv.jit.kalman outputs; on a moving signal it always trails the
  // correction, never sits on top of it.
  cv::Kalman p2;
  p2.inputs.output.value = cv::KalmanOutput::Prediction;
  p2.inputs.vel_weight.value = 0.f;
  p2.inputs.accel_weight.value = 0.f;
  feed(p2, {0.f});
  for(int t = 1; t < 10; t++)
  {
    feed(p2, {static_cast<float>(t)});
    CHECK(p2.outputs.state.value[0] < p2.corrected_state()(0));
  }
}

// --------------------------------------------------------------------- control input
//
// cv.jit's control input is inert (it allocates a 1 x C row vector where OpenCV wants C x 1,
// and never sets controlMatrix). This port injects it for real, onto the position block.
TEST_CASE("Kalman N-D control list shifts the prediction", "[kalman][nd]")
{
  cv::Kalman a, b;
  for(auto* k : {&a, &b})
  {
    k->inputs.q.value = 0.01f;
    k->inputs.r.value = 0.1f;
    k->inputs.output.value = cv::KalmanOutput::Prediction;
  }

  feed(a, {1.f, 1.f, 1.f});
  feed(b, {1.f, 1.f, 1.f});

  b.inputs.control_list.value = {0.5f, -0.25f, 0.f};
  for(int i = 0; i < 4; i++)
  {
    feed(a, {1.f, 1.f, 1.f});
    feed(b, {1.f, 1.f, 1.f});
  }

  CHECK(b.outputs.state.value[0] > a.outputs.state.value[0]);
  CHECK(b.outputs.state.value[1] < a.outputs.state.value[1]);
  CHECK(b.outputs.state.value[2] == Approx(a.outputs.state.value[2]).margin(1e-5));

  // A control list of the wrong length is ignored (it does not re-init the filter).
  b.inputs.control_list.value = {1.f, 1.f};
  const int dims_before = b.measurement_dims();
  feed(b, {1.f, 1.f, 1.f});
  CHECK(b.measurement_dims() == dims_before);
}

// ----------------------------------------------------------------------- clear / reset
TEST_CASE("Kalman clear/reset restore the seeded-from-first-list behaviour", "[kalman][nd]")
{
  auto run_stream = [](cv::Kalman& k) {
    for(int t = 0; t < 30; t++)
      feed(k, {10.f + t, -10.f - t, 5.f});
  };

  SECTION("clear message")
  {
    cv::Kalman k;
    run_stream(k);
    REQUIRE(k.outputs.state.value[0] > 20.f);

    k.clear();
    CHECK(k.measurement_dims() == 0);
    CHECK(k.state_dims() == 0);

    feed(k, {-1.f, -2.f, -3.f});
    CHECK(k.outputs.state.value[0] == Approx(-1.f).margin(1e-6));
    CHECK(k.outputs.state.value[1] == Approx(-2.f).margin(1e-6));
    CHECK(k.outputs.state.value[2] == Approx(-3.f).margin(1e-6));
    for(int i = 3; i < 9; i++)
      CHECK(k.corrected_state()(i) == Approx(0.f).margin(1e-6));
  }

  SECTION("reset message")
  {
    cv::Kalman k;
    run_stream(k);
    k.reset();
    feed(k, {-1.f, -2.f, -3.f});
    CHECK(k.outputs.state.value[0] == Approx(-1.f).margin(1e-6));
    CHECK(k.outputs.state.value[2] == Approx(-3.f).margin(1e-6));
  }

  SECTION("Reset toggle, on the rising edge only")
  {
    cv::Kalman k;
    run_stream(k);

    k.inputs.reset.value = true;
    feed(k, {-1.f, -2.f, -3.f});
    CHECK(k.outputs.state.value[0] == Approx(-1.f).margin(1e-6));

    // Held high: no further reset, so the next list is filtered, not seeded.
    feed(k, {50.f, 50.f, 50.f});
    CHECK(k.outputs.state.value[0] < 50.f);

    // Falling then rising again re-arms it.
    k.inputs.reset.value = false;
    feed(k, {50.f, 50.f, 50.f});
    k.inputs.reset.value = true;
    feed(k, {7.f, 8.f, 9.f});
    CHECK(k.outputs.state.value[0] == Approx(7.f).margin(1e-6));
    CHECK(k.outputs.state.value[1] == Approx(8.f).margin(1e-6));
    CHECK(k.outputs.state.value[2] == Approx(9.f).margin(1e-6));
  }
}

// ------------------------------------------------------------------------- degenerate
TEST_CASE("Kalman handles an empty measurement list", "[kalman][nd]")
{
  cv::Kalman k;

  // Empty from the start: the fixed-2D fallback path runs on the xy ports.
  k.inputs.measurement.value = {0.25f, -0.5f};
  k();
  CHECK(k.measurement_dims() == 0);
  CHECK(k.outputs.estimate.value.x == Approx(0.25f).margin(1e-6));
  CHECK(k.outputs.estimate.value.y == Approx(-0.5f).margin(1e-6));
  REQUIRE(k.outputs.state.value.size() == 2u);

  // Empty after an N-D stream: no crash, no stale read, and the N-D state is untouched.
  feed(k, {1.f, 2.f, 3.f});
  feed(k, {1.1f, 2.1f, 3.1f});
  REQUIRE(k.measurement_dims() == 3);
  const float kept = k.corrected_state()(0);

  k.inputs.measurement_list.value.clear();
  k();
  CHECK(k.measurement_dims() == 3);
  CHECK(k.corrected_state()(0) == Approx(kept).margin(1e-6));
  REQUIRE(k.outputs.state.value.size() == 2u);

  // ... and the N-D stream picks up where it left off.
  feed(k, {1.2f, 2.2f, 3.2f});
  CHECK(k.measurement_dims() == 3);
  REQUIRE(k.outputs.state.value.size() == 3u);
}

TEST_CASE("Kalman handles a single-element list", "[kalman][nd]")
{
  cv::Kalman k;
  feed(k, {3.f});
  CHECK(k.measurement_dims() == 1);
  CHECK(k.state_dims() == 3);
  REQUIRE(k.outputs.state.value.size() == 1u);
  CHECK(k.outputs.state.value[0] == Approx(3.f).margin(1e-6));
  // The xy mirror leaves y at zero for a 1-D filter.
  CHECK(k.outputs.estimate.value.x == Approx(3.f).margin(1e-6));
  CHECK(k.outputs.estimate.value.y == Approx(0.f).margin(1e-6));
}

// The Output enum applies to the fixed-2D fallback path too. Same hand computation as above
// but on the 4-state constant-velocity model: the first measurement seeds [x, y, 0, 0], the
// second tick predicts x + vx*dt = 2 (velocity still 0) before correcting toward 6.
TEST_CASE("Kalman Output enum applies to the 2D fallback path", "[kalman][2d]")
{
  cv::Kalman pred, corr;
  for(auto* k : {&pred, &corr})
  {
    k->inputs.q.value = 0.01f;
    k->inputs.r.value = 0.1f;
    k->inputs.measurement.value = {2.f, 2.f};
  }
  pred.inputs.output.value = cv::KalmanOutput::Prediction;
  corr.inputs.output.value = cv::KalmanOutput::Corrected;
  pred();
  corr();

  pred.inputs.measurement.value = {6.f, 6.f};
  corr.inputs.measurement.value = {6.f, 6.f};
  pred();
  corr();

  CHECK(pred.outputs.estimate.value.x == Approx(2.f).margin(1e-5));
  CHECK(corr.outputs.estimate.value.x > 3.5f);
  CHECK(corr.outputs.estimate.value.x > pred.outputs.estimate.value.x);
}

TEST_CASE("Kalman weight attributes default to cv.jit's 0.001", "[kalman][nd]")
{
  cv::Kalman k;
  CHECK(k.inputs.vel_weight.value == Approx(0.001f));
  CHECK(k.inputs.accel_weight.value == Approx(0.001f));
  // cv.jit hardcodes processNoiseCov = 1e-5 and measurementNoiseCov = 1e-1.
  CHECK(k.inputs.q.value == Approx(1e-5f));
  CHECK(k.inputs.r.value == Approx(1e-1f));
  // cv.jit.kalman outputs the *prediction*, so that is the default here too.
  CHECK(k.inputs.output.value == cv::KalmanOutput::Prediction);
}
