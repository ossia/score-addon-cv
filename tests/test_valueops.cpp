// Tests for the value-domain (no texture) objects: Noise, Lowpass.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <CV/Cpu/Lowpass.hpp>
#include <CV/Cpu/Noise.hpp>

#include <cmath>
#include <numbers>
#include <vector>

using Catch::Approx;

namespace
{
// The coefficient hand-computed from the cv.jit.lowpass patch, independently of Lowpass.hpp.
double patch_b(double cutoff, double rate)
{
  const double f = std::cos(2.0 * std::numbers::pi * (cutoff / rate));
  return f - 1.0 + std::sqrt((2.0 - f) * (2.0 - f) - 1.0);
}

std::vector<std::vector<float>> run(cv::Noise& n, int ticks)
{
  std::vector<std::vector<float>> res;
  for(int i = 0; i < ticks; i++)
  {
    n();
    res.push_back(n.outputs.values.value);
  }
  return res;
}
}

// ============================================================================== Noise

TEST_CASE("Noise emits one value per dimension", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 1234;

  n.inputs.dims.value = 1;
  n();
  CHECK(n.outputs.values.value.size() == 1);

  n.inputs.dims.value = 7;
  n();
  CHECK(n.outputs.values.value.size() == 7);

  // Clamped to [1, 128].
  n.inputs.dims.value = 0;
  n();
  CHECK(n.outputs.values.value.size() == 1);

  n.inputs.dims.value = 5000;
  n();
  CHECK(n.outputs.values.value.size() == 128);
}

TEST_CASE("Noise: the same seed reproduces the identical sequence", "[noise]")
{
  auto make = [](int seed) {
    cv::Noise n;
    n.inputs.seed.value = seed;
    n.inputs.dims.value = 4;
    n.inputs.min.value = {-5.f};
    n.inputs.max.value = {5.f};
    return run(n, 20);
  };

  const auto a = make(42);
  const auto b = make(42);
  const auto c = make(43);

  REQUIRE(a.size() == 20);
  REQUIRE(a == b);

  // Different seeds must produce a different stream (astronomically unlikely to collide).
  CHECK(a != c);
}

TEST_CASE("Noise: setting the seed control mid-stream restarts the stream", "[noise]")
{
  cv::Noise n;
  n.inputs.dims.value = 3;
  n.inputs.seed.value = 7;
  n();
  const auto first = n.outputs.values.value;

  n();
  n();
  // Re-applying the same value is not a change; force a real change back to 7.
  n.inputs.seed.value = 99;
  n();
  n.inputs.seed.value = 7;
  n();
  CHECK(n.outputs.values.value == first);
}

TEST_CASE("Noise: uniform mode stays within [min, max]", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 555;
  n.inputs.dims.value = 2;
  n.inputs.mode.value = {float(cv::Noise::Uniform)};
  n.inputs.min.value = {-3.5f};
  n.inputs.max.value = {8.25f};

  float lo = 1e30f, hi = -1e30f;
  for(int i = 0; i < 5000; i++)
  {
    n();
    for(float v : n.outputs.values.value)
    {
      CHECK(v >= -3.5f);
      CHECK(v <= 8.25f);
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  // ... and actually covers the range rather than sitting on a constant.
  CHECK(lo < -3.f);
  CHECK(hi > 7.75f);
}

TEST_CASE("Noise: reversed uniform bounds are sorted, not malformed", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 3;
  n.inputs.dims.value = 1;
  n.inputs.min.value = {10.f}; // min > max
  n.inputs.max.value = {2.f};

  for(int i = 0; i < 500; i++)
  {
    n();
    const float v = n.outputs.values.value[0];
    CHECK(v >= 2.f);
    CHECK(v <= 10.f);
  }
}

TEST_CASE("Noise: normal mode matches the requested mean and stddev", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 20240101;
  n.inputs.dims.value = 1;
  n.inputs.mode.value = {float(cv::Noise::Normal)};
  n.inputs.mean.value = {12.f};
  n.inputs.stddev.value = {3.f};

  constexpr int N = 40000;
  double sum = 0., sum2 = 0.;
  for(int i = 0; i < N; i++)
  {
    n();
    const double v = n.outputs.values.value[0];
    sum += v;
    sum2 += v * v;
  }
  const double mean = sum / N;
  const double var = sum2 / N - mean * mean;
  const double stddev = std::sqrt(var);

  // Standard error of the mean is 3/sqrt(40000) = 0.015; be generous.
  CHECK(mean == Approx(12.0).margin(0.15));
  CHECK(stddev == Approx(3.0).margin(0.15));
}

TEST_CASE("Noise: per-dimension parameters are honoured independently", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 88;
  n.inputs.dims.value = 3;
  // dim0: uniform [0,1]; dim1: uniform [100,200]; dim2: normal(mean -50, stddev 0.5)
  n.inputs.mode.value
      = {float(cv::Noise::Uniform), float(cv::Noise::Uniform), float(cv::Noise::Normal)};
  n.inputs.min.value = {0.f, 100.f, 0.f};
  n.inputs.max.value = {1.f, 200.f, 1.f};
  n.inputs.mean.value = {0.f, 0.f, -50.f};
  n.inputs.stddev.value = {1.f, 1.f, 0.5f};

  double s2 = 0.;
  constexpr int N = 2000;
  for(int i = 0; i < N; i++)
  {
    n();
    const auto& v = n.outputs.values.value;
    REQUIRE(v.size() == 3);
    CHECK(v[0] >= 0.f);
    CHECK(v[0] <= 1.f);
    CHECK(v[1] >= 100.f);
    CHECK(v[1] <= 200.f);
    // 8 sigma either side of the mean: never hit in practice.
    CHECK(v[2] > -54.f);
    CHECK(v[2] < -46.f);
    s2 += v[2];
  }
  CHECK(s2 / N == Approx(-50.0).margin(0.1));
}

TEST_CASE("Noise: the last supplied parameter repeats over the remaining dims", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 11;
  n.inputs.dims.value = 5;
  // Only two values for five dimensions: dims 2..4 inherit dim1's [100, 200].
  n.inputs.min.value = {0.f, 100.f};
  n.inputs.max.value = {1.f, 200.f};

  for(int i = 0; i < 500; i++)
  {
    n();
    const auto& v = n.outputs.values.value;
    REQUIRE(v.size() == 5);
    CHECK(v[0] >= 0.f);
    CHECK(v[0] <= 1.f);
    for(int d = 1; d < 5; d++)
    {
      CHECK(v[d] >= 100.f);
      CHECK(v[d] <= 200.f);
    }
  }
}

TEST_CASE("Noise: a single-element parameter list configures every dimension", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 12;
  n.inputs.dims.value = 6;
  n.inputs.min.value = {-2.f};
  n.inputs.max.value = {-1.f};

  for(int i = 0; i < 200; i++)
  {
    n();
    REQUIRE(n.outputs.values.value.size() == 6);
    for(float v : n.outputs.values.value)
    {
      CHECK(v >= -2.f);
      CHECK(v <= -1.f);
    }
  }
}

TEST_CASE("Noise: growing dims replicates the last stored parameter", "[noise]")
{
  cv::Noise n;
  n.inputs.seed.value = 13;
  n.inputs.dims.value = 2;
  n.inputs.min.value = {0.f, 100.f};
  n.inputs.max.value = {1.f, 200.f};
  n();
  REQUIRE(n.outputs.values.value.size() == 2);

  // Grow without touching the parameter lists: the new dims must inherit dim1's range,
  // not the defaults [0,1].
  n.inputs.min.value.clear();
  n.inputs.max.value.clear();
  n.inputs.dims.value = 4;

  for(int i = 0; i < 300; i++)
  {
    n();
    const auto& v = n.outputs.values.value;
    REQUIRE(v.size() == 4);
    CHECK(v[0] >= 0.f);
    CHECK(v[0] <= 1.f);
    for(int d = 1; d < 4; d++)
    {
      CHECK(v[d] >= 100.f);
      CHECK(v[d] <= 200.f);
    }
  }
}

TEST_CASE("Noise: the seed message reseeds", "[noise]")
{
  cv::Noise n;
  n.inputs.dims.value = 2;
  n.inputs.seed.value = 4321;
  n();
  const auto first = n.outputs.values.value;
  n.seed(); // wall-clock reseed: the stream diverges
  n();
  // A wall-clock reseed producing exactly the same pair of doubles is not credible.
  CHECK(n.outputs.values.value != first);
}

// ============================================================================ Lowpass

TEST_CASE("Lowpass: the first sample is exactly b * x", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 5.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {1.f};

  lp();

  const double b = patch_b(5.0, 60.0);
  REQUIRE(b > 0.0);
  REQUIRE(b < 1.0);
  REQUIRE(lp.outputs.out.value.size() == 1);
  CHECK(lp.outputs.out.value[0] == Approx(b).epsilon(1e-6));

  // Second sample: y = b*x + (1-b)*y1
  lp();
  CHECK(lp.outputs.out.value[0] == Approx(b + (1.0 - b) * b).epsilon(1e-6));
}

TEST_CASE("Lowpass: known coefficient at another cutoff/rate", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 1.f;
  lp.inputs.rate.value = 100.f;
  lp.inputs.in.value = {4.f};
  lp();
  CHECK(lp.outputs.out.value[0] == Approx(4.0 * patch_b(1.0, 100.0)).epsilon(1e-6));
}

TEST_CASE("Lowpass: a constant input converges monotonically to that constant", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 2.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {10.f};

  float prev = -1e30f;
  for(int i = 0; i < 400; i++)
  {
    lp();
    const float v = lp.outputs.out.value[0];
    // Never overshoots and never goes back down.
    CHECK(v >= prev);
    CHECK(v <= 10.f);
    // Strictly increasing while it is still visibly climbing (float saturates at 10 later).
    if(i < 50)
      CHECK(v > prev);
    prev = v;
  }
  CHECK(prev == Approx(10.f).margin(0.001));
}

TEST_CASE("Lowpass: a higher cutoff converges faster", "[lowpass]")
{
  auto after = [](float cutoff, int ticks) {
    cv::Lowpass lp;
    lp.inputs.cutoff.value = cutoff;
    lp.inputs.rate.value = 60.f;
    lp.inputs.in.value = {1.f};
    for(int i = 0; i < ticks; i++)
      lp();
    return lp.outputs.out.value[0];
  };

  const float slow = after(0.5f, 10);
  const float mid = after(2.f, 10);
  const float fast = after(10.f, 10);

  CHECK(slow < mid);
  CHECK(mid < fast);
  CHECK(fast < 1.0001f);
  CHECK(slow > 0.f);
}

TEST_CASE("Lowpass: a zero cutoff blocks entirely", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 0.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {1.f};
  for(int i = 0; i < 50; i++)
    lp();
  CHECK(lp.outputs.out.value[0] == Approx(0.f).margin(1e-9));
}

TEST_CASE("Lowpass: reset / clear drop the filter memory", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 5.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {1.f};

  for(int i = 0; i < 200; i++)
    lp();
  REQUIRE(lp.outputs.out.value[0] == Approx(1.f).margin(0.001));

  const double b = patch_b(5.0, 60.0);

  lp.reset();
  lp();
  CHECK(lp.outputs.out.value[0] == Approx(b).epsilon(1e-6));

  for(int i = 0; i < 200; i++)
    lp();
  REQUIRE(lp.outputs.out.value[0] == Approx(1.f).margin(0.001));

  lp.clear();
  lp();
  CHECK(lp.outputs.out.value[0] == Approx(b).epsilon(1e-6));
}

TEST_CASE("Lowpass: list elements are filtered independently", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 3.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {1.f, -2.f, 100.f, 0.f};

  for(int i = 0; i < 500; i++)
    lp();

  const auto& o = lp.outputs.out.value;
  REQUIRE(o.size() == 4);
  CHECK(o[0] == Approx(1.f).margin(0.001));
  CHECK(o[1] == Approx(-2.f).margin(0.001));
  CHECK(o[2] == Approx(100.f).margin(0.01));
  CHECK(o[3] == Approx(0.f).margin(0.001));

  // Each element follows exactly the same trajectory scaled by its own input: run a fresh
  // filter on a single element and compare tick by tick.
  cv::Lowpass a, b2;
  a.inputs.cutoff.value = b2.inputs.cutoff.value = 3.f;
  a.inputs.rate.value = b2.inputs.rate.value = 60.f;
  a.inputs.in.value = {7.f, -1.f};
  b2.inputs.in.value = {-1.f};
  for(int i = 0; i < 25; i++)
  {
    a();
    b2();
    CHECK(a.outputs.out.value[1] == Approx(b2.outputs.out.value[0]).epsilon(1e-6));
  }
}

TEST_CASE("Lowpass: an empty input list does not crash", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 5.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {};
  for(int i = 0; i < 5; i++)
    lp();
  CHECK(lp.outputs.out.value.empty());

  // ... and the filter still works afterwards.
  lp.inputs.in.value = {1.f};
  lp();
  CHECK(lp.outputs.out.value.size() == 1);
  CHECK(lp.outputs.out.value[0] == Approx(patch_b(5.0, 60.0)).epsilon(1e-6));
}

TEST_CASE("Lowpass: a growing list starts new elements from zero", "[lowpass]")
{
  cv::Lowpass lp;
  lp.inputs.cutoff.value = 5.f;
  lp.inputs.rate.value = 60.f;
  lp.inputs.in.value = {1.f};
  for(int i = 0; i < 200; i++)
    lp();
  REQUIRE(lp.outputs.out.value[0] == Approx(1.f).margin(0.001));

  lp.inputs.in.value = {1.f, 1.f};
  lp();
  const auto& o = lp.outputs.out.value;
  REQUIRE(o.size() == 2);
  CHECK(o[0] == Approx(1.f).margin(0.001));       // kept its history
  CHECK(o[1] == Approx(patch_b(5.0, 60.0)).epsilon(1e-6)); // started from zero
}

TEST_CASE("Lowpass: a cutoff above Nyquist is clamped rather than folding back", "[lowpass]")
{
  // Unclamped, cutoff == rate gives cos(2*pi) == 1 and therefore b == 0: the filter would
  // silently block everything. Clamped at rate/2 it is the fastest available setting.
  auto after = [](float cutoff) {
    cv::Lowpass lp;
    lp.inputs.cutoff.value = cutoff;
    lp.inputs.rate.value = 60.f;
    lp.inputs.in.value = {1.f};
    lp();
    return lp.outputs.out.value[0];
  };

  const float nyq = after(30.f);
  CHECK(nyq == Approx(patch_b(30.0, 60.0)).epsilon(1e-6));
  CHECK(after(60.f) == Approx(nyq).epsilon(1e-6));
  CHECK(after(1000.f) == Approx(nyq).epsilon(1e-6));
}
