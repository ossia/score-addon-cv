// Tests for buffer-readback objects (decode SSBO contents -> values) and the
// Learn/Recognize classifier pair.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <CV/Readback/CentroidReadback.hpp>
#include <CV/Readback/MomentsReadback.hpp>
#include <CV/Readback/PointListReadback.hpp>
#include <CV/Cpu/Learn.hpp>
#include <CV/Cpu/Recognize.hpp>

#include <CV/Support/Mahalanobis.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using Catch::Approx;

namespace
{
// Point a cpu_buffer_input at a vector of uint32 words.
template <typename BufIn>
void feedU32(BufIn& in, std::vector<std::uint32_t>& words)
{
  in.buffer.raw_data = reinterpret_cast<unsigned char*>(words.data());
  in.buffer.byte_size = static_cast<std::int64_t>(words.size() * sizeof(std::uint32_t));
  in.buffer.byte_offset = 0;
  in.buffer.changed = true;
}
}

// ----------------------------------------------------------------- CentroidReadback
TEST_CASE("CentroidReadback decodes sumXW/sumYW/sumW", "[readback][centroid]")
{
  cv::CentroidReadback r;
  // Layout: {sumW, sumXW, sumYW, count}. Pick sumW=1000, centroid at (0.25, 0.75).
  std::vector<std::uint32_t> buf{1000u, 250u, 750u, 500u};
  feedU32(r.inputs.buffer, buf);
  r();

  REQUIRE(r.outputs.valid.value);
  CHECK(r.outputs.center.value.x == Approx(0.25f).margin(1e-4));
  CHECK(r.outputs.center.value.y == Approx(0.75f).margin(1e-4));
  CHECK(r.outputs.mass.value == Approx(1000.f / 255.f).margin(1e-3));
}

TEST_CASE("CentroidReadback reports invalid on empty mass", "[readback][centroid]")
{
  cv::CentroidReadback r;
  std::vector<std::uint32_t> buf{0u, 0u, 0u, 0u};
  feedU32(r.inputs.buffer, buf);
  r();
  CHECK_FALSE(r.outputs.valid.value);
  CHECK(r.outputs.center.value.x == Approx(-1.f));
}

TEST_CASE("CentroidReadback handles a too-small buffer", "[readback][centroid]")
{
  cv::CentroidReadback r;
  std::vector<std::uint32_t> buf{1000u}; // only 1 word
  feedU32(r.inputs.buffer, buf);
  r();
  CHECK_FALSE(r.outputs.valid.value);
}

// ------------------------------------------------------------------ MomentsReadback
TEST_CASE("MomentsReadback yields a centroid and unit-trace Hu[0]", "[readback][moments]")
{
  cv::MomentsReadback r;
  // Construct moments consistent with a small symmetric distribution.
  // We feed plausible raw-moment integers; the centroid must be m10/m00, m01/m00.
  // m00=1000, m10=300, m01=600 -> centroid (0.3, 0.6).
  // Fill 2nd/3rd order with small values so Hu is finite.
  std::vector<std::uint32_t> buf{
      1000u, 300u, 600u, // m00, m10, m01
      200u, 110u, 380u,  // m11, m20, m02
      40u, 70u, 120u, 240u, // m30, m21, m12, m03
      64u, 64u};            // w, h
  feedU32(r.inputs.buffer, buf);
  r();

  REQUIRE(r.outputs.valid.value);
  CHECK(r.outputs.center.value.x == Approx(0.3f).margin(1e-3));
  CHECK(r.outputs.center.value.y == Approx(0.6f).margin(1e-3));
  // mass == m00 (the 0th moment / total weight), decoded raw.
  CHECK(r.outputs.mass.value == Approx(1000.f));
  // Hu[0] = eta20 + eta02 is positive for any non-degenerate blob.
  CHECK(r.outputs.hu.value[0] > 0.f);
}

TEST_CASE("MomentsReadback reports mass=0 on zero mass", "[readback][moments]")
{
  cv::MomentsReadback r;
  std::vector<std::uint32_t> buf(12, 0u);
  feedU32(r.inputs.buffer, buf);
  r();
  CHECK_FALSE(r.outputs.valid.value);
  CHECK(r.outputs.mass.value == Approx(0.f));
}

// ---------------------------------------------------------------- PointListReadback
TEST_CASE("PointListReadback decodes {count, coords[]}", "[readback][points]")
{
  cv::PointListReadback r;
  // count = 3, then 3 (x,y) float pairs. Pack as raw bytes: 1 uint32 + 6 floats.
  std::vector<std::uint8_t> raw(sizeof(std::uint32_t) + 6 * sizeof(float));
  std::uint32_t n = 3;
  std::memcpy(raw.data(), &n, sizeof(n));
  float coords[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  std::memcpy(raw.data() + sizeof(std::uint32_t), coords, sizeof(coords));

  r.inputs.buffer.buffer.raw_data = raw.data();
  r.inputs.buffer.buffer.byte_size = static_cast<std::int64_t>(raw.size());
  r();

  REQUIRE(r.outputs.count.value == 3);
  REQUIRE(r.outputs.points.value.size() == 3u);
  CHECK(r.outputs.points.value[0].x == Approx(0.1f));
  CHECK(r.outputs.points.value[1].y == Approx(0.4f));
  CHECK(r.outputs.points.value[2].x == Approx(0.5f));
}

TEST_CASE("PointListReadback clamps a count past buffer capacity", "[readback][points]")
{
  cv::PointListReadback r;
  std::vector<std::uint8_t> raw(sizeof(std::uint32_t) + 2 * sizeof(float));
  std::uint32_t n = 999; // claims 999 points but only room for 1
  std::memcpy(raw.data(), &n, sizeof(n));
  float coords[2] = {0.5f, 0.5f};
  std::memcpy(raw.data() + sizeof(std::uint32_t), coords, sizeof(coords));
  r.inputs.buffer.buffer.raw_data = raw.data();
  r.inputs.buffer.buffer.byte_size = static_cast<std::int64_t>(raw.size());
  r();
  // Should clamp to the 1 pair that actually fits, never read past the buffer.
  CHECK(r.outputs.points.value.size() == 1u);
}

// ------------------------------------------------------------------- Learn/Recognize
namespace
{
void pulse(cv::Learn& l, bool& flag)
{
  flag = false;
  l();
  flag = true;
  l();
}
}

TEST_CASE("Learn trains a model and Recognize scores Mahalanobis", "[learn][recognize]")
{
  cv::Learn learn;

  std::mt19937 rng(42);
  std::normal_distribution<float> nx(2.f, 1.f), ny(5.f, 1.4f);

  // Capture 200 samples from a 2D Gaussian.
  for(int i = 0; i < 200; ++i)
  {
    learn.inputs.feature.value = {nx(rng), ny(rng)};
    pulse(learn, learn.inputs.capture.value);
  }
  pulse(learn, learn.inputs.train.value);

  REQUIRE(learn.outputs.valid.value);
  REQUIRE(learn.outputs.dimension.value == 2);
  REQUIRE(learn.outputs.samples.value == 200);
  // Recovered mean near (2,5).
  REQUIRE(learn.outputs.mean.value.size() == 2u);
  CHECK(learn.outputs.mean.value[0] == Approx(2.f).margin(0.3));
  CHECK(learn.outputs.mean.value[1] == Approx(5.f).margin(0.4));

  cv::Recognize rec;
  rec.inputs.mean.value = learn.outputs.mean.value;
  rec.inputs.invcov.value = learn.outputs.invcov.value;
  rec.inputs.threshold.value = 3.f;

  // A sample at the mean -> ~0 distance, match.
  rec.inputs.feature.value = learn.outputs.mean.value;
  rec();
  CHECK(rec.outputs.distance.value == Approx(0.f).margin(0.05));
  CHECK(rec.outputs.match.value);

  // A far outlier -> large distance, no match.
  rec.inputs.feature.value = {20.f, 20.f};
  rec();
  CHECK(rec.outputs.distance.value > 3.f);
  CHECK_FALSE(rec.outputs.match.value);
}

TEST_CASE("Learn rejects too few samples", "[learn]")
{
  cv::Learn learn;
  learn.inputs.feature.value = {1.f, 2.f};
  pulse(learn, learn.inputs.capture.value); // only 1 sample for dim 2
  pulse(learn, learn.inputs.train.value);
  CHECK_FALSE(learn.outputs.valid.value);
}

TEST_CASE("Recognize dimension mismatch yields a large distance, not 0", "[recognize]")
{
  cv::Recognize rec;
  rec.inputs.mean.value = {0.f, 0.f};
  rec.inputs.invcov.value = {1.f, 0.f, 0.f, 1.f};
  rec.inputs.feature.value = {1.f, 2.f, 3.f}; // wrong dim
  rec();
  // A dimension mismatch must NOT read as a perfect match (distance 0).
  CHECK_FALSE(rec.outputs.match.value);
  CHECK_FALSE(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value > 1e6f);
  CHECK(rec.outputs.distance.value == cv::Recognize::invalid_distance);
}

TEST_CASE("Recognize reports valid=true on a well-formed match", "[recognize]")
{
  cv::Recognize rec;
  rec.inputs.mean.value = {0.f, 0.f};
  rec.inputs.invcov.value = {1.f, 0.f, 0.f, 1.f};
  rec.inputs.feature.value = {0.f, 0.f};
  rec.inputs.threshold.value = 3.f;
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(0.f).margin(1e-5));
  CHECK(rec.outputs.match.value);
}

TEST_CASE("Recognize Model inlet overrides mean/invcov", "[recognize][model]")
{
  cv::Recognize rec;
  // Deliberately wrong separate mean/invcov; the packed model must win.
  rec.inputs.mean.value = {99.f};
  rec.inputs.invcov.value = {1.f};
  rec.inputs.model.value = cv_support::pack_model(
      2, {0.f, 0.f}, {1.f, 0.f, 0.f, 1.f});
  rec.inputs.feature.value = {3.f, 4.f}; // distance = 5
  rec.inputs.threshold.value = 10.f;
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(5.f).margin(1e-4));
  CHECK(rec.outputs.match.value);
}

TEST_CASE("Recognize rejects a malformed Model", "[recognize][model]")
{
  cv::Recognize rec;
  rec.inputs.model.value = {2.f, 0.f}; // claims dim 2 but truncated
  rec.inputs.feature.value = {0.f, 0.f};
  rec();
  CHECK_FALSE(rec.outputs.valid.value);
  CHECK_FALSE(rec.outputs.match.value);
  CHECK(rec.outputs.distance.value == cv::Recognize::invalid_distance);
}

namespace
{
// Train a Learn object on a fixed Gaussian and return it.
cv::Learn make_trained_learn()
{
  cv::Learn learn;
  std::mt19937 rng(1234);
  std::normal_distribution<float> nx(2.f, 1.f), ny(5.f, 1.4f);
  for(int i = 0; i < 200; ++i)
  {
    learn.inputs.feature.value = {nx(rng), ny(rng)};
    pulse(learn, learn.inputs.capture.value);
  }
  pulse(learn, learn.inputs.train.value);
  return learn;
}

float distance_for(
    const std::vector<float>& mean, const std::vector<float>& invcov,
    const std::vector<float>& feature)
{
  cv::Recognize rec;
  rec.inputs.mean.value = mean;
  rec.inputs.invcov.value = invcov;
  rec.inputs.feature.value = feature;
  rec();
  return rec.outputs.distance.value;
}
}

TEST_CASE("Learn Model output round-trips through Recognize", "[learn][recognize][model]")
{
  cv::Learn learn = make_trained_learn();
  REQUIRE(learn.outputs.valid.value);
  REQUIRE_FALSE(learn.outputs.model.value.empty());

  const std::vector<float> feat{3.5f, 6.f};
  const float refDist = distance_for(
      learn.outputs.mean.value, learn.outputs.invcov.value, feat);

  // Feed only the packed Model into a fresh Recognize.
  cv::Recognize rec;
  rec.inputs.model.value = learn.outputs.model.value;
  rec.inputs.feature.value = feat;
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(refDist).margin(1e-4));
}

TEST_CASE("Learn save -> load reproduces the model", "[learn][persistence]")
{
  cv::Learn learn = make_trained_learn();
  REQUIRE(learn.outputs.valid.value);

  const auto refMean = learn.outputs.mean.value;
  const auto refInvcov = learn.outputs.invcov.value;
  const auto refModel = learn.outputs.model.value;

  const std::vector<float> feat{3.5f, 6.f};
  const float refDist = distance_for(refMean, refInvcov, feat);

  // Unique temp path.
  const std::string path
      = "/tmp/cv_learn_model_" + std::to_string(::getpid()) + ".cvlm";
  learn.inputs.file.value = path;

  // Save (rising edge).
  pulse(learn, learn.inputs.save.value);

  // Load into a fresh object.
  cv::Learn loaded;
  loaded.inputs.file.value = path;
  pulse(loaded, loaded.inputs.load.value);

  REQUIRE(loaded.outputs.valid.value);
  REQUIRE(loaded.outputs.dimension.value == learn.outputs.dimension.value);
  REQUIRE(loaded.outputs.mean.value.size() == refMean.size());
  for(std::size_t i = 0; i < refMean.size(); ++i)
    CHECK(loaded.outputs.mean.value[i] == Approx(refMean[i]));
  REQUIRE(loaded.outputs.invcov.value.size() == refInvcov.size());
  for(std::size_t i = 0; i < refInvcov.size(); ++i)
    CHECK(loaded.outputs.invcov.value[i] == Approx(refInvcov[i]));
  CHECK(loaded.outputs.model.value == refModel);

  // Distances identical before/after.
  const float loadedDist = distance_for(
      loaded.outputs.mean.value, loaded.outputs.invcov.value, feat);
  CHECK(loadedDist == Approx(refDist).margin(1e-5));

  std::remove(path.c_str());
}

TEST_CASE("Learn load of a missing file leaves outputs untouched", "[learn][persistence]")
{
  cv::Learn loaded;
  loaded.inputs.file.value = "/tmp/cv_nonexistent_model_does_not_exist.cvlm";
  pulse(loaded, loaded.inputs.load.value);
  CHECK_FALSE(loaded.outputs.valid.value);
}
