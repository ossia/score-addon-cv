// cv.jit-compatibility tests for the Learn / Recognize pair.
//
// Three things are pinned down here, all of which the port used to get wrong:
//
//  1. THE ESTIMATOR. cv.jit.learn is a true online / incremental estimator:
//     one update per learned sample, model queryable from sample 1, no training
//     step (cv.jit has no such message). Its update, verbatim from
//     cv.jit/source/projects/cv.jit.learn/cv.jit.learn.cpp:
//
//         index += 1;  a = 1/index;  b = 1 - a;
//         mean[i]    = data[i]*a + mean[i]*b;
//         cov[i*n+j] = ((data[i]-mean[i]) * (data[j]-mean[j]))*a + cov[i*n+j]*b;
//         inverse    = pseudo-inverse(cov)          // SVD, on EVERY sample
//
//     The covariance line uses the mean that was JUST updated with the same
//     sample, and the weight is a running 1/index blend, not 1/(N-1). Both are
//     "wrong" textbook-wise and both are hand-traced below precisely so nobody
//     "fixes" them.
//
//     Hand trace, samples (1,2) then (3,6) then (5,4):
//
//       step 1: index 1, a = 1, b = 0
//               mean = (1*1 + 0*0, 2*1 + 0*0)              = (1, 2)
//               dev  = data - NEW mean                     = (0, 0)
//               cov  = 0                                   (exactly zero)
//               inverse = pinv(0)                          = 0
//       step 2: index 2, a = 1/2, b = 1/2
//               mean = (3/2 + 1/2, 6/2 + 2/2)              = (2, 4)
//               dev  = (3,6) - (2,4)                       = (1, 2)
//               cov  = outer(dev)*1/2 + 0*1/2
//                    = [[1*1, 1*2],[2*1, 2*2]]/2           = [[0.5, 1],[1, 2]]
//               (a textbook step would use the OLD mean (1,2), giving
//                dev = (2,4) and cov(0,0) = 2, not 0.5)
//               cov is rank 1 -> det 0; pinv of c*w*w^T with w = (1,2)/sqrt(5),
//               c = 2.5 is (1/c)*w*w^T                     = [[.08,.16],[.16,.32]]
//       step 3: index 3, a = 1/3, b = 2/3
//               mean = (5/3 + 2*2/3, 4/3 + 4*2/3)          = (3, 4)
//               dev  = (5,4) - (3,4)                       = (2, 0)
//               cov(0,0) = 4/3 + 0.5*2/3                   = 5/3
//               cov(0,1) = 0   + 1.0*2/3                   = 2/3
//               cov(1,1) = 0   + 2.0*2/3                   = 4/3
//               det = 20/9 - 4/9 = 16/9
//               inverse = (9/16)*[[4/3, -2/3],[-2/3, 5/3]]
//                       = [[0.75, -0.375],[-0.375, 0.9375]]
//
//     For contrast, over the SAME three points:
//       unbiased 1/(N-1) covariance  = [[4, 2], [2, 4]]
//       population 1/N covariance    = [[8/3, 4/3], [4/3, 8/3]]
//     Neither is what cv.jit produces; both are asserted against below.
//
//  2. THE FILE FORMAT. cv.jit's `.mxb`:
//       int32 magic 'cvjt' | int32 size | double index
//       | double mean[n] | double covariance[n*n] | double inverse[n*n]
//     856 bytes for n == 7. The four-char code is written as a native int32, so
//     the bytes on disk are "cvjt" for a big-endian payload and "tjvc" for a
//     little-endian one; both must read. The two bundled fixtures
//     cv.jit/misc/The_letter_{A,M}.mxb are big-endian (PPC-era) and are loaded
//     here for real.
//
//  3. RESUMABLE TRAINING. `index` is in the file, so loading a model and
//     learning more samples continues the 1/index blend from where the file
//     left off (83 -> 84 -> 85 for The_letter_A) instead of restarting at 1.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <CV/Cpu/Learn.hpp>
#include <CV/Cpu/Recognize.hpp>
#include <CV/Support/Mahalanobis.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using Catch::Approx;

namespace
{
// The addon root, derived from this file's absolute path (both test runners
// compile with absolute sources), so the bundled cv.jit fixtures can be found
// regardless of the working directory.
std::string addon_dir()
{
  std::string f = __FILE__;
  const auto p = f.find_last_of('/');
  std::string tests = (p == std::string::npos) ? std::string(".") : f.substr(0, p);
  const auto q = tests.find_last_of('/');
  return (q == std::string::npos) ? std::string("..") : tests.substr(0, q);
}

std::string fixture(const char* name)
{
  return addon_dir() + "/cv.jit/misc/" + name;
}

std::string tmp_path(const char* tag)
{
  return std::string("/tmp/cv_learn_test_") + tag + "_" + std::to_string(::getpid())
         + ".mxb";
}

// Rising edge on a toggle: low tick, then high tick.
void pulse(cv::Learn& l, bool& flag)
{
  flag = false;
  l();
  flag = true;
  l();
}

void learn_one(cv::Learn& l, const std::vector<float>& v)
{
  l.inputs.feature.value = v;
  pulse(l, l.inputs.capture.value);
}
}

// =========================================================== 1. online update
TEST_CASE("Online update reproduces cv.jit.learn step by step", "[learn][online]")
{
  cv_support::OnlineModel m;

  // ---- sample 1: (1, 2)
  REQUIRE(m.update({1.f, 2.f}));
  CHECK(m.index == 1.0);
  REQUIRE(m.size() == 2);
  CHECK(m.mean(0) == Approx(1.0).margin(1e-12));
  CHECK(m.mean(1) == Approx(2.0).margin(1e-12));
  // The deviation is taken from the mean that was just set to the sample, so
  // the covariance is EXACTLY zero after one sample (and so is its inverse).
  CHECK(m.cov.cwiseAbs().maxCoeff() == 0.0);
  CHECK(m.inverse.cwiseAbs().maxCoeff() == 0.0);

  // ---- sample 2: (3, 6)
  REQUIRE(m.update({3.f, 6.f}));
  CHECK(m.index == 2.0);
  CHECK(m.mean(0) == Approx(2.0).margin(1e-12));
  CHECK(m.mean(1) == Approx(4.0).margin(1e-12));
  CHECK(m.cov(0, 0) == Approx(0.5).margin(1e-12));
  CHECK(m.cov(0, 1) == Approx(1.0).margin(1e-12));
  CHECK(m.cov(1, 0) == Approx(1.0).margin(1e-12));
  CHECK(m.cov(1, 1) == Approx(2.0).margin(1e-12));
  // A textbook step would use the mean BEFORE this sample, i.e. (1,2), giving
  // a deviation of (2,4) and cov(0,0) == 2. That must not be what we compute.
  CHECK(m.cov(0, 0) != Approx(2.0).margin(1e-9));
  // Rank-1 covariance: the SVD pseudo-inverse must not blow up.
  CHECK(m.inverse(0, 0) == Approx(0.08).margin(1e-12));
  CHECK(m.inverse(0, 1) == Approx(0.16).margin(1e-12));
  CHECK(m.inverse(1, 0) == Approx(0.16).margin(1e-12));
  CHECK(m.inverse(1, 1) == Approx(0.32).margin(1e-12));

  // ---- sample 3: (5, 4)
  REQUIRE(m.update({5.f, 4.f}));
  CHECK(m.index == 3.0);
  CHECK(m.mean(0) == Approx(3.0).margin(1e-12));
  CHECK(m.mean(1) == Approx(4.0).margin(1e-12));
  CHECK(m.cov(0, 0) == Approx(5.0 / 3.0).margin(1e-12));
  CHECK(m.cov(0, 1) == Approx(2.0 / 3.0).margin(1e-12));
  CHECK(m.cov(1, 0) == Approx(2.0 / 3.0).margin(1e-12));
  CHECK(m.cov(1, 1) == Approx(4.0 / 3.0).margin(1e-12));
  CHECK(m.inverse(0, 0) == Approx(0.75).margin(1e-9));
  CHECK(m.inverse(0, 1) == Approx(-0.375).margin(1e-9));
  CHECK(m.inverse(1, 0) == Approx(-0.375).margin(1e-9));
  CHECK(m.inverse(1, 1) == Approx(0.9375).margin(1e-9));
}

TEST_CASE("Online covariance is not the textbook covariance", "[learn][online][quirk]")
{
  // Same three points, computed the textbook way for comparison.
  const std::vector<std::vector<float>> samples{{1.f, 2.f}, {3.f, 6.f}, {5.f, 4.f}};

  cv_support::OnlineModel m;
  for(const auto& s : samples)
    REQUIRE(m.update(s));

  // The batch estimator over the same points: unbiased 1/(N-1) covariance.
  const auto batch = cv_support::compute_model(samples);
  REQUIRE(batch.valid);
  REQUIRE(batch.cov(0, 0) == Approx(4.0).margin(1e-9));
  REQUIRE(batch.cov(0, 1) == Approx(2.0).margin(1e-9));
  REQUIRE(batch.cov(1, 1) == Approx(4.0).margin(1e-9));

  // Both agree on the mean (the running mean IS the arithmetic mean)...
  CHECK(m.mean(0) == Approx(batch.mean(0)).margin(1e-9));
  CHECK(m.mean(1) == Approx(batch.mean(1)).margin(1e-9));
  // ... and disagree on every covariance entry. A "fixed" implementation using
  // 1/(N-1) or 1/N would fail here.
  CHECK(m.cov(0, 0) != Approx(batch.cov(0, 0)).margin(1e-6)); // 5/3 vs 4
  CHECK(m.cov(0, 1) != Approx(batch.cov(0, 1)).margin(1e-6)); // 2/3 vs 2
  CHECK(m.cov(1, 1) != Approx(batch.cov(1, 1)).margin(1e-6)); // 4/3 vs 4
  // Population (1/N) covariance would be [[8/3,4/3],[4/3,8/3]]: also wrong.
  CHECK(m.cov(0, 0) != Approx(8.0 / 3.0).margin(1e-6));
  CHECK(m.cov(1, 1) != Approx(8.0 / 3.0).margin(1e-6));
}

TEST_CASE("Online update: second hand trace, 5 samples", "[learn][online]")
{
  // (0,0) (2,0) (0,2) (2,2) (1,1) -- a symmetric cloud whose batch covariance is
  // exactly the identity, so any drift towards the textbook answer is visible.
  //   step 3: mean = (2/3, 2/3), dev = (-2/3, 4/3)
  //           cov(0,0) = (4/9)/3 + 0.5*2/3        = 13/27
  //           cov(0,1) = (-8/9)/3 + 0             = -8/27
  //           cov(1,1) = (16/9)/3 + 0             = 16/27
  //   step 4: mean = (1,1), dev = (1,1)
  //           cov(0,0) = 1/4 + (13/27)*3/4        = 11/18
  //           cov(0,1) = 1/4 + (-8/27)*3/4        = 1/36
  //           cov(1,1) = 1/4 + (16/27)*3/4        = 25/36
  //   step 5: dev = (0,0) -> everything is just scaled by b = 4/5
  //           cov(0,0) = (11/18)*4/5              = 22/45
  //           cov(0,1) = (1/36)*4/5               = 1/45
  //           cov(1,1) = (25/36)*4/5              = 5/9
  cv_support::OnlineModel m;
  for(auto s : {std::vector<float>{0.f, 0.f}, {2.f, 0.f}, {0.f, 2.f}, {2.f, 2.f},
                {1.f, 1.f}})
    REQUIRE(m.update(s));

  CHECK(m.index == 5.0);
  CHECK(m.mean(0) == Approx(1.0).margin(1e-12));
  CHECK(m.mean(1) == Approx(1.0).margin(1e-12));
  CHECK(m.cov(0, 0) == Approx(22.0 / 45.0).margin(1e-12));
  CHECK(m.cov(0, 1) == Approx(1.0 / 45.0).margin(1e-12));
  CHECK(m.cov(1, 0) == Approx(1.0 / 45.0).margin(1e-12));
  CHECK(m.cov(1, 1) == Approx(5.0 / 9.0).margin(1e-12));

  // The batch estimator on the same points gives exactly the identity: proof
  // the two modes are genuinely different estimators, not the same one twice.
  const auto batch = cv_support::compute_model(
      {{0.f, 0.f}, {2.f, 0.f}, {0.f, 2.f}, {2.f, 2.f}, {1.f, 1.f}});
  REQUIRE(batch.valid);
  CHECK(batch.cov(0, 0) == Approx(1.0).margin(1e-12));
  CHECK(batch.cov(0, 1) == Approx(0.0).margin(1e-12));
  CHECK(batch.cov(1, 1) == Approx(1.0).margin(1e-12));
}

TEST_CASE("Learn is queryable after ONE sample", "[learn][online]")
{
  // cv.jit.learn answers a compare request from sample 1 onwards; the port used
  // to refuse anything below dim + 1 samples and emit nothing at all.
  cv::Learn learn;
  REQUIRE(learn.inputs.mode.value == cv::LearnMode::CvJitOnline); // the default

  learn_one(learn, {1.f, 2.f});

  CHECK(learn.outputs.index.value == 1.f);
  CHECK(learn.outputs.samples.value == 1);
  // A model IS published: dimension, mean, inverse covariance and the packed
  // model are all there. No "Train" pulse was needed either.
  REQUIRE(learn.outputs.dimension.value == 2);
  REQUIRE(learn.outputs.mean.value.size() == 2u);
  CHECK(learn.outputs.mean.value[0] == 1.f);
  CHECK(learn.outputs.mean.value[1] == 2.f);
  REQUIRE(learn.outputs.invcov.value.size() == 4u);
  REQUIRE(learn.outputs.covariance.value.size() == 4u);
  REQUIRE_FALSE(learn.outputs.model.value.empty());
  // ... and it is accepted by Recognize (distance 0 everywhere, because the
  // one-sample covariance and thus its pseudo-inverse are exactly zero -- that
  // is cv.jit's answer, not a refusal).
  cv::Recognize rec;
  rec.inputs.model.value = learn.outputs.model.value;
  rec.inputs.feature.value = {17.f, -4.f};
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(0.f).margin(1e-9));
  // The "Valid" flag on Learn is about being able to DISCRIMINATE, and a
  // degenerate one-sample model cannot; the model is still published.
  CHECK_FALSE(learn.outputs.valid.value);

  // A second sample makes the covariance non-degenerate along one axis.
  learn_one(learn, {3.f, 6.f});
  CHECK(learn.outputs.index.value == 2.f);
  CHECK(learn.outputs.valid.value);
  CHECK(learn.outputs.covariance.value[0] == Approx(0.5f).margin(1e-6));
}

TEST_CASE("Learn online mode needs no Train pulse", "[learn][online]")
{
  cv::Learn a, b;
  const std::vector<std::vector<float>> samples{
      {1.f, 2.f}, {3.f, 6.f}, {5.f, 4.f}, {2.f, 3.f}};
  for(const auto& s : samples)
  {
    learn_one(a, s);
    learn_one(b, s);
  }
  pulse(b, b.inputs.train.value); // no-op re-publish in online mode

  CHECK(a.outputs.valid.value);
  CHECK(a.outputs.mean.value == b.outputs.mean.value);
  CHECK(a.outputs.invcov.value == b.outputs.invcov.value);
  CHECK(a.outputs.index.value == b.outputs.index.value);
  CHECK(a.outputs.index.value == 4.f);
}

TEST_CASE("Learn Batch mode is still reachable and is a different estimator",
          "[learn][batch]")
{
  cv::Learn learn;
  learn.inputs.mode.value = cv::LearnMode::Batch;
  for(auto s : {std::vector<float>{0.f, 0.f}, {2.f, 0.f}, {0.f, 2.f}, {2.f, 2.f},
                {1.f, 1.f}})
    learn_one(learn, s);

  // Nothing published until Train: that is the batch contract.
  CHECK_FALSE(learn.outputs.valid.value);
  pulse(learn, learn.inputs.train.value);

  REQUIRE(learn.outputs.valid.value);
  REQUIRE(learn.outputs.dimension.value == 2);
  CHECK(learn.outputs.samples.value == 5);
  CHECK(learn.outputs.mean.value[0] == Approx(1.f).margin(1e-6));
  CHECK(learn.outputs.mean.value[1] == Approx(1.f).margin(1e-6));
  // Unbiased covariance of that cloud is exactly the identity, so is its inverse.
  REQUIRE(learn.outputs.covariance.value.size() == 4u);
  CHECK(learn.outputs.covariance.value[0] == Approx(1.f).margin(1e-6));
  CHECK(learn.outputs.covariance.value[1] == Approx(0.f).margin(1e-6));
  CHECK(learn.outputs.covariance.value[3] == Approx(1.f).margin(1e-6));
  CHECK(learn.outputs.invcov.value[0] == Approx(1.f).margin(1e-6));
  CHECK(learn.outputs.invcov.value[3] == Approx(1.f).margin(1e-6));

  // Batch still refuses a degenerate set (< dim + 1 samples).
  cv::Learn few;
  few.inputs.mode.value = cv::LearnMode::Batch;
  learn_one(few, {1.f, 2.f});
  learn_one(few, {2.f, 4.f});
  pulse(few, few.inputs.train.value);
  CHECK_FALSE(few.outputs.valid.value);
  CHECK(few.outputs.dimension.value == 0);
}

TEST_CASE("Online model restarts when the feature length changes", "[learn][online]")
{
  cv_support::OnlineModel m;
  REQUIRE(m.update({1.f, 2.f}));
  REQUIRE(m.update({3.f, 6.f}));
  REQUIRE(m.index == 2.0);
  REQUIRE(m.update({1.f, 2.f, 3.f}));
  CHECK(m.size() == 3);
  CHECK(m.index == 1.0); // fresh model, not a blend of incompatible statistics
  CHECK(m.mean(2) == Approx(3.0).margin(1e-12));
  CHECK_FALSE(m.update({}));
}

// ============================================================ 2. .mxb round-trip
TEST_CASE("mxb round-trips a known model bit-identically", "[learn][mxb]")
{
  cv_support::MxbModel m;
  m.size = 3;
  m.index = 42.0;
  m.mean = {0.125, -2.5, 1e-9};
  m.cov = {1.0, 0.25, 0.0, 0.25, 2.0, -0.5, 0.0, -0.5, 4.0};
  m.inverse = {0.5, -0.0625, 1.0, -0.0625, 0.5, 0.125, 1.0, 0.125, 0.25};

  const std::string path = tmp_path("roundtrip");
  REQUIRE(cv_support::write_mxb(path, m));

  // Exact byte count of the format: 4 + 4 + 8 + 8*n + 8*n*n + 8*n*n.
  {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    REQUIRE(static_cast<std::size_t>(is.tellg()) == cv_support::mxb_byte_size(3));
    REQUIRE(cv_support::mxb_byte_size(7) == 856u); // the bundled fixtures' size
  }

  cv_support::MxbModel r;
  REQUIRE(cv_support::read_mxb(path, r));
  CHECK(r.size == m.size);
  CHECK(r.index == m.index); // exact, not Approx: doubles must round-trip
  CHECK(r.mean == m.mean);
  CHECK(r.cov == m.cov);
  CHECK(r.inverse == m.inverse);
  std::remove(path.c_str());
}

TEST_CASE("mxb reads both endiannesses identically", "[learn][mxb][endian]")
{
  cv_support::MxbModel m;
  m.size = 2;
  m.index = 9.0;
  m.mean = {1.5, -3.25};
  m.cov = {2.0, 0.5, 0.5, 8.0};
  m.inverse = {0.5079365079365079, -0.031746031746031744, -0.031746031746031744,
               0.12698412698412698};

  const std::string big = tmp_path("big");
  const std::string little = tmp_path("little");
  REQUIRE(cv_support::write_mxb(big, m, cv_support::mxb_endian::big));
  REQUIRE(cv_support::write_mxb(little, m, cv_support::mxb_endian::little));

  // The magic on disk is the literal four-char code in payload order.
  auto magic_of = [](const std::string& p) {
    std::ifstream is(p, std::ios::binary);
    char b[4] = {};
    is.read(b, 4);
    return std::string(b, 4);
  };
  CHECK(magic_of(big) == "cvjt");
  CHECK(magic_of(little) == "tjvc");

  // The payloads are byte-reversed twins...
  {
    std::ifstream ib(big, std::ios::binary), il(little, std::ios::binary);
    std::vector<char> bb((std::istreambuf_iterator<char>(ib)),
                         std::istreambuf_iterator<char>());
    std::vector<char> lb((std::istreambuf_iterator<char>(il)),
                         std::istreambuf_iterator<char>());
    REQUIRE(bb.size() == lb.size());
    CHECK(bb != lb);
    // int32 size field at offset 4 is reversed between them
    CHECK(bb[4] == lb[7]);
    CHECK(bb[7] == lb[4]);
  }

  // ... and they decode identically.
  cv_support::MxbModel rb, rl;
  REQUIRE(cv_support::read_mxb(big, rb));
  REQUIRE(cv_support::read_mxb(little, rl));
  CHECK(rb.size == rl.size);
  CHECK(rb.index == rl.index);
  CHECK(rb.mean == rl.mean);
  CHECK(rb.cov == rl.cov);
  CHECK(rb.inverse == rl.inverse);
  CHECK(rb.mean == m.mean);
  CHECK(rb.inverse == m.inverse);

  std::remove(big.c_str());
  std::remove(little.c_str());
}

TEST_CASE("mxb rejects garbage without crashing", "[learn][mxb]")
{
  cv_support::MxbModel r;

  // Wrong magic.
  {
    const std::string p = tmp_path("badmagic");
    std::ofstream os(p, std::ios::binary);
    const char junk[64] = {'n', 'o', 'p', 'e'};
    os.write(junk, sizeof(junk));
    os.close();
    CHECK_FALSE(cv_support::read_mxb(p, r));

    cv::Learn l;
    l.inputs.file.value = p;
    pulse(l, l.inputs.load.value);
    CHECK_FALSE(l.outputs.valid.value);
    CHECK(l.outputs.dimension.value == 0);
    std::remove(p.c_str());
  }

  // Right magic, truncated payload.
  {
    const std::string p = tmp_path("truncated");
    cv_support::MxbModel m;
    m.size = 2;
    m.index = 3.0;
    m.mean = {1.0, 2.0};
    m.cov = {1.0, 0.0, 0.0, 1.0};
    m.inverse = {1.0, 0.0, 0.0, 1.0};
    REQUIRE(cv_support::write_mxb(p, m));
    {
      std::ifstream is(p, std::ios::binary);
      std::vector<char> buf((std::istreambuf_iterator<char>(is)),
                            std::istreambuf_iterator<char>());
      buf.resize(buf.size() - 16);
      std::ofstream os(p, std::ios::binary | std::ios::trunc);
      os.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    CHECK_FALSE(cv_support::read_mxb(p, r));
    std::remove(p.c_str());
  }

  // Absurd size field: must not try to allocate it.
  {
    const std::string p = tmp_path("hugesize");
    std::ofstream os(p, std::ios::binary);
    const char magic[4] = {'t', 'j', 'v', 'c'};
    const std::int32_t huge = 1 << 30;
    os.write(magic, 4);
    os.write(reinterpret_cast<const char*>(&huge), 4);
    const double pad[8] = {};
    os.write(reinterpret_cast<const char*>(pad), sizeof(pad));
    os.close();
    CHECK_FALSE(cv_support::read_mxb(p, r));
    std::remove(p.c_str());
  }

  // Empty file.
  {
    const std::string p = tmp_path("empty");
    std::ofstream os(p, std::ios::binary);
    os.close();
    CHECK_FALSE(cv_support::read_mxb(p, r));
    std::remove(p.c_str());
  }

  CHECK_FALSE(cv_support::read_mxb("/tmp/cv_definitely_not_here_xyz.mxb", r));
}

TEST_CASE("The bundled cv.jit fixtures load", "[learn][mxb][fixture]")
{
  cv_support::MxbModel A, M;
  REQUIRE(cv_support::read_mxb(fixture("The_letter_A.mxb"), A));
  REQUIRE(cv_support::read_mxb(fixture("The_letter_M.mxb"), M));

  // Both were trained on the 7 moments / Hu invariants of a letter shape.
  REQUIRE(A.size == 7);
  REQUIRE(M.size == 7);
  REQUIRE(A.mean.size() == 7u);
  REQUIRE(A.cov.size() == 49u);
  REQUIRE(A.inverse.size() == 49u);

  // Plausible sample counts: whole numbers, in the tens, and different.
  CHECK(A.index == 83.0);
  CHECK(M.index == 35.0);
  CHECK(A.index != M.index);

  // Spot-check the first mean component of each (nu20 of the letter shape).
  CHECK(A.mean[0] == Approx(0.1676108065).epsilon(1e-8));
  CHECK(M.mean[0] == Approx(0.2272538668).epsilon(1e-8));
  // The two models are genuinely different data.
  CHECK(A.mean != M.mean);
  CHECK(A.cov != M.cov);

  // The stored `inverse` really is the SVD pseudo-inverse of the stored `cov`:
  // our Eigen implementation reproduces OpenCV's cv::invert(..., DECOMP_SVD).
  for(const auto* mdl : {&A, &M})
  {
    Eigen::MatrixXd cov(7, 7), inv(7, 7);
    for(int i = 0; i < 7; ++i)
      for(int j = 0; j < 7; ++j)
      {
        cov(i, j) = mdl->cov[static_cast<std::size_t>(i) * 7 + j];
        inv(i, j) = mdl->inverse[static_cast<std::size_t>(i) * 7 + j];
      }
    const Eigen::MatrixXd recomputed = cv_support::pseudo_inverse(cov);
    const double scale = inv.cwiseAbs().maxCoeff();
    REQUIRE(scale > 0.0);
    CHECK((recomputed - inv).cwiseAbs().maxCoeff() < 1e-6 * scale);
  }

  // Same probe, two models -> two clearly different Mahalanobis distances.
  // Probe = A's own mean: distance 0 from A, far from M.
  cv_support::OnlineModel oa, om;
  REQUIRE(cv_support::from_mxb(A, oa));
  REQUIRE(cv_support::from_mxb(M, om));
  std::vector<float> probe(7);
  for(int i = 0; i < 7; ++i)
    probe[static_cast<std::size_t>(i)] = static_cast<float>(A.mean[static_cast<std::size_t>(i)]);

  const double dA = oa.distance(probe);
  const double dM = om.distance(probe);
  CHECK(dA == Approx(0.0).margin(1e-3));
  CHECK(dM == Approx(17.318196).epsilon(1e-4));
  CHECK(dM > dA + 10.0);

  // And symmetrically with M's mean as the probe.
  for(int i = 0; i < 7; ++i)
    probe[static_cast<std::size_t>(i)] = static_cast<float>(M.mean[static_cast<std::size_t>(i)]);
  CHECK(om.distance(probe) == Approx(0.0).margin(1e-3));
  CHECK(oa.distance(probe) == Approx(10.735610).epsilon(1e-4));
}

TEST_CASE("Learn loads a cv.jit fixture and Recognize scores with it",
          "[learn][mxb][fixture][recognize]")
{
  cv::Learn learn;
  learn.inputs.file.value = fixture("The_letter_A.mxb");
  pulse(learn, learn.inputs.load.value);

  REQUIRE(learn.outputs.valid.value);
  REQUIRE(learn.outputs.dimension.value == 7);
  CHECK(learn.outputs.index.value == 83.f);
  CHECK(learn.outputs.samples.value == 83);
  REQUIRE(learn.outputs.mean.value.size() == 7u);
  REQUIRE(learn.outputs.invcov.value.size() == 49u);
  REQUIRE(learn.outputs.covariance.value.size() == 49u);
  REQUIRE_FALSE(learn.outputs.model.value.empty());

  cv::Recognize rec;
  rec.inputs.model.value = learn.outputs.model.value;
  rec.inputs.feature.value = learn.outputs.mean.value;
  rec.inputs.threshold.value = 1.f;
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(0.f).margin(1e-2));
  CHECK(rec.outputs.match.value);
}

TEST_CASE("Learn saves an mxb that reads back as the same model", "[learn][mxb]")
{
  cv::Learn learn;
  for(auto s : {std::vector<float>{1.f, 2.f}, {3.f, 6.f}, {5.f, 4.f}, {0.f, 1.f}})
    learn_one(learn, s);
  REQUIRE(learn.outputs.valid.value);

  const std::string path = tmp_path("save");
  learn.inputs.file.value = path;
  pulse(learn, learn.inputs.save.value);

  cv_support::MxbModel r;
  REQUIRE(cv_support::read_mxb(path, r));
  CHECK(r.size == 2);
  CHECK(r.index == 4.0);
  // Full double precision survives the trip (the old "CVLM" format truncated
  // everything to float and stored no covariance at all).
  CHECK(r.mean.size() == 2u);
  CHECK(r.cov.size() == 4u);
  CHECK(r.inverse.size() == 4u);

  cv::Learn loaded;
  loaded.inputs.file.value = path;
  pulse(loaded, loaded.inputs.load.value);
  REQUIRE(loaded.outputs.valid.value);
  CHECK(loaded.outputs.mean.value == learn.outputs.mean.value);
  CHECK(loaded.outputs.invcov.value == learn.outputs.invcov.value);
  CHECK(loaded.outputs.covariance.value == learn.outputs.covariance.value);
  CHECK(loaded.outputs.model.value == learn.outputs.model.value);
  CHECK(loaded.outputs.index.value == learn.outputs.index.value);

  std::remove(path.c_str());
}

TEST_CASE("Learn still reads the legacy CVLM format", "[learn][persistence][legacy]")
{
  // Hand-write a file in the format this port used to produce.
  const std::string path = tmp_path("legacy");
  {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    const char magic[4] = {'C', 'V', 'L', 'M'};
    const std::uint32_t version = 1u, dim = 2u;
    const float mean[2] = {1.f, 2.f};
    const float invcov[4] = {1.f, 0.f, 0.f, 0.25f};
    os.write(magic, 4);
    os.write(reinterpret_cast<const char*>(&version), 4);
    os.write(reinterpret_cast<const char*>(&dim), 4);
    os.write(reinterpret_cast<const char*>(mean), sizeof(mean));
    os.write(reinterpret_cast<const char*>(invcov), sizeof(invcov));
  }

  cv::Learn l;
  l.inputs.file.value = path;
  pulse(l, l.inputs.load.value);
  REQUIRE(l.outputs.valid.value);
  CHECK(l.outputs.dimension.value == 2);
  CHECK(l.outputs.mean.value == std::vector<float>{1.f, 2.f});
  CHECK(l.outputs.invcov.value == std::vector<float>{1.f, 0.f, 0.f, 0.25f});
  // The legacy format carries neither a covariance nor an index, so neither is
  // reported (and training cannot resume from it).
  CHECK(l.outputs.covariance.value.empty());
  CHECK(l.outputs.index.value == 0.f);

  // Distance still works: (4,6) is (3,4) away, weighted 1 and 0.25 -> sqrt(9+4).
  cv::Recognize rec;
  rec.inputs.model.value = l.outputs.model.value;
  rec.inputs.feature.value = {4.f, 6.f};
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(std::sqrt(13.f)).epsilon(1e-5));

  std::remove(path.c_str());
}

// ========================================================= 3. resumable training
TEST_CASE("Training resumes from a loaded model's index", "[learn][resume][mxb]")
{
  cv_support::MxbModel file;
  REQUIRE(cv_support::read_mxb(fixture("The_letter_A.mxb"), file));
  REQUIRE(file.index == 83.0);

  cv::Learn learn;
  learn.inputs.file.value = fixture("The_letter_A.mxb");
  pulse(learn, learn.inputs.load.value);
  REQUIRE(learn.outputs.index.value == 83.f);

  const std::vector<float> mean0 = learn.outputs.mean.value;

  // Learn two more samples.
  const std::vector<float> s1{0.2f, 0.2f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const std::vector<float> s2{0.3f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f};
  learn_one(learn, s1);
  CHECK(learn.outputs.index.value == 84.f);
  CHECK(learn.outputs.samples.value == 84);
  learn_one(learn, s2);
  CHECK(learn.outputs.index.value == 85.f);
  CHECK(learn.outputs.samples.value == 85);

  // Hand-continue the 1/index blend from the file's own mean.
  std::vector<double> expected(file.mean.begin(), file.mean.end());
  {
    const double a1 = 1.0 / 84.0, b1 = 1.0 - a1;
    for(std::size_t i = 0; i < 7; ++i)
      expected[i] = static_cast<double>(s1[i]) * a1 + expected[i] * b1;
    const double a2 = 1.0 / 85.0, b2 = 1.0 - a2;
    for(std::size_t i = 0; i < 7; ++i)
      expected[i] = static_cast<double>(s2[i]) * a2 + expected[i] * b2;
  }
  REQUIRE(learn.outputs.mean.value.size() == 7u);
  for(std::size_t i = 0; i < 7; ++i)
    CHECK(learn.outputs.mean.value[i] == Approx(expected[i]).margin(1e-6));

  // If the index had restarted at 1, the mean after two samples would be
  // s1*(1/2)+... i.e. essentially s2/2 + s1/2 = (0.25, 0.15, 0...). With the
  // index continuing from 83 it barely moves away from the file's mean.
  CHECK(learn.outputs.mean.value[0] != Approx(0.25f).margin(1e-3));
  // It only moved by (0.2-0.1676)/84 + (0.3-0.1680)/85 ~= 0.0019.
  CHECK(learn.outputs.mean.value[0] == Approx(mean0[0]).margin(0.01));
  CHECK(learn.outputs.mean.value[0] != Approx(mean0[0]).margin(1e-9)); // it DID move

  // Saving now records the continued index.
  const std::string path = tmp_path("resume");
  learn.inputs.file.value = path;
  pulse(learn, learn.inputs.save.value);
  cv_support::MxbModel again;
  REQUIRE(cv_support::read_mxb(path, again));
  CHECK(again.index == 85.0);
  CHECK(again.size == 7);
  std::remove(path.c_str());
}

TEST_CASE("Reset clears a loaded model and its index", "[learn][resume]")
{
  cv::Learn learn;
  learn.inputs.file.value = fixture("The_letter_M.mxb");
  pulse(learn, learn.inputs.load.value);
  REQUIRE(learn.outputs.index.value == 35.f);

  pulse(learn, learn.inputs.reset.value);
  CHECK(learn.outputs.index.value == 0.f);
  CHECK(learn.outputs.samples.value == 0);
  CHECK(learn.outputs.dimension.value == 0);
  CHECK_FALSE(learn.outputs.valid.value);
  CHECK(learn.outputs.mean.value.empty());

  learn_one(learn, {1.f, 2.f});
  CHECK(learn.outputs.index.value == 1.f);
  CHECK(learn.outputs.dimension.value == 2);
}

// ============================================================ 4. Recognize lists
namespace
{
cv::blob_info make_blob(
    const std::array<float, 7>& nu, const std::array<float, 7>& hu, int id)
{
  cv::blob_info b{};
  b.id = id;
  b.nu20 = nu[0];
  b.nu02 = nu[1];
  b.nu11 = nu[2];
  b.nu21 = nu[3];
  b.nu12 = nu[4];
  b.nu30 = nu[5];
  b.nu03 = nu[6];
  b.hu = hu;
  return b;
}

// Identity-covariance model of dimension n centred on the origin: the
// Mahalanobis distance is then just the Euclidean norm, so expected values are
// trivially hand-checkable.
std::vector<float> identity_model(int n)
{
  std::vector<float> mean(static_cast<std::size_t>(n), 0.f);
  std::vector<float> inv(static_cast<std::size_t>(n) * n, 0.f);
  for(int i = 0; i < n; ++i)
    inv[static_cast<std::size_t>(i) * n + i] = 1.f;
  return cv_support::pack_model(n, mean, inv);
}
}

TEST_CASE("Recognize list mode: N vectors -> N distances, order preserved",
          "[recognize][list]")
{
  cv::Recognize rec;
  rec.inputs.model.value = identity_model(2);
  // Three 2-D candidates concatenated: (3,4) (0,0) (6,8) -> 5, 0, 10.
  rec.inputs.features.value = {3.f, 4.f, 0.f, 0.f, 6.f, 8.f};
  rec();

  REQUIRE(rec.outputs.distances.value.size() == 3u);
  CHECK(rec.outputs.distances.value[0] == Approx(5.f).margin(1e-5));
  CHECK(rec.outputs.distances.value[1] == Approx(0.f).margin(1e-5));
  CHECK(rec.outputs.distances.value[2] == Approx(10.f).margin(1e-5));

  // Order really is preserved: reverse the input, reverse the output.
  rec.inputs.features.value = {6.f, 8.f, 0.f, 0.f, 3.f, 4.f};
  rec();
  REQUIRE(rec.outputs.distances.value.size() == 3u);
  CHECK(rec.outputs.distances.value[0] == Approx(10.f).margin(1e-5));
  CHECK(rec.outputs.distances.value[2] == Approx(5.f).margin(1e-5));

  // The single-vector path is untouched by the list path.
  rec.inputs.feature.value = {1.f, 0.f};
  rec();
  CHECK(rec.outputs.valid.value);
  CHECK(rec.outputs.distance.value == Approx(1.f).margin(1e-5));

  // Empty list -> empty output.
  rec.inputs.features.value.clear();
  rec();
  CHECK(rec.outputs.distances.value.empty());
}

TEST_CASE("Recognize mode selects moments or Hu invariants", "[recognize][list][mode]")
{
  // Blob 0: nu = (3,4,0,0,0,0,0) -> |nu| = 5 ; hu = (0,...,6,8) -> |hu| = 10.
  // Blob 1: nu = (0,...,0,1)     -> |nu| = 1 ; hu = (2,0,...)   -> |hu| = 2.
  const auto b0 = make_blob({3.f, 4.f, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 6.f, 8.f}, 0);
  const auto b1 = make_blob({0, 0, 0, 0, 0, 0, 1.f}, {2.f, 0, 0, 0, 0, 0, 0}, 1);

  cv::Recognize rec;
  rec.inputs.model.value = identity_model(7);
  rec.inputs.blobs.value = {b0, b1};

  REQUIRE(rec.inputs.mode.value == cv::ReconMode::Moments); // cv.jit's default (0)
  rec();
  REQUIRE(rec.outputs.distances.value.size() == 2u);
  CHECK(rec.outputs.distances.value[0] == Approx(5.f).margin(1e-5));
  CHECK(rec.outputs.distances.value[1] == Approx(1.f).margin(1e-5));

  rec.inputs.mode.value = cv::ReconMode::Hu;
  rec();
  REQUIRE(rec.outputs.distances.value.size() == 2u);
  CHECK(rec.outputs.distances.value[0] == Approx(10.f).margin(1e-5));
  CHECK(rec.outputs.distances.value[1] == Approx(2.f).margin(1e-5));

  // The nu plane order must be cv.jit.blobs.moments': nu20, nu02, nu11, nu21,
  // nu12, nu30, nu03. Weight only the third plane and check it picks nu11.
  std::vector<float> mean(7, 0.f), inv(49, 0.f);
  inv[2 * 7 + 2] = 1.f;
  cv::blob_info probe{};
  probe.nu11 = 4.f;
  probe.nu20 = 100.f; // ignored by this model
  rec.inputs.mode.value = cv::ReconMode::Moments;
  rec.inputs.model.value = cv_support::pack_model(7, mean, inv);
  rec.inputs.blobs.value = {probe};
  rec();
  REQUIRE(rec.outputs.distances.value.size() == 1u);
  CHECK(rec.outputs.distances.value[0] == Approx(4.f).margin(1e-5));
}

TEST_CASE("Recognize list mode: dimension mismatch is a sentinel, not 0",
          "[recognize][list]")
{
  // A 7-element blob descriptor against a 2-D model: no distance is meaningful.
  {
    cv::Recognize rec;
    rec.inputs.model.value = identity_model(2);
    rec.inputs.blobs.value = {make_blob({1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 1, 1, 1, 1}, 0),
                              make_blob({0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, 1)};
    rec();
    REQUIRE(rec.outputs.distances.value.size() == 2u);
    for(float d : rec.outputs.distances.value)
    {
      CHECK(d == cv::Recognize::invalid_distance);
      CHECK(d > 1e6f);
    }
  }

  // A ragged flat list (5 floats for a 2-D model).
  {
    cv::Recognize rec;
    rec.inputs.model.value = identity_model(2);
    rec.inputs.features.value = {1.f, 2.f, 3.f, 4.f, 5.f};
    rec();
    REQUIRE_FALSE(rec.outputs.distances.value.empty());
    for(float d : rec.outputs.distances.value)
      CHECK(d == cv::Recognize::invalid_distance);
  }

  // No model at all.
  {
    cv::Recognize rec;
    rec.inputs.features.value = {1.f, 2.f, 3.f, 4.f};
    rec();
    REQUIRE_FALSE(rec.outputs.distances.value.empty());
    for(float d : rec.outputs.distances.value)
      CHECK(d == cv::Recognize::invalid_distance);
  }

  // A malformed packed model.
  {
    cv::Recognize rec;
    rec.inputs.model.value = {2.f, 0.f}; // claims dim 2 but truncated
    rec.inputs.features.value = {1.f, 2.f, 3.f, 4.f};
    rec.inputs.blobs.value = {make_blob({1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}, 0)};
    rec();
    CHECK_FALSE(rec.outputs.valid.value);
    CHECK(rec.outputs.distance.value == cv::Recognize::invalid_distance);
    REQUIRE_FALSE(rec.outputs.distances.value.empty());
    for(float d : rec.outputs.distances.value)
      CHECK(d == cv::Recognize::invalid_distance);
  }
}

TEST_CASE("Recognize list mode works off a real cv.jit fixture",
          "[recognize][list][fixture]")
{
  cv::Learn A, M;
  A.inputs.file.value = fixture("The_letter_A.mxb");
  M.inputs.file.value = fixture("The_letter_M.mxb");
  pulse(A, A.inputs.load.value);
  pulse(M, M.inputs.load.value);
  REQUIRE(A.outputs.valid.value);
  REQUIRE(M.outputs.valid.value);

  // Two candidates: A's own mean and M's own mean, as a flat 14-float list.
  std::vector<float> feats;
  feats.insert(feats.end(), A.outputs.mean.value.begin(), A.outputs.mean.value.end());
  feats.insert(feats.end(), M.outputs.mean.value.begin(), M.outputs.mean.value.end());

  cv::Recognize rec;
  rec.inputs.model.value = A.outputs.model.value;
  rec.inputs.features.value = feats;
  rec();
  REQUIRE(rec.outputs.distances.value.size() == 2u);
  // Candidate 0 is exactly A's mean -> ~0; candidate 1 is M's mean -> far away.
  CHECK(rec.outputs.distances.value[0] == Approx(0.f).margin(1e-2));
  CHECK(rec.outputs.distances.value[1] == Approx(10.7356f).epsilon(1e-3));
  CHECK(rec.outputs.distances.value[1] > rec.outputs.distances.value[0] + 5.f);

  // Same candidates against M's model: the verdict flips.
  rec.inputs.model.value = M.outputs.model.value;
  rec();
  REQUIRE(rec.outputs.distances.value.size() == 2u);
  CHECK(rec.outputs.distances.value[0] == Approx(17.3182f).epsilon(1e-3));
  CHECK(rec.outputs.distances.value[1] == Approx(0.f).margin(1e-2));
}
