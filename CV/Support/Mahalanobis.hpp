#pragma once

#include <Eigen/Dense>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cv_support
{
// Shared statistical-model helpers for the cv.jit.learn / cv.jit.blobs.recon family.
// Pure Eigen, no OpenCV.

// SVD-based pseudo-inverse, robust against singular / ill-conditioned matrices.
// Singular values below tol * sigma_max are treated as zero.
inline Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& m, double tol = 1e-9)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& sv = svd.singularValues();
  const double smax = sv.size() > 0 ? sv(0) : 0.0;
  const double threshold = tol * smax;

  Eigen::VectorXd invSv(sv.size());
  for(Eigen::Index i = 0; i < sv.size(); ++i)
    invSv(i) = (sv(i) > threshold) ? (1.0 / sv(i)) : 0.0;

  return svd.matrixV() * invSv.asDiagonal() * svd.matrixU().transpose();
}

// Statistical model accumulated from a set of feature vectors.
struct StatModel
{
  bool valid = false;
  int dim = 0;
  int count = 0;
  Eigen::VectorXd mean;     // dim
  Eigen::MatrixXd cov;      // dim x dim, unbiased sample covariance
  Eigen::MatrixXd invCov;   // dim x dim, SVD pseudo-inverse of cov
};

// Compute mean + covariance + inverse-covariance (SVD pseudo-inverse) for a set
// of samples. Returns valid=false if there are too few samples (< dim + 1) or
// the dimensions are inconsistent / zero.
inline StatModel
compute_model(const std::vector<std::vector<float>>& samples)
{
  StatModel model;
  const std::size_t n = samples.size();
  if(n == 0)
    return model;

  const std::size_t dim = samples[0].size();
  if(dim == 0)
    return model;

  // All samples must share the dimension.
  for(const auto& s : samples)
    if(s.size() != dim)
      return model;

  // Need at least dim + 1 samples for a non-degenerate covariance.
  if(n < dim + 1)
    return model;

  const Eigen::Index D = static_cast<Eigen::Index>(dim);
  const Eigen::Index N = static_cast<Eigen::Index>(n);

  // Pack samples into an N x D matrix.
  Eigen::MatrixXd X(N, D);
  for(Eigen::Index i = 0; i < N; ++i)
    for(Eigen::Index j = 0; j < D; ++j)
      X(i, j) = static_cast<double>(samples[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(j)]);

  Eigen::VectorXd mean = X.colwise().mean();

  Eigen::MatrixXd centered = X.rowwise() - mean.transpose();
  // Unbiased sample covariance (1 / (N - 1)).
  Eigen::MatrixXd cov
      = (centered.transpose() * centered) / static_cast<double>(N - 1);

  // Guard against zero variance (all samples identical along every axis).
  if(cov.diagonal().maxCoeff() <= 0.0)
    return model;

  model.invCov = pseudo_inverse(cov);
  model.cov = cov;
  model.mean = mean;
  model.dim = static_cast<int>(dim);
  model.count = static_cast<int>(n);
  model.valid = true;
  return model;
}

// Serialized model layout shared by cv::Learn (output) and cv::Recognize (input)
// and the on-disk persistence file: a flat float vector
//   [ dim, mean[0..dim-1], invCov[0..dim*dim-1] (row-major) ].
// This lets a trained model round-trip through a single score cable / saved
// project, or a file written by Learn's save toggle.

// Pack a mean + flat (row-major) inverse covariance into the model layout.
inline std::vector<float>
pack_model(int dim, const std::vector<float>& mean, const std::vector<float>& invCovFlat)
{
  std::vector<float> out;
  if(dim <= 0 || mean.size() != static_cast<std::size_t>(dim)
     || invCovFlat.size() != static_cast<std::size_t>(dim) * dim)
    return out; // empty == invalid

  out.reserve(1 + mean.size() + invCovFlat.size());
  out.push_back(static_cast<float>(dim));
  out.insert(out.end(), mean.begin(), mean.end());
  out.insert(out.end(), invCovFlat.begin(), invCovFlat.end());
  return out;
}

// Unpack a model vector into dim / mean / invCov. Returns false if malformed.
inline bool unpack_model(
    const std::vector<float>& model, int& dim, std::vector<float>& mean,
    std::vector<float>& invCovFlat)
{
  dim = 0;
  mean.clear();
  invCovFlat.clear();
  if(model.empty())
    return false;

  const double d = model[0];
  if(d < 1.0 || d != std::floor(d))
    return false;
  const int D = static_cast<int>(d);

  const std::size_t expected
      = 1 + static_cast<std::size_t>(D) + static_cast<std::size_t>(D) * D;
  if(model.size() != expected)
    return false;

  dim = D;
  mean.assign(model.begin() + 1, model.begin() + 1 + D);
  invCovFlat.assign(model.begin() + 1 + D, model.end());
  return true;
}

// Mahalanobis distance d = sqrt((x - mean)^T * invCov * (x - mean)).
// mean / invCov supplied flat (invCov row-major, dim*dim). Returns < 0 on
// dimension mismatch / invalid input.
inline double mahalanobis(
    const std::vector<float>& x, const std::vector<float>& mean,
    const std::vector<float>& invCovFlat)
{
  const std::size_t dim = mean.size();
  if(dim == 0 || x.size() != dim || invCovFlat.size() != dim * dim)
    return -1.0;

  const Eigen::Index D = static_cast<Eigen::Index>(dim);

  Eigen::VectorXd diff(D);
  for(Eigen::Index i = 0; i < D; ++i)
    diff(i) = static_cast<double>(x[static_cast<std::size_t>(i)])
              - static_cast<double>(mean[static_cast<std::size_t>(i)]);

  Eigen::MatrixXd invCov(D, D);
  for(Eigen::Index i = 0; i < D; ++i)
    for(Eigen::Index j = 0; j < D; ++j)
      invCov(i, j) = static_cast<double>(
          invCovFlat[static_cast<std::size_t>(i) * dim
                     + static_cast<std::size_t>(j)]);

  const double d2 = diff.transpose() * invCov * diff;
  return std::sqrt(std::max(0.0, d2));
}

// ---------------------------------------------------------------------------
// cv.jit.learn's ONLINE (incremental) estimator.
//
// Reproduced verbatim from cv.jit/source/projects/cv.jit.learn/cv.jit.learn.cpp
// (cv_jit_learn_list, "learn" inlet):
//
//     index += 1;  a = 1/index;  b = 1 - a;
//     for(i)      mean[i]      = data[i] * a + mean[i] * b;
//     for(i,j)    cov[i*n + j] = ((data[i]-mean[i]) * (data[j]-mean[j])) * a
//                                + cov[i*n + j] * b;
//     inverse = pseudo-inverse(cov)                      // SVD, EVERY sample
//
// Two things a textbook implementation gets wrong, and which must not be
// "fixed" here:
//
//  1. The covariance loop uses the mean that was *just* updated with the very
//     same sample, not the previous mean. At step 2 with samples (1,2) then
//     (3,6) the new mean is (2,4), so the deviation used is (1,2), not (2,4);
//     cov(0,0) is 0.5, not 2.
//  2. The weighting is a running 1/index blend, not 1/(N-1) (nor 1/N over the
//     raw deviations from the final mean). After the three samples
//     (1,2), (3,6), (5,4) this gives cov = [[5/3, 2/3], [2/3, 4/3]], whereas
//     the unbiased sample covariance of those same three points is
//     [[4, 2], [2, 4]] and the population covariance [[8/3, 4/3], [4/3, 8/3]].
//
// Consequences worth knowing: after ONE sample cov is exactly zero (and so is
// its pseudo-inverse, so every distance reads 0); after k <= n samples cov has
// rank <= k - 1 and the SVD pseudo-inverse silently drops the null space. That
// is cv.jit's behaviour -- the model is queryable from the first sample and
// simply becomes more discriminating as samples accumulate. It never "refuses".
struct OnlineModel
{
  double index = 0.0;      // number of learned samples (cv.jit calls it index)
  Eigen::VectorXd mean;    // n
  Eigen::MatrixXd cov;     // n x n, cv.jit's running covariance
  Eigen::MatrixXd inverse; // n x n, pseudo-inverse of cov

  int size() const noexcept { return static_cast<int>(mean.size()); }

  // A model exists as soon as one sample has been learned.
  bool has_model() const noexcept { return index > 0.0 && mean.size() > 0; }

  // ... but it can only *discriminate* once the covariance is non-degenerate.
  bool discriminative() const noexcept
  {
    return has_model() && cov.size() > 0 && cov.diagonal().maxCoeff() > 0.0;
  }

  void allocate(int n)
  {
    mean = Eigen::VectorXd::Zero(n);
    cov = Eigen::MatrixXd::Zero(n, n);
    inverse = Eigen::MatrixXd::Zero(n, n);
  }

  void clear() noexcept
  {
    index = 0.0;
    mean.setZero();
    cov.setZero();
    inverse.setZero();
  }

  void reset()
  {
    index = 0.0;
    mean.resize(0);
    cov.resize(0, 0);
    inverse.resize(0, 0);
  }

  // Learn one sample. cv.jit has a fixed size fixed at instantiation and errors
  // out on a mismatched list; this port has no declared size, so the first
  // sample sets it and a later change of length starts a fresh model rather
  // than mixing incompatible statistics.
  bool update(const std::vector<float>& data)
  {
    const int n = static_cast<int>(data.size());
    if(n <= 0)
      return false;
    if(size() != n)
    {
      // Length change: start a fresh model rather than blending statistics of
      // different dimensions.
      allocate(n);
      index = 0.0;
    }

    index += 1.0;
    const double a = 1.0 / index;
    const double b = 1.0 - a;

    // Mean FIRST -- the covariance below deliberately uses the NEW mean.
    for(int i = 0; i < n; ++i)
      mean(i) = static_cast<double>(data[static_cast<std::size_t>(i)]) * a + mean(i) * b;

    for(int i = 0; i < n; ++i)
    {
      const double di = static_cast<double>(data[static_cast<std::size_t>(i)]) - mean(i);
      for(int j = 0; j < n; ++j)
      {
        const double dj
            = static_cast<double>(data[static_cast<std::size_t>(j)]) - mean(j);
        cov(i, j) = (di * dj) * a + cov(i, j) * b;
      }
    }

    // cv.jit recomputes the SVD pseudo-inverse on every single sample.
    inverse = pseudo_inverse(cov);
    return true;
  }

  // Mahalanobis distance of x to this model. Returns < 0 on dimension mismatch.
  double distance(const std::vector<float>& x) const
  {
    const int n = size();
    if(n <= 0 || static_cast<int>(x.size()) != n)
      return -1.0;
    Eigen::VectorXd diff(n);
    for(int i = 0; i < n; ++i)
      diff(i) = static_cast<double>(x[static_cast<std::size_t>(i)]) - mean(i);
    const double d2 = diff.transpose() * inverse * diff;
    return std::sqrt(std::max(0.0, d2));
  }
};

// Flatten an Eigen matrix row-major into a float vector (score port layout).
inline std::vector<float> flatten_rowmajor(const Eigen::MatrixXd& m)
{
  std::vector<float> out(static_cast<std::size_t>(m.rows() * m.cols()));
  for(Eigen::Index i = 0; i < m.rows(); ++i)
    for(Eigen::Index j = 0; j < m.cols(); ++j)
      out[static_cast<std::size_t>(i * m.cols() + j)] = static_cast<float>(m(i, j));
  return out;
}

inline std::vector<float> flatten(const Eigen::VectorXd& v)
{
  std::vector<float> out(static_cast<std::size_t>(v.size()));
  for(Eigen::Index i = 0; i < v.size(); ++i)
    out[static_cast<std::size_t>(i)] = static_cast<float>(v(i));
  return out;
}

// ---------------------------------------------------------------------------
// cv.jit's `.mxb` model file (cv_jit_learn_read / cv_jit_learn_write, and the
// identical reader in cv.jit.blobs.recon).
//
// Layout, with NO padding anywhere:
//     int32   magic      'cvjt'  (the Max four-char-code 0x63766A74)
//     int32   size       n, the feature dimension (7 for the moments/Hu family)
//     double  index      number of samples learned so far
//     double  mean[n]
//     double  covariance[n*n]      (row-major)
//     double  inverse[n*n]         (row-major, pseudo-inverse of covariance)
//   => total 8 + 8 + 8*n + 16*n*n bytes; 856 for n == 7.
//
// ENDIANNESS. cv.jit writes the four-char-code as a *native* int32 and, on
// reading, compares the natively-read int32 against 'cvjt' and against 'tjvc';
// the latter means "written on the other endianness, swap everything". So in
// terms of the bytes actually on disk:
//     bytes "cvjt" (63 76 6A 74)  =>  payload is BIG endian    (PPC-era files,
//                                     e.g. the bundled The_letter_*.mxb)
//     bytes "tjvc" (74 6A 76 63)  =>  payload is LITTLE endian (any x86/arm
//                                     machine, i.e. every modern Max)
// Anything else is not a cv.jit model and is rejected.
struct MxbModel
{
  int size = 0;
  double index = 0.0;
  std::vector<double> mean;     // size
  std::vector<double> cov;      // size*size, row-major
  std::vector<double> inverse;  // size*size, row-major
};

enum class mxb_endian
{
  native, // what cv.jit itself would write on this machine
  little,
  big
};

namespace detail
{
inline void swap_bytes(void* p, std::size_t n) noexcept
{
  auto* b = static_cast<unsigned char*>(p);
  for(std::size_t i = 0; i < n / 2; ++i)
  {
    const unsigned char t = b[i];
    b[i] = b[n - 1 - i];
    b[n - 1 - i] = t;
  }
}

inline bool host_is_big() noexcept
{
  return std::endian::native == std::endian::big;
}

template <typename T>
inline T read_pod(const unsigned char* src, bool swap) noexcept
{
  T v{};
  std::memcpy(&v, src, sizeof(T));
  if(swap)
    swap_bytes(&v, sizeof(T));
  return v;
}

template <typename T>
inline void append_pod(std::vector<unsigned char>& dst, T v, bool swap)
{
  unsigned char tmp[sizeof(T)];
  std::memcpy(tmp, &v, sizeof(T));
  if(swap)
    swap_bytes(tmp, sizeof(T));
  dst.insert(dst.end(), tmp, tmp + sizeof(T));
}
}

// Largest dimension we are willing to allocate from an untrusted file
// (a 4096-wide model would already be 268 MB of doubles).
inline constexpr int mxb_max_size = 4096;

// Exact serialized size of an n-dimensional .mxb model.
inline std::size_t mxb_byte_size(int n) noexcept
{
  const std::size_t N = static_cast<std::size_t>(n);
  return 4 + 4 + 8 + 8 * N + 8 * N * N + 8 * N * N;
}

// Read a cv.jit .mxb model. Returns false (leaving `out` untouched) on a wrong
// magic, a truncated file, or an implausible size -- never throws, never
// over-reads.
inline bool read_mxb(const std::string& path, MxbModel& out)
{
  std::ifstream is(path, std::ios::binary);
  if(!is)
    return false;

  std::vector<unsigned char> buf(
      (std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
  if(buf.size() < 8)
    return false;

  // Detect the payload endianness from the literal bytes of the four-char code.
  const bool payload_big
      = (buf[0] == 'c' && buf[1] == 'v' && buf[2] == 'j' && buf[3] == 't');
  const bool payload_little
      = (buf[0] == 't' && buf[1] == 'j' && buf[2] == 'v' && buf[3] == 'c');
  if(!payload_big && !payload_little)
    return false;

  const bool swap = (payload_big != detail::host_is_big());

  const std::int32_t n = detail::read_pod<std::int32_t>(buf.data() + 4, swap);
  if(n <= 0 || n > mxb_max_size)
    return false;
  if(buf.size() < mxb_byte_size(n))
    return false;

  MxbModel m;
  m.size = static_cast<int>(n);
  const std::size_t N = static_cast<std::size_t>(n);
  std::size_t off = 8;

  m.index = detail::read_pod<double>(buf.data() + off, swap);
  off += 8;

  auto read_block = [&](std::vector<double>& dst, std::size_t count) {
    dst.resize(count);
    for(std::size_t i = 0; i < count; ++i, off += 8)
      dst[i] = detail::read_pod<double>(buf.data() + off, swap);
  };
  read_block(m.mean, N);
  read_block(m.cov, N * N);
  read_block(m.inverse, N * N);

  out = std::move(m);
  return true;
}

// Write a cv.jit .mxb model. `endian` defaults to what cv.jit itself would
// produce on this machine (i.e. bytes "tjvc" + little-endian payload on x86 /
// arm64), which is what modern Max reads back without swapping.
inline bool write_mxb(
    const std::string& path, const MxbModel& m, mxb_endian endian = mxb_endian::native)
{
  const std::size_t N = static_cast<std::size_t>(m.size);
  if(m.size <= 0 || m.size > mxb_max_size)
    return false;
  if(m.mean.size() != N || m.cov.size() != N * N || m.inverse.size() != N * N)
    return false;

  const bool want_big
      = (endian == mxb_endian::big)
        || (endian == mxb_endian::native && detail::host_is_big());
  const bool swap = (want_big != detail::host_is_big());

  std::vector<unsigned char> buf;
  buf.reserve(mxb_byte_size(m.size));
  // The magic is written as the four *characters* in payload order, which is
  // exactly what a native int32 write of 'cvjt' produces on that endianness.
  if(want_big)
  {
    const unsigned char magic[4] = {'c', 'v', 'j', 't'};
    buf.insert(buf.end(), magic, magic + 4);
  }
  else
  {
    const unsigned char magic[4] = {'t', 'j', 'v', 'c'};
    buf.insert(buf.end(), magic, magic + 4);
  }

  detail::append_pod<std::int32_t>(buf, static_cast<std::int32_t>(m.size), swap);
  detail::append_pod<double>(buf, m.index, swap);
  for(double v : m.mean)
    detail::append_pod<double>(buf, v, swap);
  for(double v : m.cov)
    detail::append_pod<double>(buf, v, swap);
  for(double v : m.inverse)
    detail::append_pod<double>(buf, v, swap);

  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if(!os)
    return false;
  os.write(
      reinterpret_cast<const char*>(buf.data()),
      static_cast<std::streamsize>(buf.size()));
  return static_cast<bool>(os);
}

// Convert to / from the Eigen-backed online model.
inline MxbModel to_mxb(const OnlineModel& m)
{
  MxbModel out;
  out.size = m.size();
  out.index = m.index;
  if(out.size <= 0)
    return out;
  const std::size_t N = static_cast<std::size_t>(out.size);
  out.mean.resize(N);
  for(std::size_t i = 0; i < N; ++i)
    out.mean[i] = m.mean(static_cast<Eigen::Index>(i));
  out.cov.resize(N * N);
  out.inverse.resize(N * N);
  for(std::size_t i = 0; i < N; ++i)
    for(std::size_t j = 0; j < N; ++j)
    {
      const auto I = static_cast<Eigen::Index>(i);
      const auto J = static_cast<Eigen::Index>(j);
      out.cov[i * N + j] = m.cov(I, J);
      out.inverse[i * N + j] = m.inverse(I, J);
    }
  return out;
}

inline bool from_mxb(const MxbModel& in, OnlineModel& out)
{
  const std::size_t N = static_cast<std::size_t>(in.size);
  if(in.size <= 0 || in.mean.size() != N || in.cov.size() != N * N
     || in.inverse.size() != N * N)
    return false;

  out.allocate(in.size);
  out.index = in.index;
  for(std::size_t i = 0; i < N; ++i)
    out.mean(static_cast<Eigen::Index>(i)) = in.mean[i];
  for(std::size_t i = 0; i < N; ++i)
    for(std::size_t j = 0; j < N; ++j)
    {
      const auto I = static_cast<Eigen::Index>(i);
      const auto J = static_cast<Eigen::Index>(j);
      out.cov(I, J) = in.cov[i * N + j];
      out.inverse(I, J) = in.inverse[i * N + j];
    }
  return true;
}
}
