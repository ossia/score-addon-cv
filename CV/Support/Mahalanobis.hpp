#pragma once

#include <Eigen/Dense>

#include <cmath>
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
  Eigen::MatrixXd invCov;   // dim x dim
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
}
