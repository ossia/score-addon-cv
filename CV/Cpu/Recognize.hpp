#pragma once

#include <CV/Cpu/BlobStats.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <vector>

namespace cv
{
// cv.jit.blobs.recon's `mode` attribute: which 7 numbers of a blob's moment set
// are fed to the classifier. cv.jit.blobs.moments emits a 17-plane vector per
// blob whose planes 0..6 are the normalised central moments nu20, nu02, nu11,
// nu21, nu12, nu30, nu03 and whose planes 7..13 are Hu's seven invariants;
// recon takes either the first or the second group. Moments is mode 0 and is
// cv.jit's default, hence the first enumerator here.
enum class ReconMode
{
  Moments,
  Hu
};

// Classify feature vectors against a statistical model (cv.jit.blobs.recon).
//
// The model (mean + inverse-covariance, row-major flat) is fed in as feature
// inlets, typically produced by cv::Learn -- either as the separate Mean /
// Inverse covariance lists or as its single packed "Model" output.
//
// Two ways to score, both active in the same tick:
//  - the single "Feature" vector -> "Distance" / "Match" / "Valid" (unchanged);
//  - a whole frame's worth of candidates -> "Distances", one per candidate, in
//    input order. cv.jit.blobs.recon is a matrix operator: it consumes an entire
//    blob matrix and emits one Mahalanobis distance per blob per frame, which a
//    single-vector object cannot express. Candidates come either from
//    "Blobs" (chains directly from cv::BlobStats, "Mode" picks nu* or Hu) or
//    from "Features", a flat concatenation of N vectors of `dim` floats each.
//
// The Mahalanobis distance is d = sqrt((x - mean)^T * invCov * (x - mean)).
// Pure Eigen, no OpenCV.
struct Recognize
{
  halp_meta(name, "Recognize");
  halp_meta(c_name, "cv_recognize");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Mahalanobis distance of a feature vector to a learned "
                         "statistical model.");
  halp_meta(uuid, "45757fd0-930e-4ad6-8eb7-a7a692ac1474");

  struct
  {
    halp::val_port<"Feature", std::vector<float>> feature;
    halp::val_port<"Mean", std::vector<float>> mean;
    halp::val_port<"Inverse covariance", std::vector<float>> invcov;
    // Optional packed model (see cv::Learn "Model" output): when non-empty it
    // overrides the Mean / Inverse covariance inlets, so a model trained by
    // cv::Learn can be round-tripped through a single score cable / project save.
    // Layout: [dim, mean[dim], invcov[dim*dim]] as float.
    halp::val_port<"Model", std::vector<float>> model;
    halp::hslider_f32<"Threshold", halp::range{0.f, 100.f, 3.f}> threshold;
    // Appended after the pre-existing ports so port indices of saved projects
    // are preserved.
    // Per-blob list mode: chains straight from cv::BlobStats "Blobs".
    struct
    {
      halp_meta(name, "Blobs");
      std::vector<blob_info> value;
    } blobs;
    // Generic list mode: N feature vectors of `dim` floats, concatenated.
    halp::val_port<"Features", std::vector<float>> features;
    halp::enum_t<ReconMode, "Mode"> mode;
  } inputs;

  struct
  {
    halp::val_port<"Distance", float> distance;
    halp::val_port<"Match", bool> match;
    halp::val_port<"Valid", bool> valid;
    // Appended: one distance per candidate of the list inputs, order preserved.
    struct
    {
      halp_meta(name, "Distances");
      std::vector<float> value;
    } distances;
  } outputs;

  // Sentinel distance reported when the model / feature are dimensionally
  // inconsistent. A large finite value (rather than 0, which used to read as a
  // *perfect* match) so anything thresholding on distance rejects it.
  static constexpr float invalid_distance = 1e30f;

  // Number of elements a blob contributes in either mode (nu20..nu03 / hu0..hu6).
  static constexpr int blob_feature_size = 7;

  void operator()() noexcept;
};
}
