#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <vector>

namespace cv
{
// Classify an incoming feature vector against a statistical model
// (cv.jit.blobs.recon). The model (mean + inverse-covariance, row-major flat)
// is fed in as feature inlets, typically produced by cv::Learn. The Mahalanobis
// distance d = sqrt((x - mean)^T * invCov * (x - mean)) is emitted, along with a
// "match" boolean that is true when d is below the threshold control.
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
  } inputs;

  struct
  {
    halp::val_port<"Distance", float> distance;
    halp::val_port<"Match", bool> match;
    halp::val_port<"Valid", bool> valid;
  } outputs;

  // Sentinel distance reported when the model / feature are dimensionally
  // inconsistent. A large finite value (rather than 0, which used to read as a
  // *perfect* match) so anything thresholding on distance rejects it.
  static constexpr float invalid_distance = 1e30f;

  void operator()() noexcept;
};
}
