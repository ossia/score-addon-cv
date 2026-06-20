#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <vector>

namespace cv
{
// Accumulate feature vectors into a statistical model (cv.jit.learn).
// On "capture", the current input feature vector is added to the training set.
// On "train", the mean vector and covariance matrix of all captured samples are
// computed (Eigen), then the inverse-covariance via an SVD pseudo-inverse for
// robustness against singular covariance. The model (mean + flattened
// inverse-covariance, row-major) is emitted, plus the sample count and the
// feature dimension. Pure Eigen, no OpenCV.
//
// Persistence (like cv.jit.learn's read/write): a packed "Model" output vector
// [dim, mean, invcov] can be cabled straight into cv::Recognize's "Model" inlet
// and is saved with the score project. In addition, the "Save"/"Load" rising-edge
// toggles write/read that same packed model to/from the "File" path using plain
// std::ofstream/ifstream (self-contained binary format). On load, the same outputs
// are populated as a fresh train (mean, invcov, dimension, model, valid).
//
// MEMORY NOTE: captured samples are stored unbounded in m_samples (one push per
// rising "Capture" edge). For long-running training this grows without limit; the
// stored count is capped at max_samples below to bound memory. Recognition only
// needs the trained mean + inverse covariance, not the raw samples.
struct Learn
{
  halp_meta(name, "Learn");
  halp_meta(c_name, "cv_learn");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Accumulate feature vectors into a statistical model "
                         "(mean + inverse covariance).");
  halp_meta(uuid, "8c6a4dd8-514e-447d-a056-ad96ba5576bd");

  struct
  {
    halp::val_port<"Feature", std::vector<float>> feature;
    halp::toggle<"Capture"> capture;
    halp::toggle<"Train"> train;
    halp::toggle<"Reset"> reset;
    halp::lineedit<"File", ""> file;
    halp::toggle<"Save"> save;
    halp::toggle<"Load"> load;
  } inputs;

  struct
  {
    halp::val_port<"Mean", std::vector<float>> mean;
    halp::val_port<"Inverse covariance", std::vector<float>> invcov;
    // Packed [dim, mean, invcov] model, cable straight into cv::Recognize "Model".
    halp::val_port<"Model", std::vector<float>> model;
    halp::val_port<"Samples", int> samples;
    halp::val_port<"Dimension", int> dimension;
    halp::val_port<"Valid", bool> valid;
  } outputs;

  // Bound the stored sample buffer so unbounded "Capture" pulses cannot grow
  // memory without limit. Oldest samples are dropped once this is exceeded.
  static constexpr std::size_t max_samples = 100000;

  void operator()() noexcept;

private:
  void retrain() noexcept;
  void publish_model() noexcept;
  bool save_model() const noexcept;
  bool load_model() noexcept;

  std::vector<std::vector<float>> m_samples;
  bool m_prevCapture = false;
  bool m_prevTrain = false;
  bool m_prevReset = false;
  bool m_prevSave = false;
  bool m_prevLoad = false;
};
}
