#pragma once

#include <CV/Support/Mahalanobis.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace cv
{
// How the statistical model is estimated from the incoming feature vectors.
//
// CvJitOnline (default, first enumerator so a default-constructed object behaves
//   like cv.jit.learn): a true online / incremental estimator. Every "Capture"
//   pulse immediately folds one sample into the running mean + covariance and
//   recomputes the SVD pseudo-inverse, exactly as cv.jit.learn does on a list in
//   its right inlet. There is NO training step -- cv.jit has no such message and
//   the model is queryable from the very first sample. No raw sample is stored,
//   so memory is O(dim^2) regardless of how long training runs.
//   See cv_support::OnlineModel for the exact (deliberately non-textbook) update.
//
// Batch: the estimator this port used before cv.jit fidelity was restored. Raw
//   samples are accumulated on "Capture" and the unbiased (1/(N-1)) covariance
//   is computed on a "Train" rising edge, requiring at least dim + 1 samples.
//   Kept because it is genuinely better conditioned when all samples are
//   available up front: it never passes through the rank-deficient states the
//   online update goes through, and it weights every sample against the final
//   mean rather than against the running one.
enum class LearnMode
{
  CvJitOnline,
  Batch
};

// Accumulate feature vectors into a statistical model (cv.jit.learn).
//
// Inputs / behaviour:
//  - "Capture" (rising edge) learns the current feature vector. In CvJitOnline
//    mode the model is updated and published on the spot; in Batch mode the
//    sample is only stored.
//  - "Train" (rising edge) recomputes the batch model. It is a no-op re-publish
//    in CvJitOnline mode (kept so existing patches that pulse it keep working).
//  - "Reset" clears the model and the sample buffer (cv.jit's clear / reset).
//  - "Save" / "Load" (rising edges) write / read the "File" path.
//
// Persistence. "Save" writes cv.jit's own `.mxb` format (see cv_support::MxbModel:
// 'cvjt' magic, int32 size, double index, double mean/covariance/inverse), so a
// model trained here opens in cv.jit.learn and cv.jit.blobs.recon and vice versa.
// "Load" accepts:
//   - `.mxb`, either endianness (bytes "cvjt" = big-endian payload as in the
//     bundled cv.jit/misc/The_letter_*.mxb, bytes "tjvc" = little-endian),
//   - the legacy "CVLM" float format this port used to write, so previously
//     saved models keep loading. That format has no covariance and no index, so
//     a CVLM model can be *recognised with* but not *resumed from*.
// Because the sample index is part of the .mxb file, loading an .mxb and then
// pulsing "Capture" continues training where the file left off (index 83 -> 84,
// 85, ...) instead of restarting at 1 -- the advertised cv.jit behaviour.
//
// The packed "Model" output ([dim, mean, invcov] floats) can still be cabled
// straight into cv::Recognize's "Model" inlet and is saved with the project.
//
// MEMORY NOTE: only Batch mode stores raw samples (bounded by max_samples,
// oldest dropped). CvJitOnline stores nothing per sample.
struct Learn
{
  halp_meta(name, "Learn");
  halp_meta(c_name, "cv_learn");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Accumulate feature vectors into a statistical model "
                         "(mean + inverse covariance), cv.jit.learn-compatible.");
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
    // Appended after the pre-existing ports so port indices of saved projects
    // are preserved.
    halp::enum_t<LearnMode, "Mode"> mode;
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
    // Appended: the raw covariance (row-major, dim*dim) and cv.jit's sample
    // index, both of which are part of the .mxb file and were previously not
    // observable at all.
    halp::val_port<"Covariance", std::vector<float>> covariance;
    halp::val_port<"Index", float> index;
  } outputs;

  // Bound the stored sample buffer (Batch mode only) so unbounded "Capture"
  // pulses cannot grow memory without limit. Oldest samples are dropped.
  static constexpr std::size_t max_samples = 100000;

  void operator()() noexcept;

  // Direct access for tests / for objects embedding a Learn: the live model.
  const cv_support::OnlineModel& model_state() const noexcept { return m_model; }

private:
  void retrain() noexcept;   // Batch: recompute from m_samples
  void publish() noexcept;   // push m_model to the output ports
  bool save_model() noexcept;
  bool load_model() noexcept;

  cv_support::OnlineModel m_model;
  std::vector<std::vector<float>> m_samples;
  // Set when the loaded model came from the legacy CVLM format, which carries
  // no covariance: the model can score, but the covariance output is empty and
  // online training cannot meaningfully resume from it.
  bool m_covarianceKnown = true;
  bool m_prevCapture = false;
  bool m_prevTrain = false;
  bool m_prevReset = false;
  bool m_prevSave = false;
  bool m_prevLoad = false;
};
}
