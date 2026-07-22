#include "Learn.hpp"

#include <CV/Support/Mahalanobis.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>

namespace cv
{
void Learn::operator()() noexcept
{
  // Reset: drop everything (cv.jit's clear / reset messages).
  if(inputs.reset.value && !m_prevReset)
  {
    m_samples.clear();
    m_model.reset();
    m_covarianceKnown = true;
    publish();
  }
  m_prevReset = inputs.reset.value;

  // Capture (rising edge): learn the current feature vector.
  if(inputs.capture.value && !m_prevCapture)
  {
    const auto& f = inputs.feature.value;
    if(!f.empty())
    {
      if(inputs.mode.value == LearnMode::CvJitOnline)
      {
        // cv.jit.learn: one incremental update per sample, model immediately
        // usable -- there is no separate training step upstream.
        if(!m_covarianceKnown)
        {
          // A legacy CVLM model carries no covariance; resuming from it would
          // blend a real deviation into a fabricated zero matrix. Start over.
          m_model.reset();
          m_covarianceKnown = true;
        }
        m_model.update(f);
        publish();
      }
      else
      {
        m_samples.push_back(f);
        // Bound memory: drop the oldest sample once over the cap.
        if(m_samples.size() > max_samples)
          m_samples.erase(m_samples.begin());
      }
    }
  }
  m_prevCapture = inputs.capture.value;

  // Train (rising edge): recompute the batch model. cv.jit has no such message;
  // in online mode this is only a re-publish so existing patches keep working.
  if(inputs.train.value && !m_prevTrain)
  {
    if(inputs.mode.value == LearnMode::Batch)
      retrain();
    else
      publish();
  }
  m_prevTrain = inputs.train.value;

  // Save / Load (rising edge): persist or restore the model to/from the file.
  if(inputs.save.value && !m_prevSave)
    save_model();
  m_prevSave = inputs.save.value;

  if(inputs.load.value && !m_prevLoad)
    load_model();
  m_prevLoad = inputs.load.value;
}

void Learn::retrain() noexcept
{
  const auto model = cv_support::compute_model(m_samples);
  m_covarianceKnown = true;

  if(!model.valid)
  {
    // Not enough / inconsistent samples (< dim + 1, zero variance, ...): report
    // a clearly-empty, invalid model rather than a half-trained one.
    m_model.reset();
    m_model.index = static_cast<double>(m_samples.size());
    publish();
    return;
  }

  m_model.allocate(model.dim);
  m_model.mean = model.mean;
  m_model.cov = model.cov;
  m_model.inverse = model.invCov;
  m_model.index = static_cast<double>(m_samples.size());
  publish();
}

void Learn::publish() noexcept
{
  outputs.index = static_cast<float>(m_model.index);
  outputs.samples = static_cast<int>(std::llround(m_model.index));

  const int dim = m_model.size();
  if(dim <= 0)
  {
    outputs.dimension = 0;
    outputs.mean.value.clear();
    outputs.invcov.value.clear();
    outputs.covariance.value.clear();
    outputs.model.value.clear();
    outputs.valid = false;
    return;
  }

  outputs.dimension = dim;
  outputs.mean.value = cv_support::flatten(m_model.mean);
  outputs.invcov.value = cv_support::flatten_rowmajor(m_model.inverse);
  if(m_covarianceKnown)
    outputs.covariance.value = cv_support::flatten_rowmajor(m_model.cov);
  else
    outputs.covariance.value.clear();

  outputs.model.value
      = cv_support::pack_model(dim, outputs.mean.value, outputs.invcov.value);

  // "valid" means the model can actually discriminate, not merely that samples
  // were seen. After a single online sample the covariance -- and therefore its
  // pseudo-inverse -- is exactly zero, so every distance would read 0, i.e. a
  // perfect match for anything. The mean / invcov / model outputs ARE published
  // in that state (cv.jit lets you query from sample 1); only the flag is off.
  bool nonDegenerate = false;
  if(m_covarianceKnown)
    nonDegenerate = m_model.cov.size() > 0 && m_model.cov.diagonal().maxCoeff() > 0.0;
  else
    nonDegenerate
        = m_model.inverse.size() > 0 && m_model.inverse.cwiseAbs().maxCoeff() > 0.0;
  outputs.valid = nonDegenerate;
}

// Persistence: writes cv.jit's own .mxb (see cv_support::MxbModel).
bool Learn::save_model() noexcept
{
  if(inputs.file.value.empty() || m_model.size() <= 0)
    return false;

  return cv_support::write_mxb(inputs.file.value, cv_support::to_mxb(m_model));
}

namespace
{
// The legacy format this port used to write, kept readable so models saved by
// older builds still load:
//   "CVLM" | uint32 version=1 | uint32 dim | float mean[dim] | float invcov[dim*dim]
// It has neither a covariance matrix nor a sample index.
bool read_cvlm(
    const std::string& path, std::vector<float>& meanOut, std::vector<float>& invcovOut)
{
  std::ifstream is(path, std::ios::binary);
  if(!is)
    return false;

  char magic[4] = {};
  std::uint32_t version = 0u;
  std::uint32_t dim = 0u;
  is.read(magic, 4);
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if(!is || magic[0] != 'C' || magic[1] != 'V' || magic[2] != 'L' || magic[3] != 'M'
     || version != 1u || dim == 0u || dim > 4096u)
    return false;

  std::vector<float> meanVec(dim);
  std::vector<float> flat(static_cast<std::size_t>(dim) * dim);
  is.read(
      reinterpret_cast<char*>(meanVec.data()),
      static_cast<std::streamsize>(meanVec.size() * sizeof(float)));
  is.read(
      reinterpret_cast<char*>(flat.data()),
      static_cast<std::streamsize>(flat.size() * sizeof(float)));
  if(!is)
    return false;

  meanOut = std::move(meanVec);
  invcovOut = std::move(flat);
  return true;
}
}

bool Learn::load_model() noexcept
{
  if(inputs.file.value.empty())
    return false;

  // cv.jit .mxb first (both endiannesses); it is the richer format.
  cv_support::MxbModel mxb;
  if(cv_support::read_mxb(inputs.file.value, mxb))
  {
    cv_support::OnlineModel m;
    if(!cv_support::from_mxb(mxb, m))
      return false;
    m_model = std::move(m);
    m_covarianceKnown = true;
    m_samples.clear();
    publish();
    return true;
  }

  // Legacy CVLM fallback.
  std::vector<float> meanVec, flat;
  if(!read_cvlm(inputs.file.value, meanVec, flat))
    return false;

  const int dim = static_cast<int>(meanVec.size());
  m_model.allocate(dim);
  m_model.index = 0.0; // not recorded by the legacy format
  for(int i = 0; i < dim; ++i)
    m_model.mean(i) = static_cast<double>(meanVec[static_cast<std::size_t>(i)]);
  for(int i = 0; i < dim; ++i)
    for(int j = 0; j < dim; ++j)
      m_model.inverse(i, j)
          = static_cast<double>(flat[static_cast<std::size_t>(i) * dim + j]);
  m_covarianceKnown = false;
  m_samples.clear();
  publish();
  return true;
}
}
