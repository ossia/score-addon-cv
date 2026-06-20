#include "Learn.hpp"

#include <CV/Support/Mahalanobis.hpp>

#include <cstdint>
#include <fstream>

namespace cv
{
void Learn::operator()() noexcept
{
  // Reset: drop all accumulated samples and clear the model output.
  if(inputs.reset.value && !m_prevReset)
  {
    m_samples.clear();
    outputs.mean.value.clear();
    outputs.invcov.value.clear();
    outputs.model.value.clear();
    outputs.samples = 0;
    outputs.dimension = 0;
    outputs.valid = false;
  }
  m_prevReset = inputs.reset.value;

  // Capture (rising edge): add the current feature vector to the training set.
  if(inputs.capture.value && !m_prevCapture)
  {
    const auto& f = inputs.feature.value;
    if(!f.empty())
    {
      m_samples.push_back(f);
      // Bound memory: drop the oldest sample once over the cap.
      if(m_samples.size() > max_samples)
        m_samples.erase(m_samples.begin());
    }
  }
  m_prevCapture = inputs.capture.value;

  // Train (rising edge): (re)compute the statistical model.
  if(inputs.train.value && !m_prevTrain)
    retrain();
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

  outputs.samples = static_cast<int>(m_samples.size());
  outputs.valid = model.valid;

  if(!model.valid)
  {
    // Not enough / inconsistent samples (< dim + 1, zero variance, ...): report
    // a clearly-empty, invalid model rather than a half-trained one.
    outputs.dimension = 0;
    outputs.mean.value.clear();
    outputs.invcov.value.clear();
    outputs.model.value.clear();
    return;
  }

  const int dim = model.dim;
  outputs.dimension = dim;

  std::vector<float> meanVec(static_cast<std::size_t>(dim));
  for(int i = 0; i < dim; ++i)
    meanVec[static_cast<std::size_t>(i)] = static_cast<float>(model.mean(i));

  std::vector<float> flat(static_cast<std::size_t>(dim) * dim);
  for(int i = 0; i < dim; ++i)
    for(int j = 0; j < dim; ++j)
      flat[static_cast<std::size_t>(i) * dim + j]
          = static_cast<float>(model.invCov(i, j));

  // GOTCHA: assign through .value for aggregate/vector val_ports.
  outputs.mean.value = meanVec;
  outputs.invcov.value = flat;
  publish_model();
}

void Learn::publish_model() noexcept
{
  outputs.model.value = cv_support::pack_model(
      outputs.dimension.value, outputs.mean.value, outputs.invcov.value);
}

// On-disk format: a tiny self-contained binary blob.
//   magic "CVLM" | uint32 version=1 | uint32 dim | float mean[dim] | float invcov[dim*dim]
bool Learn::save_model() const noexcept
{
  if(inputs.file.value.empty() || !outputs.valid.value)
    return false;

  std::ofstream os(inputs.file.value, std::ios::binary | std::ios::trunc);
  if(!os)
    return false;

  const char magic[4] = {'C', 'V', 'L', 'M'};
  const std::uint32_t version = 1u;
  const std::uint32_t dim = static_cast<std::uint32_t>(outputs.dimension.value);

  os.write(magic, 4);
  os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  os.write(
      reinterpret_cast<const char*>(outputs.mean.value.data()),
      static_cast<std::streamsize>(outputs.mean.value.size() * sizeof(float)));
  os.write(
      reinterpret_cast<const char*>(outputs.invcov.value.data()),
      static_cast<std::streamsize>(outputs.invcov.value.size() * sizeof(float)));
  return static_cast<bool>(os);
}

bool Learn::load_model() noexcept
{
  if(inputs.file.value.empty())
    return false;

  std::ifstream is(inputs.file.value, std::ios::binary);
  if(!is)
    return false;

  char magic[4] = {};
  std::uint32_t version = 0u;
  std::uint32_t dim = 0u;
  is.read(magic, 4);
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if(!is || magic[0] != 'C' || magic[1] != 'V' || magic[2] != 'L'
     || magic[3] != 'M' || version != 1u || dim == 0u)
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

  // Populate the same outputs as a fresh train.
  outputs.dimension = static_cast<int>(dim);
  outputs.mean.value = std::move(meanVec);
  outputs.invcov.value = std::move(flat);
  outputs.valid = true;
  publish_model();
  return true;
}
}
