#include "Recognize.hpp"

#include <CV/Support/Mahalanobis.hpp>

namespace cv
{
void Recognize::operator()() noexcept
{
  // The packed "Model" inlet, when present, overrides the separate Mean /
  // Inverse covariance inlets (so a model trained by cv::Learn can be fed in
  // through a single cable / a saved project).
  const std::vector<float>* mean = &inputs.mean.value;
  const std::vector<float>* invcov = &inputs.invcov.value;

  std::vector<float> unpackedMean, unpackedInvcov;
  if(!inputs.model.value.empty())
  {
    int dim = 0;
    if(cv_support::unpack_model(
           inputs.model.value, dim, unpackedMean, unpackedInvcov))
    {
      mean = &unpackedMean;
      invcov = &unpackedInvcov;
    }
    else
    {
      // Malformed model: clearly invalid result, never a perfect match.
      outputs.distance = invalid_distance;
      outputs.match = false;
      outputs.valid = false;
      return;
    }
  }

  const double d = cv_support::mahalanobis(inputs.feature.value, *mean, *invcov);

  if(d < 0.0)
  {
    // Dimension mismatch / invalid model: report a clearly-invalid result.
    // A large sentinel distance (NOT 0, which read as a perfect match) plus
    // valid=false so downstream logic can reject it unambiguously.
    outputs.distance = invalid_distance;
    outputs.match = false;
    outputs.valid = false;
    return;
  }

  outputs.distance = static_cast<float>(d);
  outputs.match = d < static_cast<double>(inputs.threshold.value);
  outputs.valid = true;
}
}
