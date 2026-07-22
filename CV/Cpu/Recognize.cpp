#include "Recognize.hpp"

#include <CV/Support/Mahalanobis.hpp>

#include <array>

namespace cv
{
namespace
{
// The 7 numbers cv.jit.blobs.recon feeds the classifier, per `mode`.
std::array<float, Recognize::blob_feature_size>
blob_features(const blob_info& b, ReconMode mode) noexcept
{
  if(mode == ReconMode::Hu)
    return {b.hu[0], b.hu[1], b.hu[2], b.hu[3], b.hu[4], b.hu[5], b.hu[6]};
  // cv.jit.blobs.moments plane order for the normalised central moments.
  return {b.nu20, b.nu02, b.nu11, b.nu21, b.nu12, b.nu30, b.nu03};
}
}

void Recognize::operator()() noexcept
{
  // The packed "Model" inlet, when present, overrides the separate Mean /
  // Inverse covariance inlets (so a model trained by cv::Learn can be fed in
  // through a single cable / a saved project).
  const std::vector<float>* mean = &inputs.mean.value;
  const std::vector<float>* invcov = &inputs.invcov.value;

  std::vector<float> unpackedMean, unpackedInvcov;
  bool modelBroken = false;
  if(!inputs.model.value.empty())
  {
    int dim = 0;
    if(cv_support::unpack_model(inputs.model.value, dim, unpackedMean, unpackedInvcov))
    {
      mean = &unpackedMean;
      invcov = &unpackedInvcov;
    }
    else
    {
      modelBroken = true;
    }
  }

  // ------------------------------------------------------------ single vector
  if(modelBroken)
  {
    // Malformed model: clearly invalid result, never a perfect match.
    outputs.distance = invalid_distance;
    outputs.match = false;
    outputs.valid = false;
  }
  else
  {
    const double d = cv_support::mahalanobis(inputs.feature.value, *mean, *invcov);
    if(d < 0.0)
    {
      // Dimension mismatch / invalid model: report a clearly-invalid result.
      // A large sentinel distance (NOT 0, which read as a perfect match) plus
      // valid=false so downstream logic can reject it unambiguously.
      outputs.distance = invalid_distance;
      outputs.match = false;
      outputs.valid = false;
    }
    else
    {
      outputs.distance = static_cast<float>(d);
      outputs.match = d < static_cast<double>(inputs.threshold.value);
      outputs.valid = true;
    }
  }

  // ------------------------------------------------------------- list / blobs
  // cv.jit.blobs.recon semantics: one distance per candidate per frame, in
  // input order. Every failure path still emits the large sentinel rather than
  // 0, so a distance threshold downstream never sees a spurious perfect match.
  auto& out = outputs.distances.value;
  out.clear();

  const std::size_t dim = modelBroken ? 0u : mean->size();
  const bool modelOk
      = !modelBroken && dim > 0 && invcov->size() == dim * dim;

  if(!inputs.blobs.value.empty())
  {
    const std::size_t n = inputs.blobs.value.size();
    out.resize(n, invalid_distance);
    if(modelOk && dim == static_cast<std::size_t>(blob_feature_size))
    {
      std::vector<float> feat(static_cast<std::size_t>(blob_feature_size));
      for(std::size_t i = 0; i < n; ++i)
      {
        const auto f = blob_features(inputs.blobs.value[i], inputs.mode.value);
        for(int k = 0; k < blob_feature_size; ++k)
          feat[static_cast<std::size_t>(k)] = f[static_cast<std::size_t>(k)];
        const double d = cv_support::mahalanobis(feat, *mean, *invcov);
        out[i] = (d < 0.0) ? invalid_distance : static_cast<float>(d);
      }
    }
    // else: model dimension does not match a 7-element blob descriptor ->
    // every entry stays at the sentinel.
  }

  if(!inputs.features.value.empty())
  {
    const std::size_t total = inputs.features.value.size();
    if(!modelOk || (total % dim) != 0)
    {
      // Ragged / no usable model: still emit sentinels, never zeros. Emit as
      // many as we can guess at, at least one so the failure is visible.
      const std::size_t n = (modelOk && dim > 0) ? (total / dim) : 0u;
      out.insert(out.end(), n > 0 ? n : 1u, invalid_distance);
    }
    else
    {
      const std::size_t n = total / dim;
      std::vector<float> feat(dim);
      for(std::size_t i = 0; i < n; ++i)
      {
        for(std::size_t k = 0; k < dim; ++k)
          feat[k] = inputs.features.value[i * dim + k];
        const double d = cv_support::mahalanobis(feat, *mean, *invcov);
        out.push_back((d < 0.0) ? invalid_distance : static_cast<float>(d));
      }
    }
  }
}
}
