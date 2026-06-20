#include "OrbFeatures.hpp"

#include <CV/Support/Brief.hpp>
#include <CV/Support/EigenImage.hpp>

#include <vector>

namespace cv
{
void OrbFeatures::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || !in.width || !in.height)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  std::vector<cv_support::kp> kps;
  cv_support::detectAndDescribe(
      src.data, W, H, inputs.threshold.value, inputs.max_features.value, m_gray, kps);

  const float invW = 1.f / W;
  const float invH = 1.f / H;

  auto& out = outputs.keypoints.value;
  out.clear();
  out.reserve(kps.size());
  for(const auto& k : kps)
  {
    keypoint p;
    p.position = {k.x * invW, k.y * invH};
    p.angle = k.angle;
    p.desc = k.desc;
    out.push_back(p);
  }

  outputs.count = static_cast<int>(out.size());
}
}
