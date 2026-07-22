#include "OrbFeatures.hpp"

#include <CV/Support/Brief.hpp>
#include <CV/Support/EigenImage.hpp>

#include <algorithm>
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
  cv_support::detectAndDescribeMulti(
      src.data, W, H, inputs.threshold.value, inputs.max_features.value,
      inputs.octaves.value, m_gray, kps);

  // Positions come back in base-image pixels whatever octave they were found at, so a single
  // scaling by the base resolution is correct for every level.
  //
  // cv.jit.keypoints' exact rule (see the header): `normalize` decides the SCALE, `glcoords`
  // decides the CENTRING + Y FLIP, and the two are independent. `normalize` takes precedence
  // for the scale, so @normalize 1 @glcoords 1 gives [-0.5, 0.5] with Y flipped rather than
  // the [-1, 1] a naive reading would produce.
  const auto mode = inputs.coordinates.value;
  const bool normalize = (mode == Coordinates::Normalized);
  // GL implies centring; the toggle adds it to any scale (cv.jit's `glcoords`).
  const bool glcoords = bool(inputs.gl_centering.value) || (mode == Coordinates::GL);

  const float nsx = 1.f / W;
  const float nsy = 1.f / H;
  const float sx = normalize ? nsx : (glcoords ? 2.f * nsx : 1.f);
  const float sy = normalize ? nsy : (glcoords ? 2.f * nsy : 1.f);
  const float cx = W * 0.5f;
  const float cy = H * 0.5f;
  const float ssize = std::max(sx, sy); // cv.jit scales SIZE by max(scale_x, scale_y)

  auto& out = outputs.keypoints.value;
  out.clear();
  out.reserve(kps.size());
  for(const auto& k : kps)
  {
    keypoint p;
    if(glcoords)
      p.position = {(k.x - cx) * sx, (cy - k.y) * sy}; // Y is FLIPPED, as in cv.jit
    else
      p.position = {k.x * sx, k.y * sy};
    p.size = k.size * ssize;
    p.angle = k.angle;
    p.response = k.response;
    p.octave = k.octave;
    p.desc = k.desc;
    out.push_back(p);
  }

  outputs.count = static_cast<int>(out.size());
}
}
