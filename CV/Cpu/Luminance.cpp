#include "Luminance.hpp"

#include <CV/Support/EigenImage.hpp>

namespace cv
{
void Luminance::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const auto src = cv_support::as_rgba(in);

  // Allocate the single-channel output and fill it with luma directly.
  outputs.image.create(src.width, src.height);
  auto& out = outputs.image.texture;

  cv_support::to_gray<std::uint8_t>(src, out.bytes);

  out.changed = true;
}
}
