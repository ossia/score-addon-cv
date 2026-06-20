#include "Label.hpp"

#include <CV/Cpu/ConnectedComponents.hpp>

#include <algorithm>

namespace cv
{
namespace
{
// Deterministic blob colour from a label id.
void label_color(std::int32_t id, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b)
{
  std::uint32_t h = static_cast<std::uint32_t>(id) * 2654435761u;
  r = static_cast<std::uint8_t>(60 + ((h) & 0x7F) + 0x40);
  g = static_cast<std::uint8_t>(60 + ((h >> 8) & 0x7F) + 0x40);
  b = static_cast<std::uint8_t>(60 + ((h >> 16) & 0x7F) + 0x40);
}
}

void Label::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  const std::uint8_t thr = static_cast<std::uint8_t>(
      std::clamp(inputs.threshold.value, 0.f, 1.f) * 255.f + 0.5f);

  auto R = cv_support::label_connected(src, thr, inputs.min_size.value);
  outputs.count = R.count;

  outputs.image.create(W, H);
  outputs.labels.create(W, H);
  auto& out = outputs.image.texture;
  auto& lblOut = outputs.labels.texture;
  float* lblData = reinterpret_cast<float*>(lblOut.bytes);
  const bool colorize = inputs.colorize.value;
  const std::size_t N = static_cast<std::size_t>(W) * H;
  for(std::size_t i = 0; i < N; ++i)
  {
    std::uint8_t* d = out.bytes + i * 4;
    std::int32_t lbl = R.labels[i];

    // Numeric label field: exact id per pixel (0 = background).
    lblData[i] = static_cast<float>(lbl);

    if(lbl == 0)
    {
      d[0] = d[1] = d[2] = 0;
      d[3] = 255;
    }
    else if(colorize)
    {
      label_color(lbl, d[0], d[1], d[2]);
      d[3] = 255;
    }
    else
    {
      std::uint8_t g = static_cast<std::uint8_t>(((lbl - 1) % 255) + 1);
      d[0] = d[1] = d[2] = g;
      d[3] = 255;
    }
  }
  out.changed = true;
  lblOut.changed = true;
}
}
