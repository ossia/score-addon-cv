#include "FloodFill.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <utility>

namespace cv
{
void FloodFill::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);
  const std::size_t N = static_cast<std::size_t>(W) * H;

  auto luma = [&](int x, int y) -> float {
    const std::uint8_t* p = src.data + (static_cast<std::size_t>(y) * W + x) * 4;
    return (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  };

  const int sx = std::clamp(
      static_cast<int>(inputs.seed.value.x * W), 0, W - 1);
  const int sy = std::clamp(
      static_cast<int>(inputs.seed.value.y * H), 0, H - 1);
  const float seedLuma = luma(sx, sy);
  const float tol = inputs.tolerance.value;

  m_mask.assign(N, 0);

  // Scanline flood fill with an explicit stack of seed points. Each pop expands the maximal
  // horizontal run, then seeds the rows above and below where new same-region pixels appear.
  std::vector<std::pair<int, int>> stack;
  stack.reserve(256);
  stack.emplace_back(sx, sy);

  int filled = 0;
  auto similar = [&](int x, int y) -> bool {
    return std::abs(luma(x, y) - seedLuma) <= tol;
  };

  while(!stack.empty())
  {
    auto [px, py] = stack.back();
    stack.pop_back();
    if(m_mask[static_cast<std::size_t>(py) * W + px])
      continue;
    if(!similar(px, py))
      continue;

    // Expand left and right to the run extents.
    int xl = px;
    while(xl > 0 && !m_mask[static_cast<std::size_t>(py) * W + (xl - 1)]
          && similar(xl - 1, py))
      --xl;
    int xr = px;
    while(xr < W - 1 && !m_mask[static_cast<std::size_t>(py) * W + (xr + 1)]
          && similar(xr + 1, py))
      ++xr;

    // Fill the run and seed adjacent rows.
    for(int x = xl; x <= xr; ++x)
    {
      m_mask[static_cast<std::size_t>(py) * W + x] = 1;
      ++filled;
      if(py > 0)
      {
        int ny = py - 1;
        if(!m_mask[static_cast<std::size_t>(ny) * W + x] && similar(x, ny))
          stack.emplace_back(x, ny);
      }
      if(py < H - 1)
      {
        int ny = py + 1;
        if(!m_mask[static_cast<std::size_t>(ny) * W + x] && similar(x, ny))
          stack.emplace_back(x, ny);
      }
    }
  }

  outputs.filled = filled;

  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  for(std::size_t i = 0; i < N; ++i)
    out.bytes[i] = m_mask[i] ? std::uint8_t{255} : std::uint8_t{0};
  out.changed = true;
}
}
