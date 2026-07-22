#include "FloodFill.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>
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

  auto luma255 = [&](int x, int y) -> float {
    const std::uint8_t* p = src.data + (static_cast<std::size_t>(y) * W + x) * 4;
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
  };

  m_mask.assign(N, 0);

  // Emits the (already cleared) mask and the count. Used for every early exit too, so an
  // out-of-bounds / background seed still produces a valid, empty frame instead of leaving
  // a stale one — cv.jit clears its output matrix before calling the fill.
  auto emit = [&](int filled) {
    outputs.filled = filled;
    outputs.image.create(W, H);
    auto& out = outputs.image.texture;
    for(std::size_t i = 0; i < N; ++i)
      out.bytes[i] = m_mask[i] ? std::uint8_t{255} : std::uint8_t{0};
    out.changed = true;
  };

  // ---- Seed selection -----------------------------------------------------------------
  int sx = 0, sy = 0;
  if(inputs.seed_mode.value == FloodSeedMode::Pixels)
  {
    sx = inputs.seed_px.value.x;
    sy = inputs.seed_px.value.y;
    // cv.jit: `if((seedx >= width)||(seedx < 0)||(seedy >= height)||(seedy < 0)) return 0;`
    if(sx < 0 || sx >= W || sy < 0 || sy >= H)
    {
      emit(0);
      return;
    }
  }
  else
  {
    // Normalised seed stays clamped (legacy behaviour): it cannot be out of bounds.
    sx = std::clamp(static_cast<int>(inputs.seed.value.x * W), 0, W - 1);
    sy = std::clamp(static_cast<int>(inputs.seed.value.y * H), 0, H - 1);
  }

  // ---- Region predicate ---------------------------------------------------------------
  const bool binary = inputs.mode.value == FloodMode::Binary;
  const float thr = std::clamp(inputs.threshold.value, 0.f, 1.f) * 255.f;
  const float seedLuma = luma255(sx, sy);
  const float tol = inputs.tolerance.value * 255.f;

  // Binary: cv.jit's `in[p] != 0` on the binarised input (Threshold == 0 -> exactly that).
  // Tolerance: closeness to the seed's own luminance.
  auto similar = [&](int x, int y) -> bool {
    const float l = luma255(x, y);
    return binary ? (l > thr) : (std::abs(l - seedLuma) <= tol);
  };

  // cv.jit: `if(inData[seedx] == 0) return 0;` -- a seed on a background pixel fills
  // nothing and is not an error. In Tolerance mode the seed is trivially similar to
  // itself, so this only ever triggers in Binary mode.
  if(!similar(sx, sy))
  {
    emit(0);
    return;
  }

  // ---- Scanline flood fill, 4-connected -----------------------------------------------
  // Each pop expands the maximal horizontal run, then seeds the rows above and below at
  // the same x. A pixel that only touches the region diagonally is never reached, which is
  // what makes this 4- and not 8-connected.
  //
  // Every pixel is marked exactly once, so the loop is bounded by W*H: unlike cv.jit there
  // is no iteration cap and no silent truncation.
  std::vector<std::pair<int, int>> stack;
  stack.reserve(256);
  stack.emplace_back(sx, sy);

  int filled = 0;
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

  emit(filled);
}
}
