#include "Morphology.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>

namespace cv
{
namespace
{
// One morphological pass over a planar interleaved buffer (`planes` channels per pixel).
//
// `dilate` selects max (dilate) over the structuring element, otherwise min (erode). On a
// buffer that only holds 0 and 255 -- what binary mode builds -- max is exactly cv.jit's
// `||` over the neighbourhood and min is exactly its `&&`, so a single kernel serves both
// modes.
//
// Taps that fall outside the image are skipped rather than substituted, matching cv.jit's
// hand-written border cases (equivalent to clamp-to-edge, see the header).
void morph_pass(
    const std::uint8_t* __restrict src, std::uint8_t* __restrict dst, int W, int H,
    int planes, int radius, bool cross, bool dilate) noexcept
{
  for(int y = 0; y < H; ++y)
  {
    const int y0 = std::max(0, y - radius);
    const int y1 = std::min(H - 1, y + radius);
    for(int x = 0; x < W; ++x)
    {
      const int x0 = std::max(0, x - radius);
      const int x1 = std::min(W - 1, x + radius);

      // Seed with the centre pixel: it is always part of the structuring element, in both
      // shapes. (cv.jit likewise always includes *ip in the test.)
      std::uint8_t acc[4];
      const std::uint8_t* c = src + (static_cast<std::size_t>(y) * W + x) * planes;
      for(int p = 0; p < planes; ++p)
        acc[p] = c[p];

      for(int ny = y0; ny <= y1; ++ny)
      {
        const bool off_row = (ny != y);
        for(int nx = x0; nx <= x1; ++nx)
        {
          if(cross && off_row && nx != x)
            continue; // cross: no diagonal taps
          const std::uint8_t* s = src + (static_cast<std::size_t>(ny) * W + nx) * planes;
          if(dilate)
          {
            for(int p = 0; p < planes; ++p)
              acc[p] = std::max(acc[p], s[p]);
          }
          else
          {
            for(int p = 0; p < planes; ++p)
              acc[p] = std::min(acc[p], s[p]);
          }
        }
      }

      std::uint8_t* d = dst + (static_cast<std::size_t>(y) * W + x) * planes;
      for(int p = 0; p < planes; ++p)
        d[p] = acc[p];
    }
  }
}
}

void Morphology::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width <= 0 || in.height <= 0)
    return;

  // NOTE: cv.jit uses MAX(in.dim, out.dim) here, which over-runs the input buffer whenever
  // the output matrix is larger. We size everything from the input. See the header.
  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);
  const std::size_t npx = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);

  const int radius = std::clamp(inputs.radius.value, 1, 16);
  const bool cross = inputs.shape.value == MorphShape::Cross;
  const bool binary = inputs.binary.value;
  const int planes = binary ? 1 : 3;

  // --- Build the source plane(s) --------------------------------------------------------
  m_a.resize(npx * static_cast<std::size_t>(planes));
  m_b.resize(npx * static_cast<std::size_t>(planes));

  if(binary)
  {
    // Foreground = Rec.601 luma > Threshold*255. Threshold 0 == cv.jit's `!= 0` test.
    const float thr = std::clamp(inputs.threshold.value, 0.f, 1.f) * 255.f;
    for(std::size_t i = 0; i < npx; ++i)
    {
      const std::uint8_t* p = src.data + i * 4;
      const float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
      m_a[i] = (luma > thr) ? 255 : 0;
    }
  }
  else
  {
    for(std::size_t i = 0; i < npx; ++i)
    {
      const std::uint8_t* p = src.data + i * 4;
      std::uint8_t* d = m_a.data() + i * 3;
      d[0] = p[0];
      d[1] = p[1];
      d[2] = p[2];
    }
  }

  // --- Run the requested op (one pass, or two for open / close) --------------------------
  // Open = erode then dilate; Close = dilate then erode.
  const auto op = inputs.operation.value;
  const bool first_dilate = (op == MorphOperation::Dilate || op == MorphOperation::Close);
  const bool two_passes = (op == MorphOperation::Open || op == MorphOperation::Close);

  morph_pass(m_a.data(), m_b.data(), W, H, planes, radius, cross, first_dilate);
  const std::uint8_t* result = m_b.data();
  if(two_passes)
  {
    morph_pass(m_b.data(), m_a.data(), W, H, planes, radius, cross, !first_dilate);
    result = m_a.data();
  }

  // --- Write back to RGBA8, alpha straight from the input --------------------------------
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  for(std::size_t i = 0; i < npx; ++i)
  {
    std::uint8_t* d = out.bytes + i * 4;
    if(binary)
    {
      const std::uint8_t v = result[i]; // already exactly 0 or 255
      d[0] = v;
      d[1] = v;
      d[2] = v;
    }
    else
    {
      const std::uint8_t* s = result + i * 3;
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
    }
    d[3] = src.data[i * 4 + 3];
  }

  outputs.image.upload();
}
}
