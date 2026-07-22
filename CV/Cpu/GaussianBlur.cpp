#include "GaussianBlur.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>

namespace cv
{
namespace
{
// OpenCV's BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba (the border pixel is not repeated).
inline int reflect101(int p, int len) noexcept
{
  if(static_cast<unsigned>(p) < static_cast<unsigned>(len))
    return p;
  if(len == 1)
    return 0;
  do
  {
    if(p < 0)
      p = -p;
    else
      p = 2 * (len - 1) - p;
  } while(static_cast<unsigned>(p) >= static_cast<unsigned>(len));
  return p;
}

// OpenCV's hardcoded small-kernel table (modules/imgproc/src/smooth.dispatch.cpp,
// `small_gaussian_tab`, SMALL_GAUSSIAN_SIZE == 7). getGaussianKernel() -- which
// createGaussianKernels(), and hence cv::GaussianBlur, goes through -- returns one of these
// VERBATIM and never evaluates the sigma formula when
//     n is odd  &&  n <= 7  &&  sigma <= 0
// Rows are indexed n >> 1, i.e. by the radius, so row r is the (2r+1)-tap kernel. Each row
// already sums to exactly 1, so the normalisation below is a no-op on them.
//
// Skipping this table is not a rounding detail: it bites the DEFAULT. cv.jit.blur ships
// `radius 1, sigma 0` -> n = 3, where the table centre tap is 0.5 but the formula (sigma =
// 0.8) gives 0.522011 -- 0.022 off, i.e. 5.6 levels out of 255 on the centre tap alone.
constexpr int small_gaussian_size = 7;
constexpr float small_gaussian_tab[4][small_gaussian_size] = {
    {1.f},
    {0.25f, 0.5f, 0.25f},
    {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
    {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};
}

void GaussianBlur::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  const int radius = std::max(1, inputs.radius.value);
  const int k = radius * 2 + 1;

  // --- 1D kernel, normalised so it sums to 1 -------------------------------------------------
  // Two branches, and the ORDER matters: OpenCV consults the fixed small-kernel table FIRST
  // and returns early, so for sigma <= 0 with k in {1,3,5,7} the auto-sigma formula is never
  // evaluated at all. Only outside that window (sigma > 0, or k >= 9) does the formula run.
  const float sigma = inputs.sigma.value;
  m_kernel.resize(static_cast<std::size_t>(k));
  if(!(sigma > 0.f) && k <= small_gaussian_size)
  {
    // k is always odd here (k = 2*radius + 1), so k >> 1 == radius indexes the table row.
    const float* row = small_gaussian_tab[k >> 1];
    for(int i = 0; i < k; ++i)
      m_kernel[static_cast<std::size_t>(i)] = row[i];
  }
  else
  {
    const float s = (sigma > 0.f) ? sigma : gaussian_auto_sigma(k);
    const float scale = -0.5f / (s * s);
    float sum = 0.f;
    for(int i = 0; i < k; ++i)
    {
      const float d = static_cast<float>(i - radius);
      const float v = std::exp(scale * d * d);
      m_kernel[static_cast<std::size_t>(i)] = v;
      sum += v;
    }
    const float inv = 1.f / sum;
    for(int i = 0; i < k; ++i)
      m_kernel[static_cast<std::size_t>(i)] *= inv;
  }
  const float* kern = m_kernel.data();

  // --- Horizontal pass: RGB (alpha is not blurred) into a float scratch -----------------------
  m_tmp.resize(static_cast<std::size_t>(W) * static_cast<std::size_t>(H) * 3u);
  for(int y = 0; y < H; ++y)
  {
    const std::uint8_t* srow = src.data + static_cast<std::size_t>(y) * W * 4;
    float* drow = m_tmp.data() + static_cast<std::size_t>(y) * W * 3;
    for(int x = 0; x < W; ++x)
    {
      float r = 0.f, g = 0.f, b = 0.f;
      for(int t = 0; t < k; ++t)
      {
        const int sx = reflect101(x + t - radius, W);
        const float w = kern[t];
        const std::uint8_t* p = srow + static_cast<std::size_t>(sx) * 4;
        r += w * p[0];
        g += w * p[1];
        b += w * p[2];
      }
      float* d = drow + static_cast<std::size_t>(x) * 3;
      d[0] = r;
      d[1] = g;
      d[2] = b;
    }
  }

  // --- Vertical pass: scratch -> output, alpha copied straight through ------------------------
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  for(int y = 0; y < H; ++y)
  {
    const std::uint8_t* srow = src.data + static_cast<std::size_t>(y) * W * 4;
    std::uint8_t* drow = out.bytes + static_cast<std::size_t>(y) * W * 4;
    for(int x = 0; x < W; ++x)
    {
      float r = 0.f, g = 0.f, b = 0.f;
      for(int t = 0; t < k; ++t)
      {
        const int sy = reflect101(y + t - radius, H);
        const float w = kern[t];
        const float* p
            = m_tmp.data() + (static_cast<std::size_t>(sy) * W + x) * 3;
        r += w * p[0];
        g += w * p[1];
        b += w * p[2];
      }
      std::uint8_t* d = drow + static_cast<std::size_t>(x) * 4;
      d[0] = static_cast<std::uint8_t>(std::clamp(r + 0.5f, 0.f, 255.f));
      d[1] = static_cast<std::uint8_t>(std::clamp(g + 0.5f, 0.f, 255.f));
      d[2] = static_cast<std::uint8_t>(std::clamp(b + 0.5f, 0.f, 255.f));
      d[3] = srow[static_cast<std::size_t>(x) * 4 + 3];
    }
  }

  outputs.image.upload();
}
}
