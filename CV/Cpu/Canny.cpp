#include "Canny.hpp"

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

// OpenCV's BORDER_REPLICATE: aaaaaa|abcdefgh|hhhhhhh (the border pixel is repeated).
// This -- not reflect-101 -- is what cv::Canny passes to its Sobel calls
// (modules/imgproc/src/canny.cpp: Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0,
// BORDER_REPLICATE)), and the difference is *not* cosmetic: under reflect-101 both x-1 and
// x+1 map to the same interior index on the border column, so gx there is identically 0 and
// the frame can never carry an edge. Under replicate gx(0) = 4*(s(1) - s(0)), which is the
// real one-sided difference.
inline int replicate(int p, int len) noexcept
{
  if(p < 0)
    return 0;
  if(p >= len)
    return len - 1;
  return p;
}

// tan(22.5 deg) and tan(67.5 deg): the two sector boundaries used to quantise the gradient
// direction into {horizontal, vertical, diagonal /, diagonal \}.
constexpr float TG22 = 0.41421356237309503f;
constexpr float TG67 = 2.41421356237309503f;
}

void Canny::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const std::size_t N = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
  const auto src = cv_support::as_rgba(in);

  // cv.jit.canny: low/high are derived from `threshold` +/- `range`, each clipped to [0,255].
  // cv::Canny then FLOORS both to int (canny.cpp: `int low = cvFloor(low_thresh); int high =
  // cvFloor(high_thresh);`) before any comparison. cv.jit's `threshold`/`range` are float64,
  // so a non-integral band such as [150.7, 160.2] is reachable from the patcher and must
  // behave as [150, 160] -- hence the std::floor here rather than a raw float compare.
  const float thr = std::clamp(inputs.threshold.value, 0.f, 255.f);
  const float rng = std::clamp(inputs.range.value, 0.f, 255.f);
  const float low = std::floor(std::clamp(thr - rng, 0.f, 255.f));
  const float high = std::floor(std::clamp(thr + rng, 0.f, 255.f));

  // The magnitude map is padded with a 1-pixel ring of zeros, exactly like OpenCV's
  // `mag_buf` (which is `mapstep = cols + 2` wide, with `_norm[-1] = _norm[cols] = 0` and a
  // zeroed row above the first and below the last). That ring is what lets NMS run over
  // *every* image pixel, including the frame, without a special case -- see the header.
  const int MW = W + 2;
  const int MH = H + 2;

  m_gray.resize(N);
  m_tmp.resize(N);
  m_gx.resize(N);
  m_gy.resize(N);
  m_mag.assign(static_cast<std::size_t>(MW) * static_cast<std::size_t>(MH), 0.f);
  m_state.assign(N, 0);

  // --- 1. Rec.601 luminance in [0,255] -----------------------------------------------------
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t* p = src.data + i * 4;
    m_gray[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
  }

  // --- 2. OPTIONAL separable 3x3 binomial pre-smooth ([1 2 1]/4), reflect-101 borders -------
  // cv::Canny (what cv.jit.canny calls) does NOT pre-smooth, so this is off by default and
  // the default path is cv.jit-faithful. See the header for why it is offered at all.
  if(inputs.presmooth)
  {
    for(int y = 0; y < H; ++y)
    {
      const float* row = m_gray.data() + static_cast<std::size_t>(y) * W;
      float* dst = m_tmp.data() + static_cast<std::size_t>(y) * W;
      for(int x = 0; x < W; ++x)
      {
        const float a = row[reflect101(x - 1, W)];
        const float b = row[x];
        const float c = row[reflect101(x + 1, W)];
        dst[x] = (a + 2.f * b + c) * 0.25f;
      }
    }
    for(int y = 0; y < H; ++y)
    {
      const float* up = m_tmp.data() + static_cast<std::size_t>(reflect101(y - 1, H)) * W;
      const float* mid = m_tmp.data() + static_cast<std::size_t>(y) * W;
      const float* dn = m_tmp.data() + static_cast<std::size_t>(reflect101(y + 1, H)) * W;
      float* dst = m_gray.data() + static_cast<std::size_t>(y) * W;
      for(int x = 0; x < W; ++x)
        dst[x] = (up[x] + 2.f * mid[x] + dn[x]) * 0.25f;
    }
  }

  // --- 3. 3x3 Sobel (aperture hardwired to 3, unnormalised as in OpenCV) --------------------
  // Borders are BORDER_REPLICATE, which is what cv::Canny explicitly asks its Sobel for.
  for(int y = 0; y < H; ++y)
  {
    const int ym = replicate(y - 1, H);
    const int yp = replicate(y + 1, H);
    const float* r0 = m_gray.data() + static_cast<std::size_t>(ym) * W;
    const float* r1 = m_gray.data() + static_cast<std::size_t>(y) * W;
    const float* r2 = m_gray.data() + static_cast<std::size_t>(yp) * W;
    float* gxr = m_gx.data() + static_cast<std::size_t>(y) * W;
    float* gyr = m_gy.data() + static_cast<std::size_t>(y) * W;
    float* magr = m_mag.data() + static_cast<std::size_t>(y + 1) * MW + 1;
    for(int x = 0; x < W; ++x)
    {
      const int xm = replicate(x - 1, W);
      const int xp = replicate(x + 1, W);
      const float p00 = r0[xm], p01 = r0[x], p02 = r0[xp];
      const float p10 = r1[xm], p12 = r1[xp];
      const float p20 = r2[xm], p21 = r2[x], p22 = r2[xp];

      const float gx = (p02 + 2.f * p12 + p22) - (p00 + 2.f * p10 + p20);
      const float gy = (p20 + 2.f * p21 + p22) - (p00 + 2.f * p01 + p02);
      gxr[x] = gx;
      gyr[x] = gy;
      // L2gradient = false -> L1 norm, matching cv::Canny's default.
      magr[x] = std::fabs(gx) + std::fabs(gy);
    }
  }

  // --- 4. Non-maximum suppression + 5. double-threshold classification ----------------------
  // state: 0 = suppressed/below low, 1 = weak (low < m <= high), 2 = strong seed (m > high).
  //
  // Every image pixel is evaluated, frame included: the zero ring around m_mag stands in for
  // the missing neighbours, which is precisely what OpenCV does (its NMS loop is
  // `for(int j = 0; j < src.cols; j++)` over rows 0..rows-1 of a (rows+2)x(cols+2) map).
  m_stack.clear();
  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      const std::size_t i = static_cast<std::size_t>(y) * W + x;
      const std::size_t mi = static_cast<std::size_t>(y + 1) * MW + (x + 1);
      const float m = m_mag[mi];
      if(!(m > low))
        continue;

      const float gx = m_gx[i];
      const float gy = m_gy[i];
      const float ax = std::fabs(gx);
      const float ay = std::fabs(gy);

      bool keep;
      if(ay <= TG22 * ax)
      {
        // Gradient mostly horizontal -> the ridge is vertical: compare left/right.
        keep = m > m_mag[mi - 1] && m >= m_mag[mi + 1];
      }
      else if(ay >= TG67 * ax)
      {
        // Gradient mostly vertical -> the ridge is horizontal: compare up/down.
        keep = m > m_mag[mi - MW] && m >= m_mag[mi + MW];
      }
      else
      {
        // Diagonal sector. s = +1 when gx and gy share a sign (gradient along '\'),
        // -1 otherwise.
        //
        // BOTH comparisons are strict here, unlike the two axis-aligned cases. That is not a
        // typo: it is what OpenCV does
        //   (`if (m > _mag_p[j-s] && m > _mag_n[j+s])`, canny.cpp, versus
        //    `m > _mag_a[j-1] && m >= _mag_a[j+1]` and `m > _mag_p[j] && m >= _mag_n[j]`),
        // and it has a visible consequence: a plateau of equal magnitudes along a diagonal
        // yields ZERO edge pixels, whereas the same plateau along a row or column yields
        // exactly one (the first). Relaxing the second test to `>=` "for symmetry" changes
        // the output on 93% of random binary 9x9 images and inflates the edge count on
        // continuous-tone images by about 2%.
        const int s = (gx * gy < 0.f) ? -1 : 1;
        keep = m > m_mag[mi - MW - s] && m > m_mag[mi + MW + s];
      }

      if(!keep)
        continue;

      if(m > high)
      {
        m_state[i] = 2;
        m_stack.push_back(static_cast<std::int32_t>(i));
      }
      else
      {
        m_state[i] = 1;
      }
    }
  }

  // --- 6. Hysteresis: promote weak pixels 8-connected to a strong one -----------------------
  // Explicit stack, never recursion: an image-sized edge chain would blow the call stack.
  while(!m_stack.empty())
  {
    const std::int32_t i = m_stack.back();
    m_stack.pop_back();
    const int x = i % W;
    const int y = i / W;
    for(int dy = -1; dy <= 1; ++dy)
    {
      const int ny = y + dy;
      if(ny < 0 || ny >= H)
        continue;
      for(int dx = -1; dx <= 1; ++dx)
      {
        if(dx == 0 && dy == 0)
          continue;
        const int nx = x + dx;
        if(nx < 0 || nx >= W)
          continue;
        const std::size_t j = static_cast<std::size_t>(ny) * W + nx;
        if(m_state[j] == 1)
        {
          m_state[j] = 2;
          m_stack.push_back(static_cast<std::int32_t>(j));
        }
      }
    }
  }

  // --- 7. Emit the binary 0/255 edge map ----------------------------------------------------
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  for(std::size_t i = 0; i < N; ++i)
    out.bytes[i] = (m_state[i] == 2) ? std::uint8_t{255} : std::uint8_t{0};

  outputs.image.upload();
}
}
