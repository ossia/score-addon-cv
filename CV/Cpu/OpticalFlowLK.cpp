#include "OpticalFlowLK.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>

namespace cv
{
namespace
{
struct GrayImage
{
  std::vector<float> px;
  int w = 0;
  int h = 0;
};

inline float sampleNearest(const GrayImage& img, int x, int y)
{
  x = std::clamp(x, 0, img.w - 1);
  y = std::clamp(y, 0, img.h - 1);
  return img.px[static_cast<std::size_t>(y) * img.w + x];
}

// Bilinear sample at fractional coordinates (clamped to the image extent).
inline float sampleBilinear(const GrayImage& img, float x, float y)
{
  x = std::clamp(x, 0.f, static_cast<float>(img.w - 1));
  y = std::clamp(y, 0.f, static_cast<float>(img.h - 1));
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, img.w - 1);
  const int y1 = std::min(y0 + 1, img.h - 1);
  const float fx = x - x0;
  const float fy = y - y0;
  const float a = img.px[static_cast<std::size_t>(y0) * img.w + x0];
  const float b = img.px[static_cast<std::size_t>(y0) * img.w + x1];
  const float c = img.px[static_cast<std::size_t>(y1) * img.w + x0];
  const float d = img.px[static_cast<std::size_t>(y1) * img.w + x1];
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

// 2x2 box-downsample (halving) producing the next coarser pyramid level.
GrayImage downsample(const GrayImage& src)
{
  GrayImage dst;
  dst.w = std::max(1, src.w / 2);
  dst.h = std::max(1, src.h / 2);
  dst.px.resize(static_cast<std::size_t>(dst.w) * dst.h);
  for(int y = 0; y < dst.h; ++y)
  {
    for(int x = 0; x < dst.w; ++x)
    {
      const int sx = x * 2;
      const int sy = y * 2;
      const float s = 0.25f
                      * (sampleNearest(src, sx, sy) + sampleNearest(src, sx + 1, sy)
                         + sampleNearest(src, sx, sy + 1)
                         + sampleNearest(src, sx + 1, sy + 1));
      dst.px[static_cast<std::size_t>(y) * dst.w + x] = s;
    }
  }
  return dst;
}

// Build a coarse-to-fine pyramid: level 0 is the full-resolution image, the last level the
// coarsest. Stops early if a further halving would drop below 4px in either dimension.
std::vector<GrayImage> buildPyramid(GrayImage base, int levels)
{
  std::vector<GrayImage> pyr;
  pyr.reserve(static_cast<std::size_t>(levels));
  pyr.push_back(std::move(base));
  for(int l = 1; l < levels; ++l)
  {
    const GrayImage& prev = pyr.back();
    if(prev.w < 8 || prev.h < 8)
      break;
    pyr.push_back(downsample(prev));
  }
  return pyr;
}
}

void OpticalFlowLK::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  // Current grayscale [0,1].
  GrayImage cur;
  cur.w = W;
  cur.h = H;
  cur.px.resize(static_cast<std::size_t>(W) * H);
  for(int i = 0; i < W * H; ++i)
  {
    const std::uint8_t* p = src.data + static_cast<std::size_t>(i) * 4;
    cur.px[static_cast<std::size_t>(i)]
        = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }

  outputs.flow.value.clear();

  // Need a matching-size previous frame to compute flow.
  if(m_prev.empty() || m_pw != W || m_ph != H)
  {
    m_prev = std::move(cur.px);
    m_pw = W;
    m_ph = H;
    outputs.count = 0;
    return;
  }

  const int grid = std::max(2, inputs.grid.value);
  const int win = std::max(2, inputs.window.value);
  const int levels = std::clamp(inputs.pyramid_levels.value, 1, 4);
  const float invW = 1.f / W;
  const float invH = 1.f / H;
  const float min_mag = inputs.min_mag.value;

  GrayImage prev;
  prev.w = W;
  prev.h = H;
  prev.px = m_prev; // copy: still needed as m_prev after this frame

  // Coarse-to-fine pyramids of both frames. pyr[0] == full res, pyr.back() == coarsest.
  const auto prevPyr = buildPyramid(std::move(prev), levels);
  auto curPyr = buildPyramid(std::move(cur), levels);
  const int nlevels = static_cast<int>(prevPyr.size());

  // Spatial / temporal gradient + LK solve at a given level for a point (in that level's
  // pixel coords) with an incoming flow guess (gx_in, gy_in) in that level's pixels.
  // Returns true and writes the refined displacement (u,v) on success; false if singular.
  auto solveLevel = [&](const GrayImage& p, const GrayImage& c, float px, float py,
                        float gx_in, float gy_in, float& u_out, float& v_out) -> bool {
    float sxx = 0, syy = 0, sxy = 0, sxt = 0, syt = 0;
    for(int dy = -win; dy <= win; ++dy)
    {
      for(int dx = -win; dx <= win; ++dx)
      {
        const float x = px + dx;
        const float y = py + dy;
        // Central-difference spatial gradients on the previous level (bilinear).
        const float ix
            = 0.5f * (sampleBilinear(p, x + 1, y) - sampleBilinear(p, x - 1, y));
        const float iy
            = 0.5f * (sampleBilinear(p, x, y + 1) - sampleBilinear(p, x, y - 1));
        // Temporal: current frame is sampled at the warped location (guess applied).
        const float it
            = sampleBilinear(c, x + gx_in, y + gy_in) - sampleBilinear(p, x, y);
        sxx += ix * ix;
        syy += iy * iy;
        sxy += ix * iy;
        sxt += ix * it;
        syt += iy * it;
      }
    }
    const float det = sxx * syy - sxy * sxy;
    if(std::abs(det) < 1e-6f)
      return false;
    u_out = (-sxt * syy + syt * sxy) / det;
    v_out = (-syt * sxx + sxt * sxy) / det;
    return true;
  };

  for(int gy = 1; gy < grid; ++gy)
  {
    for(int gx = 1; gx < grid; ++gx)
    {
      const int cx = gx * W / grid;
      const int cy = gy * H / grid;

      // Coarse-to-fine: start at the coarsest level with a zero guess, refine downward.
      // Flow accumulator is kept in the current level's pixel units; doubled per level up.
      float u = 0.f, v = 0.f;
      bool ok = false;
      for(int lvl = nlevels - 1; lvl >= 0; --lvl)
      {
        const GrayImage& p = prevPyr[lvl];
        const GrayImage& c = curPyr[lvl];
        // Map the full-res grid point into this level's coordinate frame.
        const float scale = static_cast<float>(p.w) / static_cast<float>(W);
        const float px = cx * scale;
        const float py = cy * scale;

        // Iterate the LK solve a few times at this level so each refinement re-warps the
        // current frame by the updated estimate (Newton-style), letting the displacement
        // converge within the level instead of taking a single half-step.
        constexpr int kIter = 5;
        for(int it = 0; it < kIter; ++it)
        {
          float du = 0.f, dv = 0.f;
          const bool levelOk = solveLevel(p, c, px, py, u, v, du, dv);
          if(!levelOk)
            break;
          ok = true;
          u += du;
          v += dv;
          if(du * du + dv * dv < 1e-4f)
            break;
        }
        // Propagate the estimate to the next finer level (x2 the displacement).
        if(lvl > 0)
        {
          u *= 2.f;
          v *= 2.f;
        }
      }

      // u,v are now full-resolution pixel displacements. Express the velocity in normalised
      // units (divide by W,H) and gate on the normalised magnitude so the threshold behaves
      // the same regardless of the (possibly non-square) frame aspect ratio.
      const float vx = u * invW;
      const float vy = v * invH;
      const float mag = std::sqrt(vx * vx + vy * vy);

      flow_vector fv;
      fv.position = {cx * invW, cy * invH};
      if(ok && mag >= min_mag)
      {
        fv.velocity = {vx, vy};
        fv.magnitude = mag;
        fv.status = 1; // tracked
      }
      else
      {
        // Ill-conditioned (singular at every level) or gated below min_mag: emit a complete
        // grid entry with zero velocity so the caller still sees the point, marked lost.
        fv.velocity = {0.f, 0.f};
        fv.magnitude = 0.f;
        fv.status = 0; // lost
      }
      outputs.flow.value.push_back(fv);
    }
  }

  outputs.count = static_cast<int>(outputs.flow.value.size());
  m_prev = std::move(curPyr[0].px);
}
}
