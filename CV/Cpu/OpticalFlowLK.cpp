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
// coarsest. As in OpenCV's buildOpticalFlowPyramid (and cv::PointTracker, which shares this
// rule), the pyramid is truncated as soon as a level would no longer be STRICTLY LARGER than
// the search window: a level smaller than the window is all border, and the garbage coarse
// estimate it produces is then propagated into every finer level. The previous rule here was
// window-independent (`w < 8 || h < 8`), so with Window = 16 -- a 33x33 window -- an 8x8
// level was still accepted.
std::vector<GrayImage> buildPyramid(GrayImage base, int levels, int winSide)
{
  std::vector<GrayImage> pyr;
  pyr.reserve(static_cast<std::size_t>(levels));
  pyr.push_back(std::move(base));
  for(int l = 1; l < levels; ++l)
  {
    const GrayImage& prev = pyr.back();
    const int nw = std::max(1, prev.w / 2);
    const int nh = std::max(1, prev.h / 2);
    if(nw <= winSide || nh <= winSide)
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
  const int side = 2 * win + 1;
  const auto prevPyr = buildPyramid(std::move(prev), levels, side);
  auto curPyr = buildPyramid(std::move(cur), levels, side);
  const int nlevels = static_cast<int>(prevPyr.size());

  // Per-window scratch for the previous-frame gradients and intensities. They depend only on
  // the previous image, so they are constant across the LK iterations at a level.
  std::vector<float> Ix(static_cast<std::size_t>(side) * side);
  std::vector<float> Iy(Ix.size());
  std::vector<float> Ip(Ix.size());

  for(int gy = 1; gy < grid; ++gy)
  {
    for(int gx = 1; gx < grid; ++gx)
    {
      const int cx = gx * W / grid;
      const int cy = gy * H / grid;

      // Coarse-to-fine: start at the coarsest level with a zero guess, refine downward.
      // The accumulator is kept in the CURRENT level's pixel units and rescaled per axis on
      // the way down.
      float u = 0.f, v = 0.f;
      // `lost` follows OpenCV's rule, the same one cv::PointTracker implements: a level that
      // cannot be solved only forfeits ITS refinement; the point is declared lost only when
      // the FINEST level (lvl 0) is the one that fails. The old flag was the opposite -- it
      // was set by any level that solved and never cleared, so a point that was singular at
      // full resolution still reported "tracked" carrying an unrefined coarse estimate.
      bool lost = false;
      for(int lvl = nlevels - 1; lvl >= 0; --lvl)
      {
        const GrayImage& p = prevPyr[lvl];
        const GrayImage& c = curPyr[lvl];
        // Map the full-res grid point into this level's coordinate frame. The x and y ratios
        // must be taken SEPARATELY: downsample() halves each dimension independently with
        // max(1, n/2), so on an odd dimension the two ratios differ (854x480 -> level 2 is
        // 213x120, i.e. 0.24941 in x but 0.25 in y).
        const float rx = static_cast<float>(p.w) / static_cast<float>(W);
        const float ry = static_cast<float>(p.h) / static_cast<float>(H);
        const float lx = cx * rx;
        const float ly = cy * ry;

        // Structure tensor of the previous frame over the window. Accumulated in double:
        // the entries are small in [0,1] grey units and the determinant is a difference of
        // similar magnitudes.
        double A11 = 0., A12 = 0., A22 = 0.;
        for(int dy = -win, i = 0; dy <= win; ++dy)
        {
          for(int dx = -win; dx <= win; ++dx, ++i)
          {
            const float x = lx + dx;
            const float y = ly + dy;
            // Central-difference spatial gradients on the previous level (bilinear).
            const float ix
                = 0.5f * (sampleBilinear(p, x + 1, y) - sampleBilinear(p, x - 1, y));
            const float iy
                = 0.5f * (sampleBilinear(p, x, y + 1) - sampleBilinear(p, x, y - 1));
            Ix[static_cast<std::size_t>(i)] = ix;
            Iy[static_cast<std::size_t>(i)] = iy;
            Ip[static_cast<std::size_t>(i)] = sampleBilinear(p, x, y);
            A11 += ix * ix;
            A12 += ix * iy;
            A22 += iy * iy;
          }
        }

        // Same singularity test, in the same units, as cv::PointTracker: the two objects used
        // to disagree (`|det| < 1e-6` here against `D > 1e-24` there) on the very same
        // quantity, so a window that one called ill-conditioned the other happily solved.
        const double D = A11 * A22 - A12 * A12;
        if(!(D > 1e-24))
        {
          if(lvl == 0)
            lost = true;
        }
        else
        {
          // Iterate the LK solve a few times at this level so each refinement re-warps the
          // current frame by the updated estimate (Newton-style), letting the displacement
          // converge within the level instead of taking a single half-step.
          constexpr int kIter = 5;
          for(int it = 0; it < kIter; ++it)
          {
            double b1 = 0., b2 = 0.;
            for(int dy = -win, i = 0; dy <= win; ++dy)
            {
              for(int dx = -win; dx <= win; ++dx, ++i)
              {
                const float diff = sampleBilinear(c, lx + u + dx, ly + v + dy)
                                   - Ip[static_cast<std::size_t>(i)];
                b1 += Ix[static_cast<std::size_t>(i)] * diff;
                b2 += Iy[static_cast<std::size_t>(i)] * diff;
              }
            }
            const float du = static_cast<float>((-b1 * A22 + b2 * A12) / D);
            const float dv = static_cast<float>((-b2 * A11 + b1 * A12) / D);
            if(!std::isfinite(du) || !std::isfinite(dv))
            {
              if(lvl == 0)
                lost = true;
              break;
            }
            u += du;
            v += dv;
            if(du * du + dv * dv < 1e-4f)
              break;
          }
        }

        // Propagate the estimate to the next finer level. Again per axis, and using the
        // ACTUAL level dimensions rather than a hardcoded x2: on an odd dimension the finer
        // level is not exactly twice the coarser one.
        if(lvl > 0)
        {
          const GrayImage& F = prevPyr[lvl - 1];
          u *= static_cast<float>(F.w) / static_cast<float>(p.w);
          v *= static_cast<float>(F.h) / static_cast<float>(p.h);
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
      if(lost)
      {
        // Ill-conditioned at full resolution: there is no usable measurement at all.
        fv.velocity = {0.f, 0.f};
        fv.magnitude = 0.f;
        fv.status = 0;
        fv.gated = 0; // not applicable: nothing was measured to gate
      }
      else if(mag < min_mag)
      {
        // A perfectly good measurement, deliberately suppressed as "no motion". This used to
        // be reported as status = 0, i.e. indistinguishable from untrackable -- so with the
        // default min_mag every static well-conditioned point read as lost.
        fv.velocity = {0.f, 0.f};
        fv.magnitude = 0.f;
        fv.status = 1;
        fv.gated = 1;
      }
      else
      {
        fv.velocity = {vx, vy};
        fv.magnitude = mag;
        fv.status = 1;
        fv.gated = 0;
      }
      outputs.flow.value.push_back(fv);
    }
  }

  outputs.count = static_cast<int>(outputs.flow.value.size());
  m_prev = std::move(curPyr[0].px);
}
}
