#include "PointTracker.hpp"

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

// 2x2 box-downsample producing the next coarser pyramid level.
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
      dst.px[static_cast<std::size_t>(y) * dst.w + x]
          = 0.25f
            * (sampleNearest(src, sx, sy) + sampleNearest(src, sx + 1, sy)
               + sampleNearest(src, sx, sy + 1) + sampleNearest(src, sx + 1, sy + 1));
    }
  }
  return dst;
}

// pyr[0] == full resolution, pyr.back() == coarsest. As in OpenCV's
// buildOpticalFlowPyramid, the pyramid is truncated as soon as a level would no longer be
// strictly larger than the search window -- a level smaller than the window is all border
// and produces a garbage coarse estimate that the finer levels then cannot recover from.
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

// Unit conversion for `minEigThreshold`. OpenCV evaluates the minimum eigenvalue of the
// windowed structure tensor built from SCHARR derivatives of 8-BIT grey; we build ours from
// central differences of [0,1] grey. For a locally linear patch with gradient g per pixel,
// OpenCV's Scharr operator returns 32*g_8bit == 32*255*g_[0,1] where our central difference
// returns g_[0,1]. The eigenvalue is quadratic in the gradient, hence a (32*255)^2 factor.
//
// BUT OpenCV does not test the raw accumulation: lkpyramid.cpp rescales the structure tensor
// by FLT_SCALE = 1/(1<<20) before forming the eigenvalue --
//     A11 = iA11*FLT_SCALE; ...; minEig = (A22+A11 - sqrt(...)) / (2*win.w*win.h);
//     if(minEig < minEigThreshold || D < FLT_EPSILON) ...
// so the conversion factor is (32*255)^2 / 2^20 = 63.5009765625, not (32*255)^2. Getting this
// wrong by the missing 2^20 made the reported eigenvalue ~10^6x too large and the gate
// unreachable over the whole "Min eigenvalue" slider range. An independent empirical
// calibration (Scharr + FLT_SCALE on 8-bit vs central differences on [0,1], 15x15 window)
// measures 61.6, consistent with the algebraic 63.5.
//
// With the right factor OpenCV's documented default of 1e-4 -- which in practice only rejects
// windows with essentially no texture at all -- keeps exactly that meaning here.
constexpr float kOpenCVEigScale = (255.f * 32.f) * (255.f * 32.f) / 1048576.f; // 2^20

bool sameSeeds(const std::vector<feature_point>& a, const std::vector<feature_point>& b)
{
  if(a.size() != b.size())
    return false;
  for(std::size_t i = 0; i < a.size(); ++i)
    if(a[i].x != b[i].x || a[i].y != b[i].y)
      return false;
  return true;
}
}

void PointTracker::set(int index, float x, float y)
{
  if(index < 0 || index > 8191)
    return;
  if(static_cast<int>(m_pts.size()) <= index)
    m_pts.resize(static_cast<std::size_t>(index) + 1);
  auto& p = m_pts[static_cast<std::size_t>(index)];
  p.x = x;
  p.y = y;
  p.status = 1;
  p.error = 0.f;
  m_min_n = std::max(m_min_n, index + 1);
}

void PointTracker::reset()
{
  m_pts.clear();
  m_last_seeds.clear();
  m_prev.clear();
  m_pw = m_ph = 0;
  m_min_n = 0;
  m_have_prev = false;
  outputs.points.value.clear();
  outputs.count = 0;
}

void PointTracker::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);
  const std::size_t NPX = static_cast<std::size_t>(W) * H;

  GrayImage cur;
  cur.w = W;
  cur.h = H;
  cur.px.resize(NPX);
  for(std::size_t i = 0; i < NPX; ++i)
  {
    const std::uint8_t* p = src.data + i * 4;
    cur.px[i] = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }

  // --- Seeding. The seed list re-initialises the whole track set, but only when it
  // actually changes: while it is stable the tracks keep accumulating.
  const auto& seeds = inputs.seeds.value;
  if(!sameSeeds(seeds, m_last_seeds))
  {
    m_last_seeds = seeds;
    if(!seeds.empty())
    {
      m_pts.assign(seeds.size(), tracked_point{});
      for(std::size_t i = 0; i < seeds.size(); ++i)
      {
        m_pts[i].x = seeds[i].x;
        m_pts[i].y = seeds[i].y;
        m_pts[i].status = 1;
        m_pts[i].error = 0.f;
      }
    }
  }

  // Track count: the seed list wins when non-empty, else the control (never below the
  // highest index reached through `set`).
  const int N = seeds.empty() ? std::max(inputs.npoints.value, m_min_n)
                              : static_cast<int>(seeds.size());
  if(static_cast<int>(m_pts.size()) != N)
    m_pts.resize(static_cast<std::size_t>(N)); // new tracks start at (0,0), status 0

  // --- No usable reference frame: no motion this tick. The seeded positions are kept
  // (unlike cv.jit, which swaps in its uninitialised buffer here).
  if(!m_have_prev || m_pw != W || m_ph != H)
  {
    m_prev = cur.px;
    m_pw = W;
    m_ph = H;
    m_have_prev = true;
    for(auto& p : m_pts)
      p.error = 0.f;
    outputs.points.value = m_pts;
    outputs.count = N;
    return;
  }

  const int win = std::clamp(inputs.radius.value, 1, 32);
  const int levels = std::clamp(inputs.max_level.value, 0, 5) + 1;
  const int maxIter = std::clamp(inputs.max_iter.value, 1, 100);
  const float eps2 = inputs.epsilon.value * inputs.epsilon.value;
  const float minEigThr = inputs.min_eig.value;
  const float winArea = static_cast<float>((2 * win + 1) * (2 * win + 1));

  GrayImage prev;
  prev.w = W;
  prev.h = H;
  prev.px = m_prev;

  // Scratch for the per-level window gradients (constant across the LK iterations).
  const int side = 2 * win + 1;

  const auto prevPyr = buildPyramid(std::move(prev), levels, side);
  const auto curPyr = buildPyramid(std::move(cur), levels, side);
  const int nlevels = static_cast<int>(prevPyr.size());

  std::vector<float> Ix(static_cast<std::size_t>(side) * side);
  std::vector<float> Iy(Ix.size());
  std::vector<float> Ip(Ix.size());

  for(auto& pt : m_pts)
  {
    // Start from this track's last known position: this is the persistence.
    const float px0 = pt.x * static_cast<float>(W);
    const float py0 = pt.y * static_cast<float>(H);

    float gx = 0.f, gy = 0.f; // flow guess, in the current level's pixels
    bool lost = false;
    float residual = 0.f;

    for(int lvl = nlevels - 1; lvl >= 0; --lvl)
    {
      const GrayImage& P = prevPyr[static_cast<std::size_t>(lvl)];
      const GrayImage& C = curPyr[static_cast<std::size_t>(lvl)];
      const float sx = static_cast<float>(P.w) / static_cast<float>(W);
      const float sy = static_cast<float>(P.h) / static_cast<float>(H);
      const float lx = px0 * sx;
      const float ly = py0 * sy;

      // Structure tensor of the previous frame over the window (constant per level).
      // Accumulated in double: the entries are small in [0,1] grey units and the
      // determinant is a difference of similar magnitudes.
      double A11 = 0., A12 = 0., A22 = 0.;
      for(int dy = -win, i = 0; dy <= win; ++dy)
      {
        for(int dx = -win; dx <= win; ++dx, ++i)
        {
          const float x = lx + dx;
          const float y = ly + dy;
          const float ix
              = 0.5f * (sampleBilinear(P, x + 1, y) - sampleBilinear(P, x - 1, y));
          const float iy
              = 0.5f * (sampleBilinear(P, x, y + 1) - sampleBilinear(P, x, y - 1));
          Ix[static_cast<std::size_t>(i)] = ix;
          Iy[static_cast<std::size_t>(i)] = iy;
          Ip[static_cast<std::size_t>(i)] = sampleBilinear(P, x, y);
          A11 += ix * ix;
          A12 += ix * iy;
          A22 += iy * iy;
        }
      }

      const double D = A11 * A22 - A12 * A12;
      const double tr = A11 + A22;
      const double df = A11 - A22;
      const double minEig = (tr - std::sqrt(df * df + 4. * A12 * A12)) / (2. * winArea)
                            * static_cast<double>(kOpenCVEigScale);

      // OpenCV: too little texture / singular system => no refinement at this level; the
      // point is only declared lost when this happens at the finest level.
      if(minEig < minEigThr || !(D > 1e-24))
      {
        if(lvl == 0)
          lost = true;
      }
      else
      {
        for(int it = 0; it < maxIter; ++it)
        {
          double b1 = 0., b2 = 0.;
          for(int dy = -win, i = 0; dy <= win; ++dy)
          {
            for(int dx = -win; dx <= win; ++dx, ++i)
            {
              const float diff
                  = sampleBilinear(C, lx + gx + dx, ly + gy + dy) - Ip[static_cast<std::size_t>(i)];
              b1 += Ix[static_cast<std::size_t>(i)] * diff;
              b2 += Iy[static_cast<std::size_t>(i)] * diff;
            }
          }
          const float u = static_cast<float>((-b1 * A22 + b2 * A12) / D);
          const float v = static_cast<float>((-b2 * A11 + b1 * A12) / D);
          if(!std::isfinite(u) || !std::isfinite(v))
          {
            if(lvl == 0)
              lost = true;
            break;
          }
          gx += u;
          gy += v;
          if(u * u + v * v <= eps2)
            break;
        }

        if(lvl == 0)
        {
          // OpenCV's `err`: mean absolute photometric residual over the window.
          float acc = 0.f;
          for(int dy = -win, i = 0; dy <= win; ++dy)
            for(int dx = -win; dx <= win; ++dx, ++i)
              acc += std::abs(
                  sampleBilinear(C, lx + gx + dx, ly + gy + dy)
                  - Ip[static_cast<std::size_t>(i)]);
          residual = acc / winArea;
        }
      }

      // Propagate the estimate to the next finer level.
      if(lvl > 0)
      {
        const GrayImage& F = prevPyr[static_cast<std::size_t>(lvl - 1)];
        gx *= static_cast<float>(F.w) / static_cast<float>(P.w);
        gy *= static_cast<float>(F.h) / static_cast<float>(P.h);
      }
    }

    const float nx = px0 + gx;
    const float ny = py0 + gy;
    const bool outside = !(nx >= 0.f && ny >= 0.f && nx <= static_cast<float>(W - 1)
                           && ny <= static_cast<float>(H - 1))
                         || !std::isfinite(nx) || !std::isfinite(ny);

    if(lost || outside)
    {
      // Lost: keep the last coordinates. They are also next frame's starting guess, so the
      // point is retried from there (documented cv.jit behaviour).
      pt.status = 0;
      pt.error = 0.f;
    }
    else
    {
      pt.x = nx / static_cast<float>(W);
      pt.y = ny / static_cast<float>(H);
      pt.status = 1;
      pt.error = residual;
    }
  }

  outputs.points.value = m_pts;
  outputs.count = N;

  // cv.jit: std::swap(previous_points, points) -- here the update is in place, m_pts *is*
  // the next frame's starting point set.
  m_prev = curPyr[0].px;
  m_pw = W;
  m_ph = H;
}
}
