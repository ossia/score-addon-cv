#include "GoodFeatures.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>

namespace cv
{
namespace
{
// Bilinear sample of a single-channel [0,1] image, clamped to the extent.
inline float sampleBilinear(const float* img, int W, int H, float x, float y)
{
  x = std::clamp(x, 0.f, static_cast<float>(W - 1));
  y = std::clamp(y, 0.f, static_cast<float>(H - 1));
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, W - 1);
  const int y1 = std::min(y0 + 1, H - 1);
  const float fx = x - x0;
  const float fy = y - y0;
  const float a = img[static_cast<std::size_t>(y0) * W + x0];
  const float b = img[static_cast<std::size_t>(y0) * W + x1];
  const float c = img[static_cast<std::size_t>(y1) * W + x0];
  const float d = img[static_cast<std::size_t>(y1) * W + x1];
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

// cv::cornerSubPix, winSize = (win, win) in OpenCV's half-size convention (so the actual
// neighbourhood is (2*win+1)^2), zeroZone = (-1,-1) i.e. no excluded centre.
//
// The corner is the point q for which every gradient in the neighbourhood is orthogonal to
// (q - p):  sum_p w(p) * g(p) g(p)^T (q - p) = 0. That 2x2 system is solved and iterated,
// re-centring the neighbourhood on the new estimate each time (this is what makes it
// converge to the intersection of the two edges rather than to the brightest pixel).
void cornerSubPix(
    const float* img, int W, int H, float& cx, float& cy, int win, int maxIter, float eps)
{
  const int side = 2 * win + 1;
  // OpenCV's separable Gaussian-ish mask: exp(-(k/win)^2) on each axis.
  std::vector<float> mask(static_cast<std::size_t>(side) * side);
  for(int i = 0; i < side; ++i)
  {
    const float y = static_cast<float>(i - win) / static_cast<float>(win);
    const float vy = std::exp(-y * y);
    for(int j = 0; j < side; ++j)
    {
      const float x = static_cast<float>(j - win) / static_cast<float>(win);
      mask[static_cast<std::size_t>(i) * side + j] = vy * std::exp(-x * x);
    }
  }

  const float x0 = cx, y0 = cy;
  const double eps2 = static_cast<double>(eps) * eps;
  for(int iter = 0; iter < maxIter; ++iter)
  {
    double a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;
    for(int i = 0; i < side; ++i)
    {
      const float py = static_cast<float>(i - win);
      for(int j = 0; j < side; ++j)
      {
        const float px = static_cast<float>(j - win);
        const float m = mask[static_cast<std::size_t>(i) * side + j];
        const float sx = cx + px;
        const float sy = cy + py;
        const float gx = 0.5f
                         * (sampleBilinear(img, W, H, sx + 1, sy)
                            - sampleBilinear(img, W, H, sx - 1, sy));
        const float gy = 0.5f
                         * (sampleBilinear(img, W, H, sx, sy + 1)
                            - sampleBilinear(img, W, H, sx, sy - 1));
        const double gxx = static_cast<double>(gx) * gx * m;
        const double gxy = static_cast<double>(gx) * gy * m;
        const double gyy = static_cast<double>(gy) * gy * m;
        a += gxx;
        b += gxy;
        c += gyy;
        bb1 += gxx * px + gxy * py;
        bb2 += gxy * px + gyy * py;
      }
    }
    const double det = a * c - b * b;
    if(!(std::abs(det) > 1e-14))
      break;
    const double nx = cx + (c * bb1 - b * bb2) / det;
    const double ny = cy + (a * bb2 - b * bb1) / det;
    if(!std::isfinite(nx) || !std::isfinite(ny))
      break;
    const double dx = nx - cx, dy = ny - cy;
    cx = static_cast<float>(nx);
    cy = static_cast<float>(ny);
    if(cx < 0.f || cy < 0.f || cx > W - 1.f || cy > H - 1.f)
      break;
    if(dx * dx + dy * dy <= eps2)
      break;
  }

  // OpenCV rejects a refinement that wandered further than the window half-size.
  if(std::abs(cx - x0) > static_cast<float>(win) || std::abs(cy - y0) > static_cast<float>(win)
     || !std::isfinite(cx) || !std::isfinite(cy))
  {
    cx = x0;
    cy = y0;
  }
}

struct Cand
{
  int x, y;
  float score;
};
}

void GoodFeatures::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  outputs.features.value.clear();
  outputs.count = 0;

  const int W = in.width;
  const int H = in.height;
  // The 3x3 Sobel needs a 1px border, the 3x3 block sum another one: responses are only
  // defined for 2 <= x < W-2, 2 <= y < H-2. cv.jit refuses dims < 2; we need a bit more.
  if(W < 5 || H < 5)
    return;

  const auto src = cv_support::as_rgba(in);
  const std::size_t N = static_cast<std::size_t>(W) * H;

  m_gray.resize(N);
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t* p = src.data + i * 4;
    m_gray[i] = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }

  auto g = [&](int x, int y) -> float {
    return m_gray[static_cast<std::size_t>(y) * W + x];
  };

  // --- Sobel 3x3 gradients (1/8 scaling keeps the numbers near a true derivative; any
  // uniform scale cancels in the relative quality test).
  m_gx.assign(N, 0.f);
  m_gy.assign(N, 0.f);
  for(int y = 1; y < H - 1; ++y)
  {
    for(int x = 1; x < W - 1; ++x)
    {
      const float gx = (g(x + 1, y - 1) + 2.f * g(x + 1, y) + g(x + 1, y + 1))
                       - (g(x - 1, y - 1) + 2.f * g(x - 1, y) + g(x - 1, y + 1));
      const float gy = (g(x - 1, y + 1) + 2.f * g(x, y + 1) + g(x + 1, y + 1))
                       - (g(x - 1, y - 1) + 2.f * g(x, y - 1) + g(x + 1, y - 1));
      m_gx[static_cast<std::size_t>(y) * W + x] = gx * 0.125f;
      m_gy[static_cast<std::size_t>(y) * W + x] = gy * 0.125f;
    }
  }

  // --- Shi-Tomasi response: minimum eigenvalue of the 3x3-summed structure tensor.
  m_resp.assign(N, 0.f);
  for(int y = 2; y < H - 2; ++y)
  {
    for(int x = 2; x < W - 2; ++x)
    {
      float a = 0.f, b = 0.f, c = 0.f;
      for(int dy = -1; dy <= 1; ++dy)
      {
        for(int dx = -1; dx <= 1; ++dx)
        {
          const std::size_t k = static_cast<std::size_t>(y + dy) * W + (x + dx);
          const float ix = m_gx[k];
          const float iy = m_gy[k];
          a += ix * ix;
          b += ix * iy;
          c += iy * iy;
        }
      }
      const float d = a - c;
      const float lambda_min = 0.5f * ((a + c) - std::sqrt(d * d + 4.f * b * b));
      m_resp[static_cast<std::size_t>(y) * W + x] = std::max(0.f, lambda_min);
    }
  }

  // --- Region of interest (pixels). cv.jit sorts the corners then clips them.
  int rx1 = 0, ry1 = 0, rx2 = W - 1, ry2 = H - 1;
  if(inputs.useroi.value)
  {
    int x1 = inputs.roi_x1.value, y1 = inputs.roi_y1.value;
    int x2 = inputs.roi_x2.value, y2 = inputs.roi_y2.value;
    rx1 = std::clamp(std::min(x1, x2), 0, W - 1);
    ry1 = std::clamp(std::min(y1, y2), 0, H - 1);
    rx2 = std::clamp(std::max(x1, x2), 0, W - 1);
    ry2 = std::clamp(std::max(y1, y2), 0, H - 1);
  }
  const int x_lo = std::max(2, rx1);
  const int x_hi = std::min(W - 3, rx2);
  const int y_lo = std::max(2, ry1);
  const int y_hi = std::min(H - 3, ry2);
  if(x_lo > x_hi || y_lo > y_hi)
    return;

  // --- Strongest response inside the searched region (OpenCV: minMaxLoc with the mask).
  float maxResp = 0.f;
  for(int y = y_lo; y <= y_hi; ++y)
    for(int x = x_lo; x <= x_hi; ++x)
      maxResp = std::max(maxResp, m_resp[static_cast<std::size_t>(y) * W + x]);

  if(!(maxResp > 0.f))
    return; // flat / textureless region: nothing to report

  const float quality = std::clamp(inputs.quality.value, 0.001f, 1.f);
  const float thresh = maxResp * quality;

  // --- Candidates: above the relative threshold AND a 3x3 local maximum (OpenCV compares
  // the response map against its 3x3 dilation, so plateau ties are all kept and later
  // de-clustered by `Distance`).
  std::vector<Cand> cands;
  for(int y = y_lo; y <= y_hi; ++y)
  {
    for(int x = x_lo; x <= x_hi; ++x)
    {
      const float s = m_resp[static_cast<std::size_t>(y) * W + x];
      if(s < thresh)
        continue;
      bool isMax = true;
      for(int dy = -1; dy <= 1 && isMax; ++dy)
        for(int dx = -1; dx <= 1; ++dx)
        {
          if(dx == 0 && dy == 0)
            continue;
          if(m_resp[static_cast<std::size_t>(y + dy) * W + (x + dx)] > s)
          {
            isMax = false;
            break;
          }
        }
      if(isMax)
        cands.push_back({x, y, s});
    }
  }

  // Strongest first; deterministic raster tie-break.
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    if(a.score != b.score)
      return a.score > b.score;
    if(a.y != b.y)
      return a.y < b.y;
    return a.x < b.x;
  });

  // --- Greedy minimum-distance spacing, in pixels (cv.jit forces distance >= 1).
  const float minDist = std::max(1.f, inputs.distance.value);
  const float minDist2 = minDist * minDist;
  const int cap = std::max(1, inputs.max_features.value);

  std::vector<Cand> accepted;
  accepted.reserve(std::min<std::size_t>(cands.size(), static_cast<std::size_t>(cap)));
  for(const auto& c : cands)
  {
    if(static_cast<int>(accepted.size()) >= cap)
      break;
    bool tooClose = false;
    for(const auto& a : accepted)
    {
      const float ddx = static_cast<float>(c.x - a.x);
      const float ddy = static_cast<float>(c.y - a.y);
      if(ddx * ddx + ddy * ddy < minDist2)
      {
        tooClose = true;
        break;
      }
    }
    if(!tooClose)
      accepted.push_back(c);
  }

  // --- Optional sub-pixel refinement. cv.jit guards against images too small for the
  // window (minsize = aperture*2 + 5 with aperture == 3).
  constexpr int kSubPixWin = 3; // OpenCV half-size => 7x7 neighbourhood
  const bool refine
      = inputs.precision.value && W > (kSubPixWin * 2 + 5) && H > (kSubPixWin * 2 + 5);

  const float invW = 1.f / static_cast<float>(W);
  const float invH = 1.f / static_cast<float>(H);
  outputs.features.value.reserve(accepted.size());
  for(const auto& c : accepted)
  {
    float fx = static_cast<float>(c.x);
    float fy = static_cast<float>(c.y);
    if(refine)
      cornerSubPix(m_gray.data(), W, H, fx, fy, kSubPixWin, 10, 0.1f);

    feature_point fp;
    fp.x = fx * invW;
    fp.y = fy * invH;
    fp.score = c.score;
    outputs.features.value.push_back(fp);
  }

  outputs.count = static_cast<int>(outputs.features.value.size());
}
}
