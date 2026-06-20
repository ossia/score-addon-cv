#include "CamShift.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>

namespace cv
{
namespace
{
// RGB (0..255) -> HSV. h in [0,1) (hue wheel), s,v in [0,1].
inline void rgb_to_hsv(
    std::uint8_t r8, std::uint8_t g8, std::uint8_t b8, float& h, float& s, float& v) noexcept
{
  const float r = r8 * (1.f / 255.f);
  const float g = g8 * (1.f / 255.f);
  const float b = b8 * (1.f / 255.f);

  const float mx = std::max({r, g, b});
  const float mn = std::min({r, g, b});
  const float d = mx - mn;

  v = mx;
  s = (mx <= 0.f) ? 0.f : (d / mx);

  if(d <= 0.f)
  {
    h = 0.f;
    return;
  }

  float hh;
  if(mx == r)
    hh = (g - b) / d + (g < b ? 6.f : 0.f);
  else if(mx == g)
    hh = (b - r) / d + 2.f;
  else
    hh = (r - g) / d + 4.f;
  h = hh / 6.f; // normalise to [0,1)
}
}

void CamShift::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || !in.width || !in.height)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  const float minSat = std::clamp(inputs.minSat.value, 0.f, 1.f);
  const float minVal = std::clamp(inputs.minVal.value, 0.f, 1.f);

  // ---- Rising edge of Set: (re)build the target hue histogram from a ROI -----------------
  const bool risingSet = inputs.set.value && !m_prevSet;
  m_prevSet = inputs.set.value;

  if(risingSet)
  {
    const float sx = std::clamp(inputs.seed.value.x, 0.f, 1.f);
    const float sy = std::clamp(inputs.seed.value.y, 0.f, 1.f);
    const float frac = std::clamp(inputs.initSize.value, 0.001f, 1.f);

    const int cx = static_cast<int>(sx * (W - 1));
    const int cy = static_cast<int>(sy * (H - 1));
    const int half = std::max(1, static_cast<int>(frac * std::min(W, H) * 0.5f));

    const int x0 = std::clamp(cx - half, 0, W - 1);
    const int y0 = std::clamp(cy - half, 0, H - 1);
    const int x1 = std::clamp(cx + half, 0, W - 1);
    const int y1 = std::clamp(cy + half, 0, H - 1);

    m_hist.fill(0.f);
    double total = 0.0;
    for(int y = y0; y <= y1; ++y)
      for(int x = x0; x <= x1; ++x)
      {
        float h, s, v;
        rgb_to_hsv(src.at(x, y, 0), src.at(x, y, 1), src.at(x, y, 2), h, s, v);
        if(s < minSat || v < minVal)
          continue;
        int bin = static_cast<int>(h * Bins);
        if(bin >= Bins)
          bin = Bins - 1;
        m_hist[static_cast<std::size_t>(bin)] += 1.f;
        total += 1.0;
      }

    if(total > 0.0)
    {
      // Normalise so the peak bin is 1 (typical backprojection scaling).
      float mx = 0.f;
      for(float hv : m_hist)
        mx = std::max(mx, hv);
      if(mx > 0.f)
        for(float& hv : m_hist)
          hv /= mx;
      m_haveModel = true;
    }
    else
    {
      m_haveModel = false;
    }

    // Initialise search window on the seed ROI.
    m_wx = static_cast<float>(x0);
    m_wy = static_cast<float>(y0);
    m_ww = static_cast<float>(x1 - x0 + 1);
    m_wh = static_cast<float>(y1 - y0 + 1);
  }

  if(!m_haveModel)
  {
    outputs.tracking = false;
    outputs.mass = 0.f;
    return;
  }

  const bool meanShiftOnly = (inputs.mode.value == TrackMode::MeanShift);

  // ---- Backprojection sampler: pixel (x,y) -> its hue-bin probability ---------------------
  auto backproject = [&](int x, int y) -> float {
    float h, s, v;
    rgb_to_hsv(src.at(x, y, 0), src.at(x, y, 1), src.at(x, y, 2), h, s, v);
    if(s < minSat || v < minVal)
      return 0.f;
    int bin = static_cast<int>(h * Bins);
    if(bin >= Bins)
      bin = Bins - 1;
    return m_hist[static_cast<std::size_t>(bin)];
  };

  // ---- Mean-shift iterations -------------------------------------------------------------
  const int iters = std::max(1, inputs.iterations.value);
  const float minWin = 4.f;

  // Keep the window sane.
  m_ww = std::clamp(m_ww, minWin, static_cast<float>(W));
  m_wh = std::clamp(m_wh, minWin, static_cast<float>(H));
  m_wx = std::clamp(m_wx, 0.f, std::max(0.f, W - m_ww));
  m_wy = std::clamp(m_wy, 0.f, std::max(0.f, H - m_wh));

  double m00 = 0.0;
  for(int it = 0; it < iters; ++it)
  {
    const int x0 = std::clamp(static_cast<int>(std::floor(m_wx)), 0, W - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(m_wy)), 0, H - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(m_wx + m_ww)) - 1, 0, W - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(m_wy + m_wh)) - 1, 0, H - 1);

    double s00 = 0.0, s10 = 0.0, s01 = 0.0;
    for(int y = y0; y <= y1; ++y)
      for(int x = x0; x <= x1; ++x)
      {
        const float w = backproject(x, y);
        if(w <= 0.f)
          continue;
        s00 += w;
        s10 += static_cast<double>(w) * x;
        s01 += static_cast<double>(w) * y;
      }

    m00 = s00;
    if(s00 <= 0.0)
      break;

    const float ncx = static_cast<float>(s10 / s00);
    const float ncy = static_cast<float>(s01 / s00);
    const float newx = ncx - m_ww * 0.5f;
    const float newy = ncy - m_wh * 0.5f;

    const float dx = newx - m_wx;
    const float dy = newy - m_wy;

    m_wx = std::clamp(newx, 0.f, std::max(0.f, W - m_ww));
    m_wy = std::clamp(newy, 0.f, std::max(0.f, H - m_wh));

    if(std::abs(dx) < 1.f && std::abs(dy) < 1.f)
      break;
  }

  // ---- CamShift: adapt window size from zeroth moment + orientation from 2nd moments -----
  if(m00 <= 0.0)
  {
    outputs.tracking = false;
    outputs.mass = 0.f;
    return;
  }

  // Recompute moments over the converged window to derive size & orientation.
  const int x0 = std::clamp(static_cast<int>(std::floor(m_wx)), 0, W - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(m_wy)), 0, H - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil(m_wx + m_ww)) - 1, 0, W - 1);
  const int y1 = std::clamp(static_cast<int>(std::ceil(m_wy + m_wh)) - 1, 0, H - 1);

  double M00 = 0.0, M10 = 0.0, M01 = 0.0, M20 = 0.0, M02 = 0.0, M11 = 0.0;
  for(int y = y0; y <= y1; ++y)
    for(int x = x0; x <= x1; ++x)
    {
      const float w = backproject(x, y);
      if(w <= 0.f)
        continue;
      const double dw = w;
      M00 += dw;
      M10 += dw * x;
      M01 += dw * y;
      M20 += dw * x * x;
      M02 += dw * y * y;
      M11 += dw * x * y;
    }

  if(M00 <= 0.0)
  {
    outputs.tracking = false;
    outputs.mass = 0.f;
    return;
  }

  const double xc = M10 / M00;
  const double yc = M01 / M00;

  // ---- Mass: zeroth moment normalised by window area, in [0,1]. -------------------------
  // Tracking confidence: ~1 when the tracked colour fills the window, ~0 when absent.
  const double winArea
      = std::max(1.0, (double)(x1 - x0 + 1) * (double)(y1 - y0 + 1));
  const float mass = std::clamp(static_cast<float>(M00 / winArea), 0.f, 1.f);
  outputs.mass = mass;

  const float invW = 1.f / W;
  const float invH = 1.f / H;
  outputs.center.value = {static_cast<float>(xc) * invW, static_cast<float>(yc) * invH};

  if(meanShiftOnly)
  {
    // MeanShift mode (cv.jit.shift mode 0): keep the window size fixed (only the centre
    // moved during the iterations) and force angle = 0 — no size/orientation adaptation.
    m_wx = std::clamp(static_cast<float>(xc) - m_ww * 0.5f, 0.f, std::max(0.f, W - m_ww));
    m_wy = std::clamp(static_cast<float>(yc) - m_wh * 0.5f, 0.f, std::max(0.f, H - m_wh));

    outputs.size.value = {m_ww * invW, m_wh * invH};
    outputs.angle = 0.f;
    outputs.tracking = true;
    return;
  }

  // CamShift mode (cv.jit.shift mode 1): adapt size + orientation from the moments.
  // Central second moments (normalised).
  const double a = M20 / M00 - xc * xc;       // mu20
  const double b = 2.0 * (M11 / M00 - xc * yc); // 2*mu11
  const double c = M02 / M00 - yc * yc;       // mu02

  // Orientation (OpenCV CamShift formula).
  const double theta = 0.5 * std::atan2(b, a - c);

  // Eigenvalues of the covariance -> axis lengths (CamShift width/height estimate).
  const double common = std::sqrt(b * b + (a - c) * (a - c));
  double l1 = (a + c + common) * 0.5; // major variance
  double l2 = (a + c - common) * 0.5; // minor variance
  l1 = std::max(l1, 0.0);
  l2 = std::max(l2, 0.0);
  // length ~ 4*sigma (covers most of the distribution), like OpenCV's 2*2*sqrt.
  const double majAxis = 4.0 * std::sqrt(l1);
  const double minAxis = 4.0 * std::sqrt(l2);

  // Adapt the search window size from the zeroth moment for the next frame.
  // s ~ 2 * sqrt(M00) is the classic CamShift heuristic (M00 in backproj units ~ area).
  const double s = 2.0 * std::sqrt(M00);
  m_ww = std::clamp(static_cast<float>(s), minWin, static_cast<float>(W));
  m_wh = std::clamp(static_cast<float>(s), minWin, static_cast<float>(H));
  m_wx = std::clamp(static_cast<float>(xc) - m_ww * 0.5f, 0.f, std::max(0.f, W - m_ww));
  m_wy = std::clamp(static_cast<float>(yc) - m_wh * 0.5f, 0.f, std::max(0.f, H - m_wh));

  // ---- Output (normalised) ---------------------------------------------------------------
  outputs.size.value
      = {static_cast<float>(majAxis) * invW, static_cast<float>(minAxis) * invH};
  outputs.angle = static_cast<float>(theta);
  outputs.tracking = true;
}
}
