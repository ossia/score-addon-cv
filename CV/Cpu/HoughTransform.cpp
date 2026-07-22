#include "HoughTransform.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cv
{
namespace
{
inline constexpr double pi = std::numbers::pi_v<double>;
}

void HoughTransform::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width <= 0 || in.height <= 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  // cv.jit clamps at calc time; we clamp too (and our sliders already limit the range, so
  // reported == effective -- unlike cv.jit, see the header).
  const double rho = std::clamp(inputs.rho.value, 1.0, 20.0);
  const double theta = std::clamp(inputs.theta.value, hough_theta_min, hough_theta_max);

  // Both casts truncate, exactly as in cv.jit. numangle is capped at 360 by the theta
  // clamp and is at least 2 (theta <= pi/2).
  const int numangle = static_cast<int>(pi / theta);
  const int numrho = static_cast<int>(((W + H) * 2 + 1) / rho);

  outputs.acc_width = numrho;
  outputs.acc_height = numangle;
  outputs.src_width = W;
  outputs.src_height = H;
  outputs.theta_step = theta;
  outputs.rho_step = rho;

  if(numangle <= 0 || numrho <= 0)
  {
    // A coarse enough `rho` collapses numrho to 0. Emit an EMPTY texture rather than
    // returning early: leaving the previous frame's accumulator attached while
    // `Acc width` reports 0 makes the two outputs contradict each other, and a downstream
    // object that trusts the texture would keep peaking a stale frame forever.
    outputs.max_count = 0;
    m_acc.clear();
    outputs.accum.create(0, 0);
    outputs.accum.upload();
    return;
  }

  // Sin/cos tables sized to numangle (cv.jit uses fixed 360-entry tables).
  m_sin.resize(static_cast<std::size_t>(numangle));
  m_cos.resize(static_cast<std::size_t>(numangle));
  for(int n = 0; n < numangle; ++n)
  {
    // n * theta, not an accumulated `ang += theta`: same value, no drift.
    const double ang = static_cast<double>(n) * theta;
    m_sin[static_cast<std::size_t>(n)] = std::sin(ang);
    m_cos[static_cast<std::size_t>(n)] = std::cos(ang);
  }

  m_acc.assign(static_cast<std::size_t>(numangle) * static_cast<std::size_t>(numrho), 0);

  const float thr = std::clamp(inputs.threshold.value, 0.f, 255.f);
  const int offset = (numrho - 1) / 2;
  // cv.jit precomputes `irho = 1.0 / rho` and MULTIPLIES; dividing by rho instead differs
  // by an ulp and can flip a round() exactly on a bin boundary. Match cv.jit bit for bit.
  const double irho = 1.0 / rho;

  for(int i = 0; i < H; ++i)
  {
    for(int j = 0; j < W; ++j)
    {
      const std::uint8_t* p
          = src.data + (static_cast<std::size_t>(i) * W + j) * 4;
      const float lum = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
      // BINARY: the value is a yes/no, never a weight. Threshold 0 == cv.jit's `!= 0`.
      if(!(lum > thr))
        continue;

      int* row = m_acc.data();
      for(int n = 0; n < numangle; ++n, row += numrho)
      {
        const double d
            = (static_cast<double>(j) * m_cos[static_cast<std::size_t>(n)]
               + static_cast<double>(i) * m_sin[static_cast<std::size_t>(n)])
              * irho;
        // round() == half away from zero, like cv.jit's round().
        const long r = std::lround(d) + offset;
        if(static_cast<unsigned long>(r) < static_cast<unsigned long>(numrho))
          row[r]++;
      }
    }
  }

  // R8, min(count, 255). See the OUTPUT FORMAT note in the header: an r32f texture holding
  // raw counts is destroyed by score's R32F -> RGBA8 conversion (it reads the float as a
  // normalised [0,1] colour), which broke the whole HoughTransform -> Extrema chain.
  outputs.accum.create(numrho, numangle);
  auto& out = outputs.accum.texture;
  const std::size_t n = m_acc.size();
  int maxc = 0;
  for(std::size_t k = 0; k < n; ++k)
  {
    const int c = m_acc[k];
    if(c > maxc)
      maxc = c;
    out.bytes[k] = static_cast<std::uint8_t>(c < 255 ? c : 255);
  }
  outputs.max_count = maxc; // UNCLAMPED: > 255 tells the patch the texture saturated.
  outputs.accum.upload();
}
}
