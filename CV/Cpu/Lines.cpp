#include "Lines.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace cv
{
namespace
{
inline constexpr double pi = std::numbers::pi_v<double>;

// cv.jit.lines' THRESHOLD_RANGE: the Canny hysteresis band is threshold +/- 10.
inline constexpr float threshold_range = 10.f;

// Fixed-point fraction used to walk along a detected line (OpenCV uses the same 16).
inline constexpr int walk_shift = 16;

// OpenCV's computeNumangle(0, pi, theta): floor + 1, minus one when the last angle would
// duplicate the first (which is the case whenever pi is an exact multiple of theta). Kept
// verbatim because it is what cv::HoughLinesP -- and therefore cv.jit.lines -- uses; note
// that it differs from cv.jit.hough's plain (int)(pi/theta) truncation.
int compute_numangle(double theta) noexcept
{
  int numangle = static_cast<int>(std::floor(pi / theta)) + 1;
  if(numangle > 1 && std::fabs(pi - (numangle - 1) * theta) < theta / 2.0)
    --numangle;
  return numangle;
}
}

void Lines::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width <= 0 || in.height <= 0)
    return;

  outputs.lines.value.clear();
  outputs.count = 0;

  const int W = in.width;
  const int H = in.height;

  // --- 1. Canny, with cv.jit.lines' band ---------------------------------------------------
  // thresh1 = clamp(threshold - 10, 0, 255), thresh2 = clamp(threshold + 10, 0, 255):
  // exactly what cv::Canny(low, high, 3) is given. cv::Canny derives the same pair from
  // (threshold, range), so we hand it range = THRESHOLD_RANGE and let it clamp.
  const double thr = std::clamp(inputs.threshold.value, 1.0, 255.0);
  m_canny.inputs.image.texture = in;
  m_canny.inputs.threshold.value = static_cast<float>(thr);
  m_canny.inputs.range.value = threshold_range;
  m_canny.inputs.presmooth.value = false; // cv::Canny does not pre-smooth
  m_canny();

  const auto& edges = m_canny.outputs.image.texture;
  if(!edges.bytes || edges.width != W || edges.height != H)
    return;

  // --- 2. Probabilistic Hough parameters ---------------------------------------------------
  const int resolution = std::clamp(inputs.resolution.value, 1, 10);
  const double rho = static_cast<double>(resolution);
  // cv.jit: theta = pi / (180 / resolution), i.e. `resolution` degrees per bin.
  const double theta = pi / (180.0 / static_cast<double>(resolution));
  const int threshold = std::clamp(inputs.sensitivity.value, 1, 255);
  // cv::HoughLinesP rounds both to int before use.
  const int line_gap
      = static_cast<int>(std::lround(std::max(0.0, inputs.gap.value)));
  const int line_length
      = static_cast<int>(std::lround(std::max(0.0, inputs.length.value)));
  const int maxlines = std::clamp(inputs.maxlines.value, 1, 4096);

  const int numangle = compute_numangle(theta);
  const int numrho = static_cast<int>(std::lround(((W + H) * 2 + 1) / rho));
  if(numangle <= 0 || numrho <= 0)
    return;

  const double irho = 1.0 / rho;
  m_trig.resize(static_cast<std::size_t>(numangle) * 2u);
  for(int n = 0; n < numangle; ++n)
  {
    const double a = static_cast<double>(n) * theta;
    m_trig[static_cast<std::size_t>(n) * 2u] = std::cos(a) * irho;
    m_trig[static_cast<std::size_t>(n) * 2u + 1u] = std::sin(a) * irho;
  }
  const double* const ttab = m_trig.data();

  m_accum.assign(
      static_cast<std::size_t>(numangle) * static_cast<std::size_t>(numrho), 0);
  int* const accum = m_accum.data();
  const int offset = (numrho - 1) / 2;

  // --- 3. Collect the edge points ----------------------------------------------------------
  const std::size_t N = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
  m_mask.assign(N, 0);
  m_nz.clear();
  for(std::size_t k = 0; k < N; ++k)
  {
    if(edges.bytes[k])
    {
      m_mask[k] = 1;
      m_nz.push_back(static_cast<std::int32_t>(k));
    }
  }

  // Deterministic: reseeded every frame, so the same edge map always yields the same list.
  m_rng.seed(0x9E3779B9u);

  // --- 4. Process the points in random order ------------------------------------------------
  for(int count = static_cast<int>(m_nz.size()); count > 0; --count)
  {
    const int idx
        = std::uniform_int_distribution<int>{0, count - 1}(m_rng);
    const std::int32_t packed = m_nz[static_cast<std::size_t>(idx)];
    // "Remove" the chosen point by overwriting it with the last live one.
    m_nz[static_cast<std::size_t>(idx)] = m_nz[static_cast<std::size_t>(count - 1)];

    const int j = packed % W;
    const int i = packed / W;

    // Already consumed by a previously extracted segment?
    if(!m_mask[static_cast<std::size_t>(packed)])
      continue;

    // Vote for every angle and track the strongest bin. max_val starts at threshold-1, so
    // a line is accepted as soon as some bin reaches `threshold` votes.
    int max_val = threshold - 1;
    int max_n = 0;
    {
      int* adata = accum;
      for(int n = 0; n < numangle; ++n, adata += numrho)
      {
        const long r
            = std::lround(j * ttab[n * 2] + i * ttab[n * 2 + 1]) + offset;
        if(static_cast<unsigned long>(r) >= static_cast<unsigned long>(numrho))
          continue;
        const int val = ++adata[r];
        if(max_val < val)
        {
          max_val = val;
          max_n = n;
        }
      }
    }
    if(max_val < threshold)
      continue;

    // --- 5. Walk both ways along the found line to get the segment endpoints ---------------
    // (a, b) is the line DIRECTION: perpendicular to the (cos, sin) normal.
    const double a = -ttab[max_n * 2 + 1];
    const double b = ttab[max_n * 2];
    int x0 = j;
    int y0 = i;
    int dx0 = 0;
    int dy0 = 0;
    int xflag = 0;
    if(std::fabs(a) > std::fabs(b))
    {
      xflag = 1;
      dx0 = a > 0 ? 1 : -1;
      dy0 = static_cast<int>(
          std::lround(b * (1 << walk_shift) / std::fabs(a)));
      y0 = (y0 << walk_shift) + (1 << (walk_shift - 1));
    }
    else
    {
      xflag = 0;
      dy0 = b > 0 ? 1 : -1;
      dx0 = static_cast<int>(
          std::lround(a * (1 << walk_shift) / std::fabs(b)));
      x0 = (x0 << walk_shift) + (1 << (walk_shift - 1));
    }

    // line_end[0] is reached walking in (dx0, dy0), line_end[1] in the opposite direction.
    // Both are always written: the very first cell visited is the seed point itself, which
    // is by construction still in the mask.
    int end_x[2] = {j, j};
    int end_y[2] = {i, i};

    for(int k = 0; k < 2; ++k)
    {
      int gap = 0;
      int x = x0, y = y0, dx = dx0, dy = dy0;
      if(k > 0)
      {
        dx = -dx;
        dy = -dy;
      }
      for(;; x += dx, y += dy)
      {
        const int j1 = xflag ? x : (x >> walk_shift);
        const int i1 = xflag ? (y >> walk_shift) : y;
        if(j1 < 0 || j1 >= W || i1 < 0 || i1 >= H)
          break;

        if(m_mask[static_cast<std::size_t>(i1) * W + j1])
        {
          gap = 0;
          end_x[k] = j1;
          end_y[k] = i1;
        }
        else if(++gap > line_gap)
        {
          break;
        }
      }
    }

    // Chebyshev length test, exactly as in cv::HoughLinesP.
    const bool good_line = std::abs(end_x[1] - end_x[0]) >= line_length
                           || std::abs(end_y[1] - end_y[0]) >= line_length;

    // --- 6. Re-walk to consume the pixels, un-voting them if the segment is kept -----------
    for(int k = 0; k < 2; ++k)
    {
      int x = x0, y = y0, dx = dx0, dy = dy0;
      if(k > 0)
      {
        dx = -dx;
        dy = -dy;
      }
      for(;; x += dx, y += dy)
      {
        const int j1 = xflag ? x : (x >> walk_shift);
        const int i1 = xflag ? (y >> walk_shift) : y;
        if(j1 < 0 || j1 >= W || i1 < 0 || i1 >= H)
          break;

        auto& m = m_mask[static_cast<std::size_t>(i1) * W + j1];
        if(m)
        {
          if(good_line)
          {
            int* adata = accum;
            for(int n = 0; n < numangle; ++n, adata += numrho)
            {
              const long r
                  = std::lround(j1 * ttab[n * 2] + i1 * ttab[n * 2 + 1]) + offset;
              if(static_cast<unsigned long>(r) < static_cast<unsigned long>(numrho))
                adata[r]--;
            }
          }
          m = 0;
        }

        if(i1 == end_y[k] && j1 == end_x[k])
          break;
      }
    }

    if(good_line)
    {
      outputs.lines.value.push_back(line_segment{
          .x1 = static_cast<float>(end_x[0]),
          .y1 = static_cast<float>(end_y[0]),
          .x2 = static_cast<float>(end_x[1]),
          .y2 = static_cast<float>(end_y[1])});
      if(static_cast<int>(outputs.lines.value.size()) >= maxlines)
        break;
    }
  }

  outputs.count = static_cast<int>(outputs.lines.value.size());
}
}
