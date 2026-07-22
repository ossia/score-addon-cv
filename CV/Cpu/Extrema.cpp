#include "Extrema.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cstdint>

namespace cv
{

void Extrema::operator()() noexcept
{
  // Geometry pass-through first: it is independent of the image and must keep flowing on a
  // tick that brings no new frame (otherwise a patch that only moves `Rho` would stall the
  // scalars downstream while the peaks stay valid).
  outputs.theta_step = inputs.theta_step.value;
  outputs.rho_step = inputs.rho_step.value;
  outputs.src_width = inputs.src_width.value;
  outputs.src_height = inputs.src_height.value;

  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  outputs.peaks.value.clear();
  outputs.count = 0;
  outputs.acc_width = W;
  outputs.acc_height = H;

  if(W < 3 || H < 3)
    return;

  // cv.jit clamps the attributes at calc time; do the same so an out-of-range value set
  // programmatically (or by a host that ignores the slider range) cannot misbehave.
  // [0, 255], not cv.jit's [0, 4096]: the accumulator arrives as RGBA8 (see the header).
  const int threshold = std::clamp(inputs.threshold.value, 0, 255);
  const int maxpoints = std::clamp(inputs.maxpoints.value, 1, 4096);
  const bool n8 = (inputs.mode.value == ExtremaNeighbourhood::Neighbours8);

  // Decode the accumulator once into a flat integer buffer. The row stride is the real
  // width (see the header note on the cv.jit stride bug).
  m_acc.resize(static_cast<std::size_t>(W) * H);
  {
    // std::size_t, not int: W * H overflows a signed int past ~46341 square.
    const std::size_t n = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
    for(std::size_t i = 0; i < n; ++i)
    {
      const std::uint8_t* p = src.data + i * 4;
      const float y = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
      m_acc[i] = static_cast<int>(y + 0.5f);
    }
  }

  const int* const a = m_acc.data();
  const int s = W; // row stride, in cells

  // (W-2)*(H-2) in std::size_t: as plain int it overflows on a >46341-square accumulator.
  outputs.peaks.value.reserve(std::min(
      static_cast<std::size_t>(maxpoints),
      static_cast<std::size_t>(W - 2) * static_cast<std::size_t>(H - 2)));

  // Border row/column is never tested, exactly like cv.jit.
  for(int i = 1; i <= H - 2; ++i)
  {
    const int* row = a + static_cast<std::size_t>(i) * s;
    for(int j = 1; j <= W - 2; ++j)
    {
      const int v = row[j];
      if(v <= threshold)
        continue;

      // Strict '>' everywhere: a plateau of equal cells produces no peak.
      bool ok = v > row[j + 1]     // E
                && v > row[j - 1]  // W
                && v > row[j - s]  // N
                && v > row[j + s]; // S
      if(ok && n8)
      {
        ok = v > row[j - s + 1]     // NE
             && v > row[j - s - 1]  // NW
             && v > row[j + s + 1]  // SE
             && v > row[j + s - 1]; // SW
      }
      if(!ok)
        continue;

      outputs.peaks.value.push_back(
          hough_peak{.x = j, .y = i, .value = static_cast<float>(v)});

      // cv.jit bails out of the whole scan as soon as maxpoints is reached: results are
      // first-found in raster order, NOT the strongest peaks.
      if(static_cast<int>(outputs.peaks.value.size()) >= maxpoints)
        goto done;
    }
  }

done:
  outputs.count = static_cast<int>(outputs.peaks.value.size());
}
}
