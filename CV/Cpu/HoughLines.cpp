#include "HoughLines.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace cv
{
namespace
{
// M_PI is not portable (MSVC needs _USE_MATH_DEFINES); use std::numbers.
inline constexpr double pi = std::numbers::pi_v<double>;

// Below this |sin(theta)| the line is treated as vertical (see the header note).
// theta is quantised to `Theta step` (>= pi/8192 in practice, and the port floor for an
// explicit step is likewise far above this), so the smallest non-zero |sin| is
// sin(pi/8192) ~ 3.8e-4: only theta bins that really are horizontal-normal (theta ~ 0 or
// ~ pi) can trip it.
inline constexpr double vertical_eps = 1e-9;
}

void HoughLines::operator()() noexcept
{
  outputs.lines.value.clear();
  outputs.count = 0;

  const auto& peaks = inputs.peaks.value;
  if(peaks.empty())
    return;

  const int w = std::max(1, inputs.width.value);
  const int h = std::max(1, inputs.height.value);
  const int accW = std::max(2, inputs.acc_width.value);
  const int accH = std::max(1, inputs.acc_height.value);
  const int threshold = std::clamp(inputs.threshold.value, 0, 255);
  const int maxlines = std::clamp(inputs.maxlines.value, 1, 4096);
  const bool centred = (inputs.rho_origin.value == HoughRhoOrigin::ImageCentre);

  // theta step: take the producer's exact value when it is wired, otherwise invert the
  // accumulator height. The two DIFFER -- cv::HoughTransform derives its height with
  // numangle = (int)(pi/theta), a truncation -- so pi/accH is only a fallback, never the
  // right answer when the producer can be asked. See the header.
  const double explicit_dtheta = inputs.theta_step.value;
  const double dTheta = explicit_dtheta > 0.0 ? explicit_dtheta
                                              : pi / static_cast<double>(accH);

  // rho scale: same story. The corner-origin fallback is cv.jit's (2*(w+h)+1)/accW; the
  // centre-origin one inverts Hough.cs's normalised +/- sqrt(1/2) axis over accW-1 steps.
  const double explicit_drho = inputs.rho_step.value;
  const double rho_scale
      = explicit_drho > 0.0
            ? explicit_drho
            : (centred ? std::numbers::sqrt2_v<double> * static_cast<double>(w)
                             / static_cast<double>(accW - 1)
                       : (2.0 * (static_cast<double>(w) + static_cast<double>(h)) + 1.0)
                             / static_cast<double>(accW));

  // rho offset, in BINS. cv::HoughTransform / cv.jit compute it as the INTEGER
  // (numrho - 1) / 2, so a float (accW-1)/2.0 is half a bin off on every even accW -- ~2 px
  // at the default rho = 4. Hough.cs, by contrast, really does put rho = 0 at the
  // fractional centre 127.5 of its 256 bins.
  const double rho_offset = centred ? (static_cast<double>(accW) - 1.0) / 2.0
                                    : static_cast<double>((accW - 1) / 2);

  // Centre-origin accumulators measure rho from the image centre, which in pixel-INDEX
  // coordinates is ((w-1)/2, (h-1)/2) -- Hough.cs samples at uv = (pos + 0.5)/size, so the
  // half-pixel is already accounted for. Converting to the corner frame the segment
  // equations below use is a per-theta additive term.
  const double cx = (static_cast<double>(w) - 1.0) / 2.0;
  const double cy = (static_cast<double>(h) - 1.0) / 2.0;

  outputs.lines.value.reserve(std::min<std::size_t>(peaks.size(), maxlines));

  for(const auto& p : peaks)
  {
    if(!(p.value > static_cast<float>(threshold)))
      continue;

    const double theta = static_cast<double>(p.y) * dTheta;

    const double st = std::sin(theta);
    const double ct = std::cos(theta);

    // Always in the corner frame from here on: the segment equations below assume
    // rho = x*cos(theta) + y*sin(theta) with x, y measured from the image origin.
    double rho = (static_cast<double>(p.x) - rho_offset) * rho_scale;
    if(centred)
      rho += cx * ct + cy * st;

    line_segment seg{};
    if(std::abs(st) < vertical_eps)
    {
      // Vertical line: rho = x * cos(theta) with |cos(theta)| == 1, so x = rho / cos.
      // The Max abstraction would emit +/-inf here.
      const double x = rho / (ct >= 0.0 ? std::max(ct, vertical_eps)
                                        : std::min(ct, -vertical_eps));
      seg.x1 = static_cast<float>(x);
      seg.y1 = 0.f;
      seg.x2 = static_cast<float>(x);
      seg.y2 = static_cast<float>(h);
    }
    else
    {
      seg.x1 = 0.f;
      seg.y1 = static_cast<float>(rho / st);
      seg.x2 = static_cast<float>(w);
      seg.y2 = static_cast<float>((rho - static_cast<double>(w) * ct) / st);
    }

    outputs.lines.value.push_back(seg);
    if(static_cast<int>(outputs.lines.value.size()) >= maxlines)
      break;
  }

  outputs.count = static_cast<int>(outputs.lines.value.size());
}
}
