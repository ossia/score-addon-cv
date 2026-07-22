#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <numbers>
#include <vector>

namespace cv
{
// cv.jit's own degree constant. It is NOT pi/180: 0.01745329252 * 180 = 3.1415926536,
// which is larger than pi = 3.14159265358979. That tiny excess is visible in the derived
// accumulator size -- see `Theta` below -- and is reproduced here on purpose.
inline constexpr double hough_deg = 0.01745329252;

// Clamp bounds for `theta`, in radians. cv.jit clamps to [DEG/2, HALF_PI] using its own
// constants; we use the exact half-degree and right angle instead, which makes the two
// extremes land on round accumulator heights (360 and 2) instead of 359 and 1.
inline constexpr double hough_theta_min = std::numbers::pi_v<double> / 360.0; // 0.5 deg
inline constexpr double hough_theta_max = std::numbers::pi_v<double> / 2.0;   // 90 deg

// cv.jit.hough's constructor default: 5 degrees, expressed with cv.jit's DEG constant.
inline constexpr double hough_theta_default = 5.0 * hough_deg;

// Classical (standard) Hough transform -- the port of cv.jit.hough.
//
// WHY THIS EXISTS ALONGSIDE CV/Shaders/Analysis/Hough.cs: the compute shader hardwires a
// 180 x 256 accumulator and parameterises rho in *normalised* units, so on a non-square
// image the rho axis is stretched and the reported bin no longer corresponds to a real
// distance in pixels. cv.jit exposes genuine `rho` (pixels) and `theta` (radians)
// resolutions and derives the accumulator size from the image; that is what this object
// does. It is also unit-testable, which the shader is not.
//
// ACCUMULATOR GEOMETRY (identical to cv.jit.hough, and to what CV/Cpu/Extrema.hpp +
// CV/Cpu/HoughLines.hpp already expect):
//
//   numangle = (int)(pi / theta)                    -- accumulator HEIGHT (rows = theta)
//   numrho   = (int)(((W + H) * 2 + 1) / rho)       -- accumulator WIDTH  (cols = rho)
//
// and, for every "on" pixel (j = x, i = y) and every angle index n in [0, numangle):
//
//   theta_n = n * theta
//   r       = round((j * cos(theta_n) + i * sin(theta_n)) / rho) + (numrho - 1) / 2
//   accum[n][r]++
//
// Both casts are TRUNCATIONS, not roundings; this is load-bearing. With the cv.jit default
// theta = 5 * 0.01745329252 the quotient pi/theta is 35.999999996, so numangle is 35, not
// the 36 a textbook pi/180 implementation would produce. Do not "fix" this.
//
// VOTING IS BINARY. cv.jit takes a pre-thresholded 1-plane char matrix (typically straight
// out of cv.jit.canny) and tests `*ip != 0`: the pixel VALUE is ignored, every non-zero
// pixel contributes exactly one vote per angle. Score textures are RGBA8, so we binarise on
// Rec.601 luminance with `Threshold` (strict `>`); the default of 0 reproduces `!= 0`
// exactly.
//
// DELIBERATE DEVIATIONS from cv.jit:
//  - cv.jit clamps `rho` and `theta` into local variables at calc time but never writes the
//    clamped values back to the attributes, so what the object reports and what it uses can
//    silently diverge. Our ports are range-limited (so the UI cannot leave the valid band)
//    AND clamped again at calc time, so the two always agree.
//  - cv.jit builds fixed 360-entry sin/cos tables regardless of numangle; the theta clamp is
//    what keeps numangle within them. Our tables are sized to numangle.
//  - cv.jit accumulates the angle (`ang += theta`), which drifts by up to numangle ulps at
//    the last bin. We evaluate n * theta directly.
//  - cv.jit writes accum[n][r] with no bounds check. r cannot normally leave [0, numrho)
//    but the margin depends on integer truncation of numrho; we skip out-of-range bins
//    rather than corrupt memory.
//
// OUTPUT FORMAT -- and why it is r8 and not r32f.
//
// The accumulator is emitted as an **r8** texture, numrho wide x numangle high, one BYTE
// per bin holding min(vote count, 255). Column = rho bin, row = theta bin.
//
// It used to be an r32f texture holding the raw count, and that made the documented
// HoughTransform -> Extrema -> HoughLines chain IMPOSSIBLE to run in score. cv::Extrema
// takes an RGBA8 texture, and score's converter
// (score/src/plugins/score-plugin-avnd/Crousti/TextureConversion.hpp:350-364) treats an
// R32F texel as a NORMALISED [0,1] colour:
//
//     uint8_t gray = qBound(0, int(src[i] * 255.0f), 255);
//
// so a bin with 1 vote and a bin with 64 votes BOTH arrive as 255. The accumulator became a
// saturated plateau and Extrema's strict `>` returned zero peaks, always. R8 -> RGBA8
// (TextureConversion.hpp:257-272) is instead a plain byte copy into r/g/b, and 0..255 is
// exactly the range Extrema's Rec.601 luma decode expects, so the chain now works.
//
// THE 255-VOTE CEILING is the price: a bin that collects more than 255 votes is clamped and
// becomes indistinguishable from any other saturated bin (which, on a plateau, Extrema then
// rejects). `Max count` reports the true, UNCLAMPED maximum of the accumulator so a patch
// can tell that saturation happened and so a downstream threshold can be expressed as a
// fraction of it; the 8-bit ceiling is why cv::Extrema's `Threshold` range is [0, 255].
// One vote is one lit pixel on the line, so a frame reaches the ceiling only with lines
// longer than 255 px; raise `Rho`/`Theta` (fewer, fatter bins do NOT help -- they collect
// more votes) or pre-decimate the edge map if that is a problem in practice.
//
// GEOMETRY OUTPUTS: `Acc width` / `Acc height` mirror the derived accumulator size,
// `Src width` / `Src height` the source frame size, and `Theta step` / `Rho step` the
// EXACT bin steps actually used. The last two matter: cv::HoughLines used to re-derive
// them as pi/accH and (2*(w+h)+1)/accW, which is not the inverse of what this object does
// -- numangle = (int)(pi/theta) TRUNCATES, so at the cv.jit default theta the true step is
// 0.0872665 rad while pi/35 is 0.0897598, a 2.86% error that reaches ~4.9 degrees at the
// top theta bin. Wiring these two outputs into cv::HoughLines removes the error entirely.
struct HoughTransform
{
  halp_meta(name, "Hough transform");
  halp_meta(c_name, "cv_hough");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Classical Hough transform: accumulates (rho, theta) votes for every non-zero "
      "pixel. Feed the accumulator to Extrema, then to Hough lines.");
  halp_meta(uuid, "c1a70000-0045-4a00-9000-000000000001");

  struct
  {
    halp::texture_input<"In"> image;

    // cv.jit attributes are float64; double-valued sliders keep the derived integer sizes
    // exactly reproducible (in float, theta = pi/4 rounds up and numangle becomes 3).
    // rho: distance resolution in PIXELS, clamped to [1, 20].
    halp::hslider_t<double, "Rho", halp::range{1., 20., 4.}> rho;
    // theta: angular resolution in RADIANS, clamped to [0.5 deg, 90 deg].
    halp::hslider_t<
        double, "Theta", halp::range{hough_theta_min, hough_theta_max, hough_theta_default}>
        theta;
    // Not a cv.jit attribute: cv.jit.hough is fed an already-binary matrix. 0 == "any
    // non-zero pixel votes", which is cv.jit's exact behaviour.
    halp::hslider_f32<"Threshold", halp::range{0.f, 255.f, 0.f}> threshold;
  } inputs;

  struct
  {
    // min(votes, 255) per bin -- see the OUTPUT FORMAT note above.
    halp::texture_output<"Accumulator", halp::r8_texture> accum;
    halp::val_port<"Acc width", int> acc_width;   // numrho
    halp::val_port<"Acc height", int> acc_height; // numangle
    // The UNCLAMPED maximum vote count of the frame; > 255 means the texture saturated.
    halp::val_port<"Max count", int> max_count;
    // Exact bin steps in use. Feed these to cv::HoughLines instead of letting it guess.
    halp::val_port<"Theta step", double> theta_step; // radians per accumulator row
    halp::val_port<"Rho step", double> rho_step;     // pixels per accumulator column
    // Source frame size, so cv::HoughLines does not have to be told it a second time.
    halp::val_port<"Src width", int> src_width;
    halp::val_port<"Src height", int> src_height;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<int> m_acc;  // numangle * numrho vote counts, row-major
  std::vector<double> m_sin;
  std::vector<double> m_cos;
};
}
