#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Canny edge detector (cv.jit.canny), OpenCV-free.
//
// Port of the *algorithm* behind cv.jit.canny, which forwards to cv::Canny(src, dst, low,
// high, 3). The Max object exposes two attributes -- `threshold` (default 150) and `range`
// (default 10), both clipped to [0,255] -- from which the hysteresis band is derived:
//     low  = floor(clamp(threshold - range, 0, 255))
//     high = floor(clamp(threshold + range, 0, 255))
// The floors are cv::Canny's own, not ours: it does `int low = cvFloor(low_thresh); int
// high = cvFloor(high_thresh);` before comparing anything. Both cv.jit attributes are
// float64, so a band like [150.7, 160.2] is reachable from a patcher and must collapse
// onto [150, 160].
//
// Pipeline (all on Rec.601 luminance, since score textures are RGBA8):
//   1. OPTIONAL 3x3 separable binomial [1 2 1]/4 pre-smooth (sigma ~= 0.85), reflect-101
//      borders. OFF by default: cv::Canny -- which is what cv.jit.canny calls -- does NOT
//      pre-smooth; its only smoothing is Sobel's own [1 2 1] tap. Enabling this lowers
//      gradient magnitudes slightly, so a given threshold becomes less sensitive. Keep it
//      off for cv.jit parity; turn it on to suppress noise on grainy sources.
//   2. 3x3 Sobel gradients gx, gy -- aperture is hardwired to 3, as in cv.jit -- with
//      BORDER_REPLICATE borders, which is what cv::Canny explicitly passes to Sobel
//      (`Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE)`). Note this
//      is NOT reflect-101: under reflect-101 the border column's x-1 and x+1 both land on
//      index 1, so gx would be identically 0 there and an edge one pixel inside the frame
//      would be reported one pixel too far in.
//   3. L1 gradient magnitude |gx| + |gy| (L2gradient = false, cv::Canny's default).
//   4. Non-maximum suppression along the gradient direction, quantised to 4 sectors
//      (horizontal / vertical / both diagonals) using the tan(22.5)/tan(67.5) tests. The
//      two axis-aligned sectors compare `m > previous && m >= next`; the diagonal sector
//      uses TWO STRICT comparisons, as OpenCV does, so a diagonal plateau of equal
//      magnitudes produces no edge pixel at all while a horizontal or vertical one
//      produces exactly one.
//   5. Double-threshold hysteresis: pixels above `high` seed an explicit stack, which
//      propagates through 8-connected neighbours that are above `low`.
//
// Sobel is *not* normalised (no /8), exactly like OpenCV: a full 0->255 step yields a raw
// magnitude of ~765, so the default band [140,160] is quite permissive. This is deliberate --
// it is the numeric behaviour cv.jit users are calibrated against.
//
// Output is a binary 0/255 single-channel (r8) edge map with 1-pixel-thin ridges. Every
// pixel is a candidate, the outermost ring included: NMS runs over the full image against a
// 1-pixel ring of ZERO magnitude, which is exactly OpenCV's arrangement -- its magnitude
// buffer is (rows+2)x(cols+2) with the ring zeroed (`_norm[-1] = _norm[src.cols] = 0`) and
// its NMS loop is `for(int j = 0; j < src.cols; j++)` over rows 0..rows-1. Force-zeroing the
// frame instead would not merely drop a few pixels: combined with replicate borders it
// SHIFTS a near-frame edge by one pixel (an 8x8 with a step at column 1 lights column 0 in
// OpenCV, column 1 if the frame is excluded).
//
// KNOWN RESIDUAL DIVERGENCE (measured, not fixed here). cv.jit.canny is fed a 1-plane *char*
// matrix, so every magnitude OpenCV sees is an integer; we start from RGBA8 and keep the
// Rec.601 luma in float, and 0.299f + 0.587f + 0.114f == 1.0000000075f, so on a gray input
// our magnitudes come out ~7.5e-9 relative too large. That is invisible except when a
// magnitude lands exactly on `low` or `high`, where it flips the comparison. Measured against
// the real cv::Canny on 200 continuous-tone 64x64 gray images: 529 of 819200 pixels differ
// (0.065%), 89355 edge pixels vs OpenCV's 89138. Rounding the luma to an integer -- which is
// what a char plane would have given cv.jit -- makes the two bit-identical (0 differing
// pixels on those 200 images and on 300000 random binary 9x9 ones), but it also quantises the
// input for genuinely coloured sources, so it is left as a deliberate, documented choice
// rather than changed silently.
struct Canny
{
  halp_meta(name, "Canny edges");
  halp_meta(c_name, "cv_canny");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Canny edge detector with hysteresis thresholding.");
  halp_meta(uuid, "c1a70000-0001-4a00-9000-000000000001");

  struct
  {
    halp::texture_input<"In"> image;
    // cv.jit.canny attributes: float64, clipped to [0,255], defaults 150 / 10.
    halp::hslider_f32<"Threshold", halp::range{0.f, 255.f, 150.f}> threshold;
    halp::hslider_f32<"Range", halp::range{0.f, 255.f, 10.f}> range;
    // Not a cv.jit attribute: cv::Canny has no pre-smoothing stage. Off = cv.jit parity.
    halp::toggle<"Pre-smooth"> presmooth;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_gray; // W*H luminance in [0,255], smoothed in place
  std::vector<float> m_tmp;  // W*H separable-pass scratch
  std::vector<float> m_gx;   // W*H horizontal Sobel response
  std::vector<float> m_gy;   // W*H vertical Sobel response
  // (W+2)*(H+2) |gx| + |gy|, with a 1-pixel ring of zeros so NMS can read the neighbours of
  // a frame pixel without a bounds test. Pixel (x, y) lives at (y + 1) * (W + 2) + (x + 1).
  std::vector<float> m_mag;
  std::vector<std::uint8_t> m_state; // W*H: 0 = no, 1 = weak candidate, 2 = edge
  std::vector<std::int32_t> m_stack; // explicit hysteresis stack (no recursion)
};
}
