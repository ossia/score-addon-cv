#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// OpenCV's auto-sigma formula (modules/imgproc/src/smooth.dispatch.cpp, getGaussianKernel):
//     sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
// `k` is the full kernel size (radius * 2 + 1), so it gives 0.8 for k = 3, 1.1 for k = 5,
// 1.4 for k = 7, 2.0 for k = 11.
//
// !! It is NOT what sigma == 0 does for small kernels. getGaussianKernel checks the fixed
// `small_gaussian_tab` first and returns it verbatim whenever
//     k is odd  &&  k <= 7  &&  sigma <= 0
// so for k in {1, 3, 5, 7} -- which includes cv.jit.blur's default radius 1 -- this formula
// is never evaluated. It only takes effect for k >= 9 (radius >= 4) with sigma == 0, and for
// an explicitly requested sigma > 0 it is not used either. Kept exposed because tests still
// need to name the formula in order to check that the small kernels do NOT follow it; the
// table itself lives in GaussianBlur.cpp.
[[nodiscard]] constexpr float gaussian_auto_sigma(int k) noexcept
{
  return 0.3f * ((static_cast<float>(k) - 1.f) * 0.5f - 1.f) + 0.8f;
}

// Gaussian blur (cv.jit.blur), OpenCV-free.
//
// Port of the algorithm behind cv.jit.blur, which forwards to
//     cv::GaussianBlur(src, dst, cv::Size(radius*2+1, radius*2+1), sigma, sigma)
// with attributes `radius` (long, default 1) and `sigma` (float, default 0).
//
// A `sigma` of 0 means "let OpenCV pick the kernel" rather than "no blur", and OpenCV picks
// it in two different ways depending on the size:
//   * k in {1, 3, 5, 7}: a hardcoded table of exact dyadic kernels -- {1}, {1,2,1}/4,
//     {1,4,6,4,1}/16, {1,7,14,18,14,7,1}/64 -- returned verbatim, no Gaussian evaluated.
//     This is the branch the object's own default (radius 1, k = 3) lands in.
//   * k >= 9: the auto-sigma formula, see gaussian_auto_sigma() above.
// Both are reproduced. Using the formula for k = 3 instead of the table costs 0.022 on the
// centre tap (0.522011 vs 0.5), which is 5.6 levels out of 255 -- on a 0->255 step the
// pixel just inside the bright side comes out 194 rather than OpenCV's 191.
//
// The 2D Gaussian is separable, so it is applied as a horizontal pass followed by a vertical
// pass (O(k) per pixel instead of O(k^2)). Borders use reflect-101, OpenCV's default:
// gfedcb|abcdefgh|gfedcba.
//
// score textures are RGBA8, so R, G and B are each blurred independently and alpha is passed
// through untouched (blurring alpha would bleed transparency across the image).
struct GaussianBlur
{
  halp_meta(name, "Gaussian blur");
  halp_meta(c_name, "cv_gaussian_blur");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Separable Gaussian blur with an OpenCV-compatible kernel.");
  halp_meta(uuid, "c1a70000-0002-4a00-9000-000000000002");

  struct
  {
    halp::texture_input<"In"> image;
    // cv.jit.blur attributes. radius >= 1 (kernel size = radius*2+1); sigma >= 0, with
    // 0 meaning "auto-derive from the kernel size".
    halp::hslider_i32<"Radius", halp::range{1, 64, 1}> radius;
    halp::hslider_f32<"Sigma", halp::range{0.f, 32.f, 0.f}> sigma;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image; // RGBA8, alpha preserved
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_kernel; // normalised 1D kernel, size 2*radius+1
  std::vector<float> m_tmp;    // W*H*3 horizontal-pass scratch (RGB only)
};
}
