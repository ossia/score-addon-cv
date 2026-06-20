#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Scanline flood fill from a seed point (cv.jit.floodfill). Fills the connected region of
// pixels whose luminance is within tolerance of the seed pixel, outputs the filled mask and
// the filled pixel count. Sequential region-growing -> Path A.
//
// NOTE: this differs from cv.jit.floodfill's behaviour. cv.jit performs a *binary* fill:
// it grows over pixels that are exactly non-zero (an already-thresholded mask), so connectivity
// is all-or-nothing. Here the fill is *tolerance-based* on continuous luminance: a pixel joins
// the region when |luma(pixel) - luma(seed)| <= tolerance. With tolerance == 0 on a binary
// image the two behave the same; with a non-zero tolerance this fills smoothly-varying regions
// that cv.jit's binary fill would not.
struct FloodFill
{
  halp_meta(name, "Flood fill");
  halp_meta(c_name, "cv_floodfill");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Flood-fill the region around a seed point by luminance similarity.");
  halp_meta(uuid, "7b1c4e89-2a05-4d3f-9e6b-8c0a1f2d5b40");

  struct
  {
    halp::texture_input<"In"> image;
    halp::xy_pad_f32<"Seed", halp::range{0.f, 1.f, 0.5f}> seed; // normalised
    halp::hslider_f32<"Tolerance", halp::range{0.f, 1.f, 0.1f}> tolerance;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
    halp::val_port<"Filled", int> filled;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<std::uint8_t> m_mask;
};
}
