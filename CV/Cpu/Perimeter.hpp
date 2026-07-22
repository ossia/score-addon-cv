#pragma once

/* score-addon-cv — cv.jit.perimeter (OpenCV-free).
 *
 * The Max abstraction is literally `cv.jit.binedge` -> `cv.jit.mass`:
 *   - cv.jit.binedge marks a foreground pixel white iff at least one of its 8 neighbours is
 *     background;
 *   - cv.jit.mass sums the char matrix and divides by 255, i.e. counts those marked pixels.
 * So the perimeter is the number of boundary pixels of the binary region, over the whole
 * image (all regions pooled — cv.jit.perimeter is not per-blob).
 *
 * BORDER RULE. cv.jit.binedge.cpp handles the image border by simply *not testing* the
 * neighbours that fall off the image (its first/last row and column loops list only the
 * in-image neighbours). Equivalently: off-image neighbours are clamp-to-edge replicas of the
 * pixel itself, hence never background. This is the default here (`Closed border` off), and
 * it means a completely white image has perimeter 0: nothing bounds it.
 *
 * `Closed border` on switches to zero-padding — off-image neighbours count as background, so
 * the image edge bounds the region and a completely white W*H image has perimeter
 * 2*W + 2*H - 4 (its border ring). That is usually what you want when a blob is cropped by
 * the frame, so it is offered as an option, but it is *not* cv.jit behaviour and is off by
 * default.
 */

#include <CV/Support/EigenImage.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cv
{
// Shared binary-shape measurement, used by both Perimeter and Circularity.
namespace shape
{
struct BinaryMeasures
{
  int area{};      // foreground pixel count (cv.jit.mass of the binarised image)
  int perimeter{}; // boundary pixel count  (cv.jit.mass of cv.jit.binedge)
};

// Binarise `src` on Rec.601 luma (foreground iff luma >= thr, matching
// CV/Cpu/ConnectedComponents.hpp) and measure area + boundary-pixel count.
[[nodiscard]] inline BinaryMeasures measure(
    const cv_support::RgbaView& src, std::uint8_t thr, bool closed_border) noexcept
{
  BinaryMeasures out;
  if(!src.valid())
    return out;

  const int W = src.width;
  const int H = src.height;
  const std::size_t N = static_cast<std::size_t>(W) * H;

  // One pass to the binary mask so the 8-neighbour test is a plain lookup.
  std::vector<std::uint8_t> fg(N, 0);
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t* p = src.data + i * 4;
    const float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    if(luma >= static_cast<float>(thr))
    {
      fg[i] = 1;
      ++out.area;
    }
  }

  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      if(!fg[static_cast<std::size_t>(y) * W + x])
        continue;

      bool boundary = false;
      for(int dy = -1; dy <= 1 && !boundary; ++dy)
      {
        for(int dx = -1; dx <= 1; ++dx)
        {
          if(dx == 0 && dy == 0)
            continue;
          const int nx = x + dx;
          const int ny = y + dy;
          if(nx < 0 || ny < 0 || nx >= W || ny >= H)
          {
            // Off-image: cv.jit.binedge does not test it; `closed_border` treats it as
            // background instead.
            if(closed_border)
            {
              boundary = true;
              break;
            }
            continue;
          }
          if(!fg[static_cast<std::size_t>(ny) * W + nx])
          {
            boundary = true;
            break;
          }
        }
      }
      if(boundary)
        ++out.perimeter;
    }
  }
  return out;
}

// The 8-bit luma cutoff a [0,1] threshold control maps to (same rounding as Label).
[[nodiscard]] inline std::uint8_t threshold_to_u8(float t) noexcept
{
  return static_cast<std::uint8_t>(std::clamp(t, 0.f, 1.f) * 255.f + 0.5f);
}
}

struct Perimeter
{
  halp_meta(name, "Perimeter");
  halp_meta(c_name, "cv_perimeter");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Count the boundary pixels of a binarised image (binedge + mass).");
  halp_meta(uuid, "c1a70000-000a-4a00-9000-00000000000a");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    // Off (default) = cv.jit.binedge: off-image neighbours are not tested.
    // On = the image border counts as background.
    halp::toggle<"Closed border"> closed_border;
  } inputs;

  struct
  {
    halp::val_port<"Perimeter", int> perimeter;
  } outputs;

  void operator()() noexcept
  {
    auto& in = inputs.image.texture;
    if(!in.changed || !in.bytes || in.width <= 0 || in.height <= 0)
      return;

    const auto m = shape::measure(
        cv_support::as_rgba(in), shape::threshold_to_u8(inputs.threshold.value),
        inputs.closed_border.value);

    outputs.perimeter = m.perimeter;
  }
};
}
