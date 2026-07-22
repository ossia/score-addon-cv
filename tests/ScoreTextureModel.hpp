#pragma once

// A faithful model of what score ACTUALLY does when an r32f texture output is wired into a
// texture input.
//
// Why this file exists
// -------------------
// Every `halp::texture_input<...>` in this addon is RGBA8. When a producer declares an
// `r32f_texture` output, score does NOT hand the floats through: it converts, in
// score/src/plugins/score-plugin-avnd/Crousti/TextureConversion.hpp (case QRhiTexture::R32F):
//
//     uint8_t gray = qBound(0, int(src[i] * 255.0f), 255);
//     dst[i*4 + 0..2] = gray;  dst[i*4 + 3] = 255;
//
// i.e. the float is *interpreted as if it were already in [0,1]*, multiplied by 255 and
// TRUNCATED (int(), not a round) into a byte, replicated over R/G/B with opaque alpha.
//
// Consequences a test must respect:
//   * anything <= 0 arrives as 0, anything >= 1 arrives as 255;
//   * a value of 1/255 or more above zero is already visible, and 1.0 saturates;
//   * the quantisation step is 1/255 and the error is a *full* step of downward bias
//     (truncation), not half a step.
//
// This is the reason the addon fixes the contract "an r32f texture output carries [0,1]" and
// publishes any physical scale on a separate value port. A test that hand-encodes its data
// between two objects instead of going through the functions below is not testing the patch
// a user would build — that mistake is exactly what let CartoPol -> PolToCar ship broken.
//
// USE `score_r32f_to_rgba8` / `connect_r32f` FOR EVERY object-to-object texture hop.

#include "TestImage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cvtest
{

// score's exact R32F -> 8-bit grey step: qBound(0, int(v * 255.0f), 255).
// `int(...)` truncates toward zero, so this is a floor for positive values; negatives and
// NaN land on 0, >= 1.0 lands on 255.
inline std::uint8_t score_r32f_to_gray8(float v) noexcept
{
  if(std::isnan(v))
    return 0; // int(NaN) is UB in C++; treat it as the clamp low end.
  const float s = v * 255.0f;
  if(!(s > 0.0f))
    return 0;
  if(s >= 255.0f)
    return 255;
  return static_cast<std::uint8_t>(static_cast<int>(s));
}

// Convert a whole r32f texture output into the RGBA8 image a downstream texture_input would
// really see.
template <typename R32fOutput>
inline Image score_r32f_to_rgba8(const R32fOutput& port)
{
  const auto& t = port.texture;
  Image img(t.width, t.height, 0);
  if(!t.bytes || t.width <= 0 || t.height <= 0)
    return img;

  const float* src = reinterpret_cast<const float*>(t.bytes);
  const std::size_t N = static_cast<std::size_t>(t.width) * static_cast<std::size_t>(t.height);
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t g = score_r32f_to_gray8(src[i]);
    img.px[i * 4 + 0] = g;
    img.px[i * 4 + 1] = g;
    img.px[i * 4 + 2] = g;
    img.px[i * 4 + 3] = 255;
  }
  return img;
}

// The "patch cord": r32f output -> RGBA8 input, through score's real conversion.
// `storage` must outlive the consumer's invocation (the texture input is zero-copy).
template <typename R32fOutput, typename TextureInput>
inline void connect_r32f(const R32fOutput& out, TextureInput& in, Image& storage)
{
  storage = score_r32f_to_rgba8(out);
  feed(in, storage);
}

// Contract check: every r32f value a CV object emits must be inside [0,1], otherwise score's
// conversion above destroys it. Call this on every r32f output of every object under test.
template <typename R32fOutput>
inline bool r32f_in_unit_range(const R32fOutput& port) noexcept
{
  const auto& t = port.texture;
  if(!t.bytes || t.width <= 0 || t.height <= 0)
    return true;
  const float* src = reinterpret_cast<const float*>(t.bytes);
  const std::size_t N = static_cast<std::size_t>(t.width) * static_cast<std::size_t>(t.height);
  for(std::size_t i = 0; i < N; ++i)
    if(!(src[i] >= 0.0f && src[i] <= 1.0f))
      return false;
  return true;
}

} // namespace cvtest
