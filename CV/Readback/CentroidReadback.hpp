#pragma once

#include <halp/buffer.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <cstddef>
#include <cstdint>

namespace cv
{
// Reads the SSBO produced by Centroid.cs ({sumW, sumXW, sumYW, count} as uint32) and emits
// a clean normalised centroid point + mass, instead of the raw integer vector that the
// generic BufferToArray would give. Pairs with CV/Shaders/Analysis/Centroid.cs.
struct CentroidReadback
{
  halp_meta(name, "Centroid readback");
  halp_meta(c_name, "cv_centroid_readback");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Decode a Centroid.cs SSBO into a normalised (x,y) point and mass.");
  halp_meta(uuid, "2e6c9a17-4b3d-4e58-9a1c-7d0f2b6e4c91");

  struct
  {
    halp::cpu_buffer_input<"In"> buffer;
  } inputs;

  struct
  {
    halp::val_port<"Center", halp::xy_type<float>> center;
    halp::val_port<"Mass", float> mass;
    halp::val_port<"Valid", bool> valid;
  } outputs;

  // ---------------------------------------------------------------------------------
  // WIRE CONTRACT with CV/Shaders/Analysis/Centroid.cs. The SSBO "LAYOUT" block there is
  // the single source of truth; these constants mirror it word for word. If you reorder
  // or insert a field in the shader you MUST update this block in the same commit -- the
  // static_asserts below only catch an inconsistent edit *here*, they cannot see the .cs
  // file, so treat the two as one unit.
  //
  //   index : 0     1      2      3      4        5         6
  //   field : sumW  sumXW  sumYW  count  sumWHi   sumXWHi   sumYWHi
  // ---------------------------------------------------------------------------------
  static constexpr std::size_t idx_sumW = 0;
  static constexpr std::size_t idx_sumXW = 1;
  static constexpr std::size_t idx_sumYW = 2;
  static constexpr std::size_t idx_count = 3;
  static constexpr std::size_t idx_hi_base = 4; // hi words follow the 4 low words
  static constexpr std::size_t min_words = 4;   // legacy buffer: low words only
  static constexpr std::size_t full_words = 7;  // low words + the 3 hi words

  static_assert(idx_hi_base == idx_count + 1, "hi words must directly follow the low block");
  static_assert(full_words == idx_hi_base + 3, "one hi word per 64-bit accumulator");
  static_assert(min_words == idx_hi_base, "the legacy prefix is exactly the low block");

  // Must match `const float W_SCALE` in Centroid.cs. It is cv.jit's 0..255 char scale:
  // each pixel contributes round(luma * W_SCALE) to sumW, so sumW is a cv.jit.sum-style
  // raw total and `mass` below divides it back out (cv.jit.mass semantics).
  static constexpr float w_scale = 255.0f;
  static_assert(w_scale == 255.0f, "W_SCALE is the 8-bit char scale; changing it here "
                                   "without changing Centroid.cs silently rescales mass");

  void operator()() noexcept
  {
    auto u = inputs.buffer.cast<std::uint32_t>();
    if(u.size() < min_words)
    {
      outputs.valid = false;
      return;
    }

    // Centroid.cs accumulates each weighted sum as a 64-bit (hi:lo) pair to avoid uint32
    // overflow above ~16.8M px. When the buffer carries the hi words we fold them in
    // (lo + hi*2^32); otherwise (legacy/small buffer) we use the low word only, which is
    // exact below the 16.8M-px ceiling.
    constexpr double k2p32 = 4294967296.0; // 2^32
    const bool hasHi = u.size() >= full_words;
    const double sumW = u[idx_sumW] + (hasHi ? u[idx_hi_base + 0] * k2p32 : 0.0);
    const double sumXW = u[idx_sumXW] + (hasHi ? u[idx_hi_base + 1] * k2p32 : 0.0);
    const double sumYW = u[idx_sumYW] + (hasHi ? u[idx_hi_base + 2] * k2p32 : 0.0);

    if(sumW <= 0.0)
    {
      // No mass: report center -1,-1 like cv.jit.centroids does on an empty image.
      outputs.center.value = {-1.f, -1.f};
      outputs.mass = 0.f;
      outputs.valid = false;
      return;
    }

    outputs.center.value = {
        static_cast<float>(sumXW / sumW), static_cast<float>(sumYW / sumW)};
    outputs.mass = static_cast<float>(sumW / w_scale);
    outputs.valid = true;
  }
};
}
