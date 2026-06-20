#pragma once

#include <halp/buffer.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

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

  // Must match Centroid.cs W_SCALE.
  static constexpr float w_scale = 255.0f;

  void operator()() noexcept
  {
    auto u = inputs.buffer.cast<std::uint32_t>();
    if(u.size() < 4)
    {
      outputs.valid = false;
      return;
    }

    // Centroid.cs accumulates each weighted sum as a 64-bit (hi:lo) pair to avoid uint32
    // overflow above ~16.8M px. The low words are at indices 0..2; the matching high words
    // (sumWHi, sumXWHi, sumYWHi) are appended at indices 4..6. When the buffer carries them
    // we fold them in (lo + hi*2^32); otherwise (legacy/small buffer) we use the low word
    // only, which is exact below the 16.8M-px ceiling.
    constexpr double k2p32 = 4294967296.0; // 2^32
    const bool hasHi = u.size() >= 7;
    const double sumW = u[0] + (hasHi ? u[4] * k2p32 : 0.0);
    const double sumXW = u[1] + (hasHi ? u[5] * k2p32 : 0.0);
    const double sumYW = u[2] + (hasHi ? u[6] * k2p32 : 0.0);

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
