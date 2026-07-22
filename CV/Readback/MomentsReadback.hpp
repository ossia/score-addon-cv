#pragma once

#include <halp/buffer.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cv
{
// Decodes the Moments.cs SSBO into centroid, mass, orientation, eccentricity and the 7 Hu
// invariants.
//
// The SSBO holds the raw image moments up to 3rd order plus the image size, as uint32, in
// TWO halves: 12 low words {m00,m10,m01,m11,m20,m02,m30,m21,m12,m03,w,h} at indices 0..11
// followed by 10 high words {m00Hi..m03Hi} at indices 12..21, for 22 words total. (The
// earlier comment here said "12 fields", which described only the low half.)
//
// Positions are normalised to [0,1] and weighted by 8-bit luma, so the fixed-point scale
// cancels in every ratio-based quantity (centroid, orientation, eccentricity, Hu).
// `mass` = m00 is reported RAW, i.e. sum(luma*255) in cv.jit's 0..255 char units -- the
// same convention as cv.jit.sum. Divide by 255 on the patch side if you want cv.jit.mass
// units. (See the DESCRIPTION of CV/Shaders/Analysis/Sum.cs for the same distinction.)
//
// `orientation` uses 0.5*atan2(2*mu11, mu20-mu02) in [-pi/2, pi/2]. That is the convention
// CV/Cpu/BlobStats.hpp calls BlobFormula::Normalized, NOT cv.jit.blobs.orientation's
// atan-with-quadrant-fixups in [0, pi). cv.jit.moments itself emits only raw moments and
// has no orientation outlet, so there is no cv.jit value to match here; if you need the
// cv.jit angle convention, feed the moments through BlobStats instead.
struct MomentsReadback
{
  halp_meta(name, "Moments readback");
  halp_meta(c_name, "cv_moments_readback");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Decode Moments.cs into centroid, orientation, eccentricity, Hu.");
  halp_meta(uuid, "9c4e1a2b-7d83-45f0-8b16-3e2c0a9d6f47");

  struct
  {
    halp::cpu_buffer_input<"In"> buffer;
  } inputs;

  struct
  {
    halp::val_port<"Center", halp::xy_type<float>> center;       // normalised
    halp::val_port<"Mass", float> mass;                          // m00 (total weight)
    halp::val_port<"Orientation", float> orientation;            // radians [-pi/2, pi/2]
    halp::val_port<"Eccentricity", float> eccentricity;          // [0,1)
    halp::val_port<"Hu", std::array<float, 7>> hu;
    halp::val_port<"Valid", bool> valid;
  } outputs;

  // ---------------------------------------------------------------------------------
  // WIRE CONTRACT with CV/Shaders/Analysis/Moments.cs. The SSBO "LAYOUT" block there is
  // the single source of truth; these constants mirror it word for word. If you reorder
  // or insert a field in the shader you MUST update this block in the same commit -- the
  // static_asserts below only catch an inconsistent edit *here*, they cannot see the .cs
  // file, so treat the two as one unit.
  //
  //   0    1    2    3    4    5    6    7    8    9    10  11
  //   m00  m10  m01  m11  m20  m02  m30  m21  m12  m03  w   h
  //   12     13     14     15     16     17     18     19     20     21
  //   m00Hi  m10Hi  m01Hi  m11Hi  m20Hi  m02Hi  m30Hi  m21Hi  m12Hi  m03Hi
  // ---------------------------------------------------------------------------------
  static constexpr std::size_t moment_count = 10;  // m00..m03, each a 64-bit (hi:lo) pair
  static constexpr std::size_t idx_w = 10;         // image width  (see note below)
  static constexpr std::size_t idx_h = 11;         // image height (see note below)
  static constexpr std::size_t min_words = 12;     // legacy buffer: the low half only
  static constexpr std::size_t idx_hi_base = 12;   // hi words follow the 12 low words
  static constexpr std::size_t full_words = 22;    // low half + one hi word per moment

  static_assert(idx_w == moment_count, "w/h sit directly after the 10 low moment words");
  static_assert(idx_h == idx_w + 1, "h follows w");
  static_assert(min_words == idx_h + 1, "the legacy prefix is exactly the low half");
  static_assert(idx_hi_base == min_words, "hi words start right after the low half");
  static_assert(full_words == idx_hi_base + moment_count, "one hi word per moment");

  // NOTE on w/h (indices idx_w / idx_h): deliberately NOT consumed. Moments.cs accumulates
  // with positions already normalised to [0,1], so every quantity this readback produces
  // (centroid, orientation, eccentricity, Hu) is resolution-independent and needs no pixel
  // dimensions. They are kept in the layout so a raw BufferToArray dump stays
  // self-describing and so a future pixel-unit variant does not need a wire change. Reading
  // them here would be the bug, not the fix.

  // Must match `const float M3_SCALE` in Moments.cs: the 3rd-order accumulators are scaled
  // by it in the shader so sub-unit x^3/y^3 contributions do not quantise to 0 (Hu[2..6]
  // depend on them); we divide it back out so the central moments and Hu invariants land on
  // the same scale as the lower-order moments.
  static constexpr double m3_scale = 256.0;
  static_assert(m3_scale == 256.0, "M3_SCALE must track Moments.cs; changing it here alone "
                                   "silently rescales every 3rd-order moment and Hu[2..6]");

  void operator()() noexcept
  {
    auto u = inputs.buffer.cast<std::uint32_t>();
    if(u.size() < min_words)
    {
      outputs.valid = false;
      return;
    }

    // Moments.cs carries every raw moment as a 64-bit (hi:lo) pair to avoid uint32 overflow
    // above ~16.8M px. Fold the hi words in when the buffer carries them; otherwise use the
    // low word only (exact below the 16.8M-px ceiling).
    constexpr double k2p32 = 4294967296.0; // 2^32
    const bool hasHi = u.size() >= full_words;
    auto m = [&](std::size_t loIdx) -> double {
      return u[loIdx] + (hasHi ? u[idx_hi_base + loIdx] * k2p32 : 0.0);
    };

    const double m00 = m(0);
    const double m10 = m(1), m01 = m(2);
    const double m11 = m(3), m20 = m(4), m02 = m(5);
    const double m30 = m(6) / m3_scale, m21 = m(7) / m3_scale, m12 = m(8) / m3_scale,
                 m03 = m(9) / m3_scale;

    if(m00 <= 0.0)
    {
      outputs.center.value = {-1.f, -1.f};
      outputs.mass = 0.f;
      outputs.valid = false;
      return;
    }

    // m00 is the 0th moment == total weight (what cv.jit.moments exposes as mass).
    outputs.mass = static_cast<float>(m00);

    const double xb = m10 / m00;
    const double yb = m01 / m00;
    outputs.center.value = {static_cast<float>(xb), static_cast<float>(yb)};

    // Central moments via the standard raw->central transforms.
    const double mu20 = m20 - xb * m10;
    const double mu02 = m02 - yb * m01;
    const double mu11 = m11 - xb * m01;
    const double mu30 = m30 - 3.0 * xb * m20 + 2.0 * xb * xb * m10;
    const double mu03 = m03 - 3.0 * yb * m02 + 2.0 * yb * yb * m01;
    const double mu21 = m21 - 2.0 * xb * m11 - yb * m20 + 2.0 * xb * xb * m01;
    const double mu12 = m12 - 2.0 * yb * m11 - xb * m02 + 2.0 * yb * yb * m10;

    // Orientation + eccentricity from 2nd-order central moments (use mu/m00).
    const double c20 = mu20 / m00, c02 = mu02 / m00, c11 = mu11 / m00;
    outputs.orientation = static_cast<float>(0.5 * std::atan2(2.0 * c11, c20 - c02));
    const double common = std::sqrt(std::max(0.0, (c20 - c02) * (c20 - c02) + 4.0 * c11 * c11));
    const double l1 = 0.5 * (c20 + c02 + common);
    const double l2 = 0.5 * (c20 + c02 - common);
    outputs.eccentricity = (l1 > 1e-12)
        ? static_cast<float>(std::sqrt(std::max(0.0, 1.0 - l2 / l1)))
        : 0.f;

    // Normalised central moments eta_pq = mu_pq / m00^(1 + (p+q)/2).
    auto eta = [&](double mu, int pq) { return mu / std::pow(m00, 1.0 + pq / 2.0); };
    const double n20 = eta(mu20, 2), n02 = eta(mu02, 2), n11 = eta(mu11, 2);
    const double n30 = eta(mu30, 3), n03 = eta(mu03, 3);
    const double n21 = eta(mu21, 3), n12 = eta(mu12, 3);

    // Hu's seven moment invariants.
    const double a = n30 + n12, b = n21 + n03;
    const double a2 = a * a, b2 = b * b;
    std::array<float, 7> hu{};
    hu[0] = static_cast<float>(n20 + n02);
    hu[1] = static_cast<float>((n20 - n02) * (n20 - n02) + 4.0 * n11 * n11);
    hu[2] = static_cast<float>((n30 - 3.0 * n12) * (n30 - 3.0 * n12)
                               + (3.0 * n21 - n03) * (3.0 * n21 - n03));
    hu[3] = static_cast<float>(a2 + b2);
    hu[4] = static_cast<float>(
        (n30 - 3.0 * n12) * a * (a2 - 3.0 * b2)
        + (3.0 * n21 - n03) * b * (3.0 * a2 - b2));
    hu[5] = static_cast<float>(
        (n20 - n02) * (a2 - b2) + 4.0 * n11 * a * b);
    hu[6] = static_cast<float>(
        (3.0 * n21 - n03) * a * (a2 - 3.0 * b2)
        - (n30 - 3.0 * n12) * b * (3.0 * a2 - b2));
    outputs.hu = hu;

    outputs.valid = true;
  }
};
}
