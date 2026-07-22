#pragma once

/* score-addon-cv — cv.jit.cartopol / cv.jit.poltocar (OpenCV-free).
 *
 * The Max abstractions are trivial jit.op chains over *float32* 1-plane matrices:
 *   cartopol: amp = sqrt(x*x + y*y), phase = atan2(y, x)
 *   poltocar: x = amp*cos(phase),    y = amp*sin(phase)
 * They are used on vector fields (an optical-flow field being the canonical case).
 *
 * =======================================================================================
 * THE r32f CONTRACT — read this before adding any r32f output anywhere in this addon
 * =======================================================================================
 * score converts an `r32f_texture` output into the RGBA8 that every `texture_input` in this
 * addon expects, in score/src/plugins/score-plugin-avnd/Crousti/TextureConversion.hpp
 * (case QRhiTexture::R32F):
 *
 *     uint8_t gray = qBound(0, int(src[i] * 255.0f), 255);
 *
 * It **interprets the float as if it were already in [0,1]** and truncates. Nothing else is
 * possible: a texture is 8 bits per channel by the time it reaches the next object.
 *
 * Therefore, addon-wide:
 *
 *     AN r32f TEXTURE OUTPUT CARRIES A VALUE IN [0,1]. ANY PHYSICAL SCALE NEEDED TO
 *     RECOVER WORLD UNITS IS PUBLISHED ON A SEPARATE VALUE PORT.
 *
 * Writing raw world units to an r32f port is a bug: everything at or above 1.0 arrives as
 * white and everything at or below 0.0 arrives as black. (That is precisely what CartoPol,
 * TemporalStats, CumulativeMean, Label and HornSchunck all used to do.)
 *
 * ---------------------------------------------------------------------------------------
 * THE CODEC — `cv::polar_codec`
 * ---------------------------------------------------------------------------------------
 * One codec, implemented once here, applied on write and undone on read, so producers,
 * consumers and the unit tests all agree. `l` below is the [0,1] channel value: on the way
 * IN it is the Rec.601 luma of an RGBA8 pixel divided by 255; on the way OUT it is the float
 * written to the r32f texture.
 *
 *   - SIGNED quantity (x, y components):  v = (2*l - 1) * Range      [-Range, +Range]
 *     i.e. mid-grey is zero, black is -Range, white is +Range. Quantisation step once it
 *     has been through score's conversion is 2*Range/255. This is the usual "bipolar in an
 *     unsigned texture" convention and is what the flow-field shaders under CV/Shaders use.
 *   - UNSIGNED quantity (amplitude):      a = l * Range              [0, Range]
 *   - PHASE:                              p = (2*l - 1) * pi         [-pi, +pi]
 *     matching atan2's own range, so phase needs no user scaling.
 *
 * `encode_*01` are the exact inverses of `decode_*` and are what an object writes to an
 * r32f output. `encode_*` (returning std::uint8_t) are the same maps quantised to a byte,
 * for tests and for anything that produces an RGBA8 field directly.
 *
 * A `Signed input` toggle (default on) lets CartoPol also read a unipolar field
 * (v = l * Range) for sources that store components in [0,1].
 *
 * ---------------------------------------------------------------------------------------
 * THE sqrt(2)*Range AMPLITUDE CEILING
 * ---------------------------------------------------------------------------------------
 * With both components decoded into [-Range, +Range] (signed mode) or [0, Range]
 * (unsigned), the largest representable amplitude is the diagonal of that square:
 *
 *     amp_max = sqrt(Range^2 + Range^2) = sqrt(2) * Range
 *
 * so THAT, not `Range`, is the divisor CartoPol normalises the amplitude by, and it is what
 * it publishes on `Amplitude scale`. Using `Range` instead would clip every vector more than
 * 45 degrees off an axis. A downstream PolToCar must therefore be given
 * `Range = <CartoPol's Amplitude scale>`, not the CartoPol `Range` — wire the value port.
 */

#include <CV/Support/EigenImage.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cv
{

// The [0,1] <-> world-units codec described above. Public so that tests, and anything that
// wants to feed PolToCar from CartoPol's output, can encode/decode consistently.
namespace polar_codec
{
inline constexpr float pi = 3.14159265358979323846f;

// Rec.601 luma of an RGBA8 pixel, normalised to [0,1].
[[nodiscard]] inline float luma01(const std::uint8_t* p) noexcept
{
  return (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.0f / 255.0f);
}

// ---------------------------------------------------------------- decode: [0,1] -> world
[[nodiscard]] inline float decode_signed(float l01, float range) noexcept
{
  return (2.0f * l01 - 1.0f) * range;
}

[[nodiscard]] inline float decode_unsigned(float l01, float range) noexcept
{
  return l01 * range;
}

[[nodiscard]] inline float decode_phase(float l01) noexcept
{
  return (2.0f * l01 - 1.0f) * pi;
}

// ---------------------------------------------------------------- encode: world -> [0,1]
// Exact inverses of the three above. THESE are what an object writes to an r32f output.
// Out-of-range values clamp (and are therefore lost — pick the scale so they do not occur).
[[nodiscard]] inline float encode_signed01(float v, float range) noexcept
{
  if(!(range > 0.f))
    return 0.5f;
  return std::clamp(0.5f * (v / range) + 0.5f, 0.f, 1.f);
}

[[nodiscard]] inline float encode_unsigned01(float v, float range) noexcept
{
  if(!(range > 0.f))
    return 0.f;
  return std::clamp(v / range, 0.f, 1.f);
}

[[nodiscard]] inline float encode_phase01(float p) noexcept
{
  return std::clamp(0.5f * (p / pi) + 0.5f, 0.f, 1.f);
}

// ------------------------------------------------- the same maps quantised to a byte
// For RGBA8 producers and for tests that need to build an 8-bit input field.
[[nodiscard]] inline std::uint8_t encode_signed(float v, float range) noexcept
{
  if(!(range > 0.f))
    return 128;
  return static_cast<std::uint8_t>(encode_signed01(v, range) * 255.0f + 0.5f);
}

[[nodiscard]] inline std::uint8_t encode_unsigned(float v, float range) noexcept
{
  if(!(range > 0.f))
    return 0;
  return static_cast<std::uint8_t>(encode_unsigned01(v, range) * 255.0f + 0.5f);
}

[[nodiscard]] inline std::uint8_t encode_phase(float p) noexcept
{
  return static_cast<std::uint8_t>(encode_phase01(p) * 255.0f + 0.5f);
}

// The amplitude ceiling of a component pair decoded with `range`: see the block comment.
[[nodiscard]] inline float amplitude_ceiling(float range) noexcept
{
  return 1.41421356237f * range;
}
}

// cv.jit.cartopol: per-pixel cartesian -> polar over a pair of component images.
struct CartoPol
{
  halp_meta(name, "Cartesian to polar");
  halp_meta(c_name, "cv_cartopol");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description, "Per-pixel cartesian to polar over an (x,y) field: amplitude + phase.");
  halp_meta(uuid, "c1a70000-0007-4a00-9000-000000000007");

  struct
  {
    halp::texture_input<"X"> x;
    halp::texture_input<"Y"> y;
    // Component scale: the value a fully-white (or, unsigned off, fully-saturated) pixel
    // stands for.
    halp::hslider_f32<"Range", halp::range{0.0001f, 100.f, 1.f}> range;
    // On: pixels are bipolar, mid-grey = 0 (default, matches the flow-field convention).
    // Off: pixels are unipolar in [0, Range].
    halp::toggle<"Signed input", halp::toggle_setup{.init = true}> is_signed;
  } inputs;

  struct
  {
    // Both r32f outputs obey the addon-wide [0,1] contract (see the top of this file).
    //   amplitude_world = Amplitude * `Amplitude scale`
    //   phase_radians   = (2*Phase - 1) * pi          == polar_codec::decode_phase(Phase)
    halp::texture_output<"Amplitude", halp::r32f_texture> amplitude;
    halp::texture_output<"Phase", halp::r32f_texture> phase;
    // What a fully-white `Amplitude` pixel stands for: sqrt(2) * Range. Wire this into a
    // downstream PolToCar's `Range`.
    halp::val_port<"Amplitude scale", float> amplitude_scale;
  } outputs;

  void operator()() noexcept
  {
    auto& tx = inputs.x.texture;
    auto& ty = inputs.y.texture;
    if(!tx.changed || !tx.bytes || tx.width <= 0 || tx.height <= 0)
      return;
    if(!ty.changed || !ty.bytes || ty.width <= 0 || ty.height <= 0)
      return;
    // The two component fields must describe the same grid; anything else is a patching
    // error, and silently resampling would hide it. Emit nothing.
    if(tx.width != ty.width || tx.height != ty.height)
      return;

    const int W = tx.width;
    const int H = tx.height;
    const auto sx = cv_support::as_rgba(tx);
    const auto sy = cv_support::as_rgba(ty);

    const float range = inputs.range.value;
    const bool sgn = inputs.is_signed.value;
    // The largest amplitude the decoded component square can produce; normalising by it is
    // what keeps `Amplitude` inside [0,1] with nothing clipped.
    const float amp_scale = polar_codec::amplitude_ceiling(range);

    outputs.amplitude.create(W, H);
    outputs.phase.create(W, H);
    float* amp = outputs.amplitude.texture.bytes;
    float* pha = outputs.phase.texture.bytes;

    const std::size_t N = static_cast<std::size_t>(W) * H;
    for(std::size_t i = 0; i < N; ++i)
    {
      const float lx = polar_codec::luma01(sx.data + i * 4);
      const float ly = polar_codec::luma01(sy.data + i * 4);
      const float vx = sgn ? polar_codec::decode_signed(lx, range)
                           : polar_codec::decode_unsigned(lx, range);
      const float vy = sgn ? polar_codec::decode_signed(ly, range)
                           : polar_codec::decode_unsigned(ly, range);

      amp[i] = polar_codec::encode_unsigned01(std::sqrt(vx * vx + vy * vy), amp_scale);
      pha[i] = polar_codec::encode_phase01(std::atan2(vy, vx));
    }

    outputs.amplitude_scale = amp_scale;
    outputs.amplitude.upload();
    outputs.phase.upload();
  }
};
}
