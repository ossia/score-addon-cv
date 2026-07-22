#pragma once

/* score-addon-cv — cv.jit.poltocar (OpenCV-free).
 *
 * The exact inverse of CV/Cpu/CartoPol.hpp: x = amp*cos(phase), y = amp*sin(phase).
 *
 * The [0,1] r32f contract and the codec (and the reasoning behind both) are documented once,
 * at the top of CartoPol.hpp, which this header pulls in. Both objects apply the SAME codec:
 * CartoPol encodes on write, PolToCar decodes on read, so a CartoPol -> PolToCar chain
 * round-trips through score's r32f -> RGBA8 conversion.
 *
 * WIRING (this is the part that used to be wrong):
 *   CartoPol.Amplitude ---> PolToCar.Amplitude
 *   CartoPol.Phase     ---> PolToCar.Phase
 *   CartoPol."Amplitude scale" (value) ---> PolToCar."Range"
 * `Range` is the amplitude ceiling sqrt(2)*<CartoPol Range>, NOT the CartoPol `Range`.
 * Leaving them equal clips every vector more than 45 degrees off an axis.
 */

#include <CV/Cpu/CartoPol.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cmath>
#include <cstdint>

namespace cv
{
// cv.jit.poltocar: the exact inverse of CartoPol.
struct PolToCar
{
  halp_meta(name, "Polar to cartesian");
  halp_meta(c_name, "cv_poltocar");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description, "Per-pixel polar to cartesian: amplitude + phase back to an (x,y) field.");
  halp_meta(uuid, "c1a70000-0008-4a00-9000-000000000008");

  struct
  {
    halp::texture_input<"Amplitude"> amplitude;
    halp::texture_input<"Phase"> phase;
    // What a fully-white amplitude pixel stands for. Set from CartoPol's `Amplitude scale`
    // output when chaining. The x/y components come out in these units.
    halp::hslider_f32<"Range", halp::range{0.0001f, 100.f, 1.f}> range;
  } inputs;

  struct
  {
    // r32f, [0,1] per the addon contract: the components are SIGNED, so they are carried
    // with the codec's bipolar encoding.
    //   x_world = (2*X - 1) * `Component scale`   == polar_codec::decode_signed(X, scale)
    // Feeding these straight back into a CartoPol with `Signed input` on and
    // `Range` = `Component scale` closes the loop exactly.
    halp::texture_output<"X", halp::r32f_texture> x;
    halp::texture_output<"Y", halp::r32f_texture> y;
    // The full component swing, == `Range`: a component can never exceed the amplitude.
    halp::val_port<"Component scale", float> component_scale;
  } outputs;

  void operator()() noexcept
  {
    auto& ta = inputs.amplitude.texture;
    auto& tp = inputs.phase.texture;
    if(!ta.changed || !ta.bytes || ta.width <= 0 || ta.height <= 0)
      return;
    if(!tp.changed || !tp.bytes || tp.width <= 0 || tp.height <= 0)
      return;
    if(ta.width != tp.width || ta.height != tp.height)
      return;

    const int W = ta.width;
    const int H = ta.height;
    const auto sa = cv_support::as_rgba(ta);
    const auto sp = cv_support::as_rgba(tp);

    const float range = inputs.range.value;

    outputs.x.create(W, H);
    outputs.y.create(W, H);
    float* ox = outputs.x.texture.bytes;
    float* oy = outputs.y.texture.bytes;

    const std::size_t N = static_cast<std::size_t>(W) * H;
    for(std::size_t i = 0; i < N; ++i)
    {
      const float a
          = polar_codec::decode_unsigned(polar_codec::luma01(sa.data + i * 4), range);
      const float p = polar_codec::decode_phase(polar_codec::luma01(sp.data + i * 4));

      // |a*cos|, |a*sin| <= a <= range, so nothing is ever clipped by this encoding.
      ox[i] = polar_codec::encode_signed01(a * std::cos(p), range);
      oy[i] = polar_codec::encode_signed01(a * std::sin(p), range);
    }

    outputs.component_scale = range;
    outputs.x.upload();
    outputs.y.upload();
  }
};
}
