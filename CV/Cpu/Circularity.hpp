#pragma once

/* score-addon-cv — cv.jit.circularity (OpenCV-free).
 *
 * The Max abstraction feeds one binary matrix to both cv.jit.mass (area) and
 * cv.jit.perimeter, then computes
 *
 *      circularity = 4*pi*Area / Perimeter^2
 *
 * (`* 1.` squares the perimeter via a second inlet, `* 12.566376` is 4*pi, `sel 0.` guards
 * the division when the squared perimeter is zero). 1.0 is a perfect disc; thin or ragged
 * shapes score lower. Area and perimeter use exactly the same binarisation, so we compute
 * both in one pass here (CV/Cpu/Perimeter.hpp's shared helper) instead of thresholding twice.
 *
 * Deviations, both deliberate:
 *  - 4*pi is the exact constant, not cv.jit's truncated 12.566376 (a ~1e-6 relative
 *    difference; there is no reason to reproduce the typo).
 *  - Area and Perimeter are exposed as outputs too, so the object is self-contained and you
 *    do not need to run cv.jit.mass / cv.jit.perimeter alongside it to interpret the score.
 *
 * The border rule and threshold are Perimeter's; see CV/Cpu/Perimeter.hpp. Note that with the
 * cv.jit default (`Closed border` off) a shape touching the image edge has an under-counted
 * perimeter and therefore an inflated circularity — that is cv.jit's behaviour as well.
 */

#include <CV/Cpu/Perimeter.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

namespace cv
{
struct Circularity
{
  halp_meta(name, "Circularity");
  halp_meta(c_name, "cv_circularity");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "How disc-like a binary shape is: 4*pi*Area / Perimeter^2.");
  halp_meta(uuid, "c1a70000-000b-4a00-9000-00000000000b");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::toggle<"Closed border"> closed_border;
  } inputs;

  struct
  {
    halp::val_port<"Circularity", float> circularity; // 1.0 = perfect disc
    halp::val_port<"Area", int> area;                 // foreground pixel count
    halp::val_port<"Perimeter", int> perimeter;       // boundary pixel count
  } outputs;

  void operator()() noexcept
  {
    auto& in = inputs.image.texture;
    if(!in.changed || !in.bytes || in.width <= 0 || in.height <= 0)
      return;

    const auto m = shape::measure(
        cv_support::as_rgba(in), shape::threshold_to_u8(inputs.threshold.value),
        inputs.closed_border.value);

    outputs.area = m.area;
    outputs.perimeter = m.perimeter;

    // Guard perimeter^2 == 0 (empty image, or a threshold that kept nothing).
    const double p = static_cast<double>(m.perimeter);
    const double p2 = p * p;
    if(p2 <= 0.)
    {
      outputs.circularity = 0.f;
      return;
    }

    constexpr double four_pi = 4.0 * 3.14159265358979323846;
    outputs.circularity = static_cast<float>(four_pi * static_cast<double>(m.area) / p2);
  }
};
}
