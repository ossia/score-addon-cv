#pragma once

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace cv
{
// One-pole temporal low-pass filter for control values (cv.jit.lowpass, a .maxpat
// abstraction). Smooths an incoming stream of floats or lists of floats, element-wise and
// independently per element, with the cutoff expressed in Hz:
//
//    w = 2*pi * cutoff / rate
//    f = cos(w)
//    b = f - 1 + sqrt((2 - f)^2 - 1)
//    y[n] = b*x[n] + (1 - b)*y[n-1]
//
// The coefficient formula is reproduced exactly from the patch's `calc_coeff` subpatcher
// (`expr -1. + $f1 + sqrt(pow(2 - $f1, 2) - 1.)` fed by `cos(2*pi*cutoff/sr)`), and the
// recurrence from its `vexpr $f1 * $f3 + $f2 * (1.0 - $f3)`.
//
// Deviations from cv.jit.lowpass, all deliberate:
//  * SAMPLE RATE. The Max abstraction measures the sample rate from the wall clock:
//    `cpuclock` differences give the interval between two successive events in ms and
//    `sr = 1000 / max(dt_ms, 0.0001)`. A control-rate Avendish object is driven by the host's
//    tick and has no equivalent clock, so the sample rate is exposed as a "Rate (Hz)" input
//    (the rate at which the object is being ticked), defaulting to 60. Set it to the host's
//    control rate for the cutoff to be correct in Hz.
//  * The cutoff/rate ratio is clamped to [0, 0.5] (Nyquist). Unclamped, the cosine wraps and
//    a cutoff above the sample rate silently makes the filter block everything instead of
//    passing it; the patch has the same flaw but hides it behind a measured sample rate.
//  * `clear` and `reset` are two messages doing the same thing, as in the patch's
//    `route cutoff clear reset done`: both drop the filter memory so the next input restarts
//    from zero.
struct Lowpass
{
  halp_meta(name, "Lowpass (values)");
  halp_meta(c_name, "cv_lowpass");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "One-pole temporal low-pass filter on a value or list of values, cutoff in Hz "
      "relative to the tick rate. Filters each list element independently.");
  halp_meta(uuid, "c1a70000-0004-4a00-9000-000000000004");

  struct
  {
    struct
    {
      halp_meta(name, "In") std::vector<float> value;
    } in;

    halp::hslider_f32<"Cutoff (Hz)", halp::range{0.f, 100.f, 1.f}> cutoff;
    // Host tick rate; stands in for cv.jit.lowpass's clock-measured sample rate.
    halp::hslider_f32<"Rate (Hz)", halp::range{0.001f, 1000.f, 60.f}> rate;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Out") std::vector<float> value;
    } out;
  } outputs;

  // Messages: both drop the filter memory (cv.jit.lowpass's `clear` / `reset`).
  void reset() { m_state.clear(); }
  void clear() { m_state.clear(); }

  halp_start_messages(Lowpass)
    halp_mem_fun(reset)
    halp_mem_fun(clear)
  halp_end_messages

  void operator()() noexcept
  {
    const auto& in = inputs.in.value;
    auto& out = outputs.out.value;

    const std::size_t n = in.size();
    out.resize(n);
    if(n == 0)
      return; // nothing to filter; the memory is left untouched.

    // New elements start from 0, as `pv prev_val` does in the abstraction.
    m_state.resize(n, 0.0);

    const double b = coefficient(inputs.cutoff.value, inputs.rate.value);
    for(std::size_t i = 0; i < n; i++)
    {
      m_state[i] = b * static_cast<double>(in[i]) + (1.0 - b) * m_state[i];
      out[i] = static_cast<float>(m_state[i]);
    }
  }

private:
  static double coefficient(double cutoff, double rate) noexcept
  {
    // `maximum 0.0001` in the patch's ms_interval_to_sr guards the division.
    const double sr = rate > 0.0001 ? rate : 0.0001;
    const double ratio = std::clamp(cutoff / sr, 0.0, 0.5);

    const double w = 2.0 * std::numbers::pi * ratio;
    const double f = std::cos(w);
    const double d = (2.0 - f) * (2.0 - f) - 1.0;
    const double b = f - 1.0 + std::sqrt(d > 0.0 ? d : 0.0);
    return std::clamp(b, 0.0, 1.0);
  }

  std::vector<double> m_state;
};
}
