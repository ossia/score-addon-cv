#pragma once

/* score-addon-cv — cv.jit.mean (true cumulative temporal average), OpenCV-free.
 *
 * IMPORTANT: this is NOT the exponential leak implemented by CV/Shaders/Filters/TemporalMean.fs.
 * cv.jit.mean keeps a frame counter and averages *every* frame observed since the last reset
 * with an exact 1/N weight:
 *
 *     index++;  a = 1/index;  b = 1 - a;
 *     state = in*a + state*b;
 *
 * which is the arithmetic mean of all frames, not a leaky integrator. After N frames the output
 * is the exact running mean (an EMA would still be converging towards it and would never settle
 * on the exact value). The `reset` message sets index back to 0; cv.jit deliberately does NOT
 * clear the accumulator, because the very next frame has a = 1 / b = 0 and therefore overwrites
 * it wholesale — we reproduce that exactly.
 *
 * cv.jit keeps the accumulator in a private double buffer at full precision for integer input
 * types and truncates only on output; we do the same (see the rounding note on `quantize`).
 *
 * UPDATE FORM: the accumulator is advanced with the incremental
 *
 *     s += (x - s) * a          which is algebraically  x*a + s*(1 - a)
 *
 * and NOT with the literal product form, for the reason spelled out in
 * CV/Cpu/TemporalStats.hpp (which implements the same cv.jit.mean update): the product form
 * is not a fixed point in binary floating point — with x constant it evaluates x*(a + (1-a))
 * and drifts by ~1 ulp per frame, so a perfectly static pixel never settles exactly. The two
 * headers deliberately use the same form; they used to disagree.
 *
 * ---------------------------------------------------------------------------------------
 * OUTPUT SCALE — the addon-wide r32f contract
 * ---------------------------------------------------------------------------------------
 * The `Mean` r32f output carries the running mean's Rec.601 luma NORMALISED to [0,1], not
 * [0,255]. An r32f texture output must be in [0,1] or score's conversion to the RGBA8 that
 * every texture input expects destroys it — the contract, with the score source reference,
 * is documented once at the top of CV/Cpu/CartoPol.hpp. Multiply by the `Luma scale` value
 * port (255) for the [0,255] luma cv.jit's char path works in. The `Out` RGBA8 image is
 * unaffected: it is already 8-bit and already carries the quantised mean.
 */

#include <CV/Support/EigenImage.hpp>

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cv
{
struct CumulativeMean
{
  halp_meta(name, "Cumulative mean");
  halp_meta(c_name, "cv_cumulative_mean");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "True cumulative temporal average (cv.jit.mean): exact 1/N mean of every frame "
      "since the last reset, not an exponential leak.");
  halp_meta(uuid, "c1a70000-000d-4a00-9000-00000000000d");

  struct
  {
    halp::texture_input<"In"> image;
    // Rising-edge reset, mirroring the `reset` message (same semantics). Provided in
    // addition to the message so the object can be driven from a plain control cable.
    halp::toggle<"Reset"> reset;
  } inputs;

  struct
  {
    // RGBA8 running mean, quantised on output exactly like cv.jit.mean's char path.
    halp::texture_output<"Out"> image;
    // Un-quantised luma of the running mean, NORMALISED to [0,1] per the r32f contract:
    // r32f so nothing is lost to 8-bit rounding inside the object.
    halp::texture_output<"Mean", halp::r32f_texture> mean;
    // Number of frames accumulated since the last reset (cv.jit.mean's `index`).
    halp::val_port<"Frames", int> frames;
    // 255: `Mean * Luma scale` is the [0,255] luma cv.jit reports.
    halp::val_port<"Luma scale", float> luma_scale;
  } outputs;

  // cv.jit.mean's "reset" message: restart the averaging. The accumulator is intentionally
  // left as-is — the next frame runs with a = 1, b = 0 and replaces it entirely.
  void reset() noexcept { m_index = 0; }

  halp_start_messages(CumulativeMean)
    halp_mem_fun(reset)
  halp_end_messages

  void operator()() noexcept
  {
    // Handle the reset edge before the "changed" guard so a reset is never swallowed by a
    // frame that carries no new texture.
    if(inputs.reset.value && !m_was_reset)
    {
      reset();
      m_was_reset = true;
    }
    else if(!inputs.reset.value)
    {
      m_was_reset = false;
    }

    auto& in = inputs.image.texture;
    if(!in.changed || !in.bytes || in.width <= 0 || in.height <= 0)
      return;

    const int W = in.width;
    const int H = in.height;
    const std::size_t N = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);

    // cv.jit reallocates its buffer and zeroes `index` whenever the matrix geometry changes.
    if(W != m_width || H != m_height)
    {
      m_width = W;
      m_height = H;
      m_state.assign(N * 4, 0.0);
      m_index = 0;
    }

    ++m_index;
    const double a = 1.0 / static_cast<double>(m_index);

    outputs.image.create(W, H);
    outputs.mean.create(W, H);

    const auto src = cv_support::as_rgba(in);
    std::uint8_t* dst = outputs.image.texture.bytes;
    float* meanOut = outputs.mean.texture.bytes;

    for(std::size_t i = 0; i < N; ++i)
    {
      const std::uint8_t* p = src.data + i * 4;
      double* s = m_state.data() + i * 4;

      // Incremental form of `in*a + state*(1-a)`; see the UPDATE FORM note in the header.
      for(int c = 0; c < 4; ++c)
        s[c] += (static_cast<double>(p[c]) - s[c]) * a;

      for(int c = 0; c < 4; ++c)
        dst[i * 4 + c] = quantize(s[c]);

      // Normalised to [0,1]: an r32f output carries [0,1] (see the header).
      meanOut[i] = static_cast<float>(
          (0.299 * s[0] + 0.587 * s[1] + 0.114 * s[2]) * (1.0 / 255.0));
    }

    outputs.image.texture.changed = true;
    outputs.mean.texture.changed = true;
    outputs.frames = static_cast<int>(m_index);
    outputs.luma_scale = 255.f;
  }

private:
  // DELIBERATE DEVIATION: cv.jit does `(char)*buf`, i.e. truncation toward zero. Because
  // 1/N is not representable in binary, the exact mean of e.g. {10,20,30} lands on
  // 19.999999999999996 and truncation would report 19 — an off-by-one on the extremely
  // common "the running mean is an exact integer" case. We round to nearest instead, which
  // is what the algorithm intends; the full-precision value is still available on `Mean`.
  [[nodiscard]] static std::uint8_t quantize(double v) noexcept
  {
    return static_cast<std::uint8_t>(std::clamp(v + 0.5, 0.0, 255.0));
  }

  std::vector<double> m_state; // width*height*4, full-precision accumulator
  int m_width{};
  int m_height{};
  long long m_index{}; // cv.jit.mean's frame counter
  bool m_was_reset{};
};
}
