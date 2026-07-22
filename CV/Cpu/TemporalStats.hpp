#pragma once

/* score-addon-cv — cv.jit.variance / cv.jit.stddev (per-pixel *temporal* statistics).
 *
 * Both cv.jit objects are abstractions built out of two cv.jit.mean instances
 * (cv.jit/patchers/cv.jit.variance.maxpat, cv.jit.stddev.maxpat). Reading the patch cords:
 *
 *     input -> [t l l]
 *       right outlet fires FIRST  -> [cv.jit.mean] -> mu           (outlet: mean)
 *       left  outlet fires SECOND -> [jit.op @op -] with mu on the right inlet -> d = x - mu
 *       d -> [jit.op @op *] with d on both inlets -> d*d
 *       d*d -> [cv.jit.mean] -> var                                 (outlet: variance)
 *     cv.jit.stddev is the same graph plus [jit.op @op sqrt] on the variance.
 *
 * Two consequences worth stating, because they are easy to get wrong:
 *  - the mean is updated with the current frame *before* the deviation is taken, so
 *    d = x_N - mean(x_1..x_N), not x_N - mean(x_1..x_{N-1});
 *  - the outer expectation is again cv.jit.mean's exact 1/N cumulative average, so this is a
 *    running (biased, 1/N-normalised) variance over every frame since reset, not an EMA.
 *
 * NOTE on outlet order in the originals: cv.jit.stddev's outlet 0 is the *standard deviation*
 * and outlet 1 is the *mean* — the reverse of the visual top-to-bottom reading of the patch.
 * cv.jit.variance is outlet 0 = mean, outlet 1 = variance. Here the ports are named, so the
 * ambiguity disappears; we declare them Mean / Variance / StdDev and emit all three from one
 * object so it is self-contained.
 *
 * The `reset` message clears both accumulators and the frame counter.
 *
 * ---------------------------------------------------------------------------------------
 * OUTPUT SCALE — the addon-wide r32f contract
 * ---------------------------------------------------------------------------------------
 * An r32f texture output carries [0,1]; score converts it to the RGBA8 every texture input
 * expects by interpreting the float as if it already were in that range (the contract, with
 * the score source reference, is documented once at the top of CV/Cpu/CartoPol.hpp).
 * This object used to emit [0,255] luma units, which meant that any pixel brighter than
 * luma 1/255 arrived at the next object as pure white and every variance arrived saturated.
 *
 * So the frame is reduced to Rec.601 luma NORMALISED to [0,1] and every statistic is
 * computed in those units:
 *
 *     Mean      in [0, 1]       Variance  in [0, 0.25]      StdDev  in [0, 0.5]
 *
 * (the extremes belong to a pixel alternating between pure black and pure white). All three
 * are therefore inside the contract by construction, with nothing clipped.
 *
 * To get back to the [0,255] scale cv.jit's char path works in — the one hand-computed
 * cv.jit numbers are quoted in — multiply by the `Luma scale` value port (255):
 *
 *     mean_255     = Mean     * Luma scale
 *     variance_255 = Variance * Luma scale^2      (a variance scales quadratically)
 *     stddev_255   = StdDev   * Luma scale
 */

#include <CV/Support/EigenImage.hpp>

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cv
{
struct TemporalStats
{
  halp_meta(name, "Temporal statistics");
  halp_meta(c_name, "cv_temporal_stats");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Per-pixel temporal mean / variance / standard deviation over every frame since "
      "the last reset (cv.jit.variance, cv.jit.stddev).");
  halp_meta(uuid, "c1a70000-000c-4a00-9000-00000000000c");

  struct
  {
    halp::texture_input<"In"> image;
    // Rising-edge equivalent of the `reset` message.
    halp::toggle<"Reset"> reset;
  } inputs;

  struct
  {
    // r32f throughout so nothing is re-quantised to 8 bits inside the object; all three are
    // in NORMALISED luma units and therefore inside the [0,1] contract (see the header
    // block). `Luma scale` converts back to the [0,255] scale cv.jit works in.
    halp::texture_output<"Mean", halp::r32f_texture> mean;         // [0, 1]
    halp::texture_output<"Variance", halp::r32f_texture> variance; // [0, 0.25]
    halp::texture_output<"StdDev", halp::r32f_texture> stddev;     // [0, 0.5]
    halp::val_port<"Frames", int> frames;
    // 255: mean*scale, stddev*scale, variance*scale^2 are the cv.jit-scale values.
    halp::val_port<"Luma scale", float> luma_scale;
  } outputs;

  // cv.jit's `reset` message: both cv.jit.mean instances in the patch are reset. Their
  // buffers are overwritten on the next frame anyway (a = 1, b = 0), so zeroing here is
  // behaviourally identical and keeps the exposed textures sane in the meantime.
  void reset() noexcept
  {
    m_index = 0;
    std::fill(m_mean.begin(), m_mean.end(), 0.0);
    std::fill(m_var.begin(), m_var.end(), 0.0);
  }

  halp_start_messages(TemporalStats)
    halp_mem_fun(reset)
  halp_end_messages

  void operator()() noexcept
  {
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

    if(W != m_width || H != m_height)
    {
      m_width = W;
      m_height = H;
      m_mean.assign(N, 0.0);
      m_var.assign(N, 0.0);
      m_index = 0;
    }

    ++m_index;
    const double a = 1.0 / static_cast<double>(m_index); // cv.jit.mean's `a`; b = 1 - a

    outputs.mean.create(W, H);
    outputs.variance.create(W, H);
    outputs.stddev.create(W, H);

    const auto src = cv_support::as_rgba(in);
    float* mOut = outputs.mean.texture.bytes;
    float* vOut = outputs.variance.texture.bytes;
    float* sOut = outputs.stddev.texture.bytes;

    for(std::size_t i = 0; i < N; ++i)
    {
      const std::uint8_t* p = src.data + i * 4;
      // DELIBERATE DEVIATION: cv.jit works plane-by-plane on the whole Jitter matrix.
      // Our texture input is RGBA8 and the exact single-channel outputs are r32f, so the
      // frame is reduced to Rec.601 luma, NORMALISED to [0,1] so that the outputs satisfy
      // the r32f contract. Multiply by `Luma scale` (255) to land back on the [0,255] scale
      // cv.jit's char path operates in.
      const double x
          = (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) * (1.0 / 255.0);

      // mu_N = x_N/N + mu_{N-1}*(1 - 1/N)   (cv.jit.mean, updated *first*)
      //
      // Written in the algebraically identical incremental form
      //     mu += (x - mu) * a     ==     x*a + mu*(1 - a)
      // because the literal product form is not a fixed point in binary floating point:
      // with x constant it returns x*(a + (1-a)) which drifts by ~1 ulp, leaving a ~1e-28
      // noise floor in the variance of a *perfectly static* pixel. The incremental form is
      // exact there, so an unchanging region reports variance == 0 and stddev == 0 rather
      // than denormal dust. Identical to within rounding everywhere else.
      // CV/Cpu/CumulativeMean.hpp implements the same cv.jit.mean update and uses the same
      // incremental form for the same reason — the two headers are kept in agreement.
      const double mu = m_mean[i] + (x - m_mean[i]) * a;
      m_mean[i] = mu;

      // d = x - mu, then the same cumulative mean applied to d*d.
      const double d = x - mu;
      const double v = m_var[i] + (d * d - m_var[i]) * a;
      m_var[i] = v;

      mOut[i] = static_cast<float>(mu);
      vOut[i] = static_cast<float>(v);
      sOut[i] = static_cast<float>(std::sqrt(v < 0.0 ? 0.0 : v));
    }

    outputs.mean.texture.changed = true;
    outputs.variance.texture.changed = true;
    outputs.stddev.texture.changed = true;
    outputs.frames = static_cast<int>(m_index);
    outputs.luma_scale = 255.f;
  }

private:
  std::vector<double> m_mean; // width*height running mean of x
  std::vector<double> m_var;  // width*height running mean of (x - mean)^2
  int m_width{};
  int m_height{};
  long long m_index{};
  bool m_was_reset{};
};
}
