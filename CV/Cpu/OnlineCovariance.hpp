#pragma once

/* score-addon-cv — cv.jit.covariance (temporal, online, N-dimensional accumulator).
 *
 * NOT the same thing as CV/Shaders/Analysis/Covariance.cs, which computes a *spatial* scalar
 * covariance between two images and merely shares the name. cv.jit.covariance consumes a
 * variable-length 1-D float vector once per frame and accumulates statistics over time,
 * emitting an N×N matrix.
 *
 * Its update (source/projects/cv.jit.covariance/cv.jit.covariance.cpp) is NOT textbook
 * covariance and is reproduced here verbatim, quirks included:
 *
 *     n = ++frame_counter;   nn = (n-1)/n;
 *     if(n == 1) { mean[i] = in[i]; var[i] = 0; }
 *     else {
 *       for(i) mean[i] = mean[i]*nn + in[i]/n;
 *       n--;  nn = (n-1)/n;              // <-- n is decremented HERE, *before* the var loop
 *       for(i) var[i] = var[i]*nn + (in[i]-mean[i])/n;
 *     }
 *     out[j*N + i] = var[j] * var[i];    // rank-1 outer product
 *
 * Two things follow that a textbook implementation would get wrong:
 *  - `var` is a running mean of *deviations* (first moment of x - mean), not a variance;
 *    it is signed and can be negative.
 *  - because the emitted matrix is the outer product var·varᵀ, it is rank 1 and symmetric,
 *    so it is not a covariance matrix in any usual sense.
 * The `n--` placement means the var accumulator lags the mean accumulator by one frame's
 * worth of weight; dropping it changes every value from frame 3 onwards.
 *
 * The upstream header notes: "This external has been deprecated. It is provided only for
 * backward compatibility." Ported faithfully anyway, so patches that relied on its exact
 * (odd) numbers keep working.
 *
 * `reset` sets n = 0 and clears mean/var. Any change of input length does the same.
 */

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>

#include <vector>

namespace cv
{
struct OnlineCovariance
{
  halp_meta(name, "Online covariance");
  halp_meta(c_name, "cv_online_covariance");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Online N-dimensional temporal accumulator over a float vector (cv.jit.covariance, "
      "deprecated upstream): emits the rank-1 N x N outer product of the running mean "
      "deviation.");
  halp_meta(uuid, "c1a70000-000e-4a00-9000-00000000000e");

  struct
  {
    // Variable-length list input: one observation vector per tick.
    halp::val_port<"Vector", std::vector<float>> vec;
    // Rising-edge equivalent of the `reset` message.
    halp::toggle<"Reset"> reset;
  } inputs;

  struct
  {
    // The N x N result, flattened ROW-MAJOR: element (row j, column i) is at index j*N + i.
    // (Symmetric here, so the convention only matters for consumers that reshape it.)
    halp::val_port<"Matrix", std::vector<float>> matrix;
    // Exposed for introspection / testing; cv.jit keeps these in private Jitter matrices.
    halp::val_port<"Mean", std::vector<float>> mean;
    // cv.jit's `var` accumulator: the running mean of (x - mean). Signed, NOT a variance.
    halp::val_port<"Deviation", std::vector<float>> deviation;
    halp::val_port<"Size", int> size;
    halp::val_port<"Frames", int> frames;
  } outputs;

  // cv.jit.covariance's `reset` message.
  void reset() noexcept
  {
    m_n = 0;
    m_mean.assign(m_mean.size(), 0.0);
    m_var.assign(m_var.size(), 0.0);
  }

  halp_start_messages(OnlineCovariance)
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

    const auto& in = inputs.vec.value;
    const std::size_t sz = in.size();

    // Empty input: nothing to accumulate. cv.jit would be handed a degenerate matrix here;
    // we treat it as a length change to 0, i.e. the state is cleared and no frame is counted.
    if(sz == 0)
    {
      m_mean.clear();
      m_var.clear();
      m_n = 0;
      outputs.matrix.value.clear();
      outputs.mean.value.clear();
      outputs.deviation.value.clear();
      outputs.size = 0;
      outputs.frames = 0;
      return;
    }

    // "if((in_minfo.dim[0]!=x->size) ...) { clear var; clear mean; x->size = ...; x->n = 0; }"
    if(sz != m_mean.size())
    {
      m_mean.assign(sz, 0.0);
      m_var.assign(sz, 0.0);
      m_n = 0;
    }

    // n is a *local* double in cv.jit; the object's counter keeps the incremented value while
    // the local one gets decremented halfway through. Reproduced exactly.
    ++m_n;
    double n = static_cast<double>(m_n);
    double nn = (n - 1.0) / n;

    if(m_n == 1)
    {
      for(std::size_t i = 0; i < sz; ++i)
        m_mean[i] = in[i];
      for(std::size_t i = 0; i < sz; ++i)
        m_var[i] = 0.0;
    }
    else
    {
      for(std::size_t i = 0; i < sz; ++i)
        m_mean[i] = m_mean[i] * nn + (static_cast<double>(in[i]) / n);

      n -= 1.0;          // <-- the quirk: var uses n-1, not n
      nn = (n - 1.0) / n;

      for(std::size_t i = 0; i < sz; ++i)
      {
        const double temp = static_cast<double>(in[i]) - m_mean[i];
        m_var[i] = m_var[i] * nn + (temp / n);
      }
    }

    // out[j][i] = var[j] * var[i]
    auto& out = outputs.matrix.value;
    out.resize(sz * sz);
    for(std::size_t j = 0; j < sz; ++j)
    {
      const double vj = m_var[j];
      for(std::size_t i = 0; i < sz; ++i)
        out[j * sz + i] = static_cast<float>(vj * m_var[i]);
    }

    outputs.mean.value.assign(m_mean.begin(), m_mean.end());
    outputs.deviation.value.assign(m_var.begin(), m_var.end());
    outputs.size = static_cast<int>(sz);
    outputs.frames = static_cast<int>(m_n);
  }

private:
  // DELIBERATE DEVIATION: cv.jit dispatches on the incoming matrix type (float32 / float64).
  // We always accumulate in double and emit float, i.e. the float64 path — the more accurate
  // of the two, and the one the object's mop actually forces (jit_mop_single_type float64).
  std::vector<double> m_mean;
  std::vector<double> m_var;
  long long m_n{};
  bool m_was_reset{};
};
}
