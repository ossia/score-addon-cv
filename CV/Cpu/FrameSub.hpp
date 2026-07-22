#pragma once

/* score-addon-cv — cv.jit.framesub (OpenCV-free).
 *
 * The Max abstraction is one `jit.op @op absdiff` fed by a `t l l`: the incoming matrix goes
 * to the left inlet (computing against whatever the right inlet still holds, i.e. the
 * *previous* frame) and is then stored into the right inlet for the next tick. So:
 *
 *      out[n] = |frame[n] - frame[n-1]|      (per channel, absolute difference)
 *
 * This is NOT the same thing as CV/Shaders/Filters/FrameDiff.fs, which differences against a
 * slowly-updated running-average background; frame subtraction is a pure motion/edge-of-motion
 * detector with no memory beyond one frame.
 *
 * Deviations from the Max abstraction, both deliberate:
 *  - First frame: jit.op's right inlet starts as a zero matrix, so cv.jit.framesub passes the
 *    very first frame through unchanged (|A - 0| = A). We output black instead: "no previous
 *    frame" is not the same statement as "everything moved", and a whole-frame flash on the
 *    first tick is a nuisance for anything downstream that thresholds motion.
 *  - Alpha: we difference R,G,B only and write alpha = 255. Differencing an opaque frame
 *    against an opaque frame gives alpha 0, i.e. a fully transparent (invisible) result.
 */

#include <CV/Support/EigenImage.hpp>

#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cv
{
struct FrameSub
{
  halp_meta(name, "Frame subtraction");
  halp_meta(c_name, "cv_framesub");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Absolute difference between the current and the previous frame.");
  halp_meta(uuid, "c1a70000-0009-4a00-9000-000000000009");

  struct
  {
    halp::texture_input<"In"> image;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image; // RGBA8, |cur - prev| per colour channel
  } outputs;

  void operator()() noexcept
  {
    auto& in = inputs.image.texture;
    if(!in.changed || !in.bytes || in.width <= 0 || in.height <= 0)
      return;

    const int W = in.width;
    const int H = in.height;
    const auto src = cv_support::as_rgba(in);
    const std::size_t N = static_cast<std::size_t>(W) * H;

    outputs.image.create(W, H);
    std::uint8_t* dst = outputs.image.texture.bytes;

    // No usable history (first frame, or the input was resized): emit black rather than
    // garbage or a full-frame flash, and start the history over.
    const bool have_prev = (m_pw == W && m_ph == H && m_prev.size() == N * 4);
    if(!have_prev)
    {
      for(std::size_t i = 0; i < N; ++i)
      {
        std::uint8_t* d = dst + i * 4;
        d[0] = d[1] = d[2] = 0;
        d[3] = 255;
      }
      m_prev.assign(src.data, src.data + N * 4);
      m_pw = W;
      m_ph = H;
      outputs.image.upload();
      return;
    }

    for(std::size_t i = 0; i < N; ++i)
    {
      const std::uint8_t* c = src.data + i * 4;
      const std::uint8_t* p = m_prev.data() + i * 4;
      std::uint8_t* d = dst + i * 4;
      for(int k = 0; k < 3; ++k)
        d[k] = static_cast<std::uint8_t>(c[k] > p[k] ? c[k] - p[k] : p[k] - c[k]);
      d[3] = 255;
    }

    // The current frame becomes the history for the next tick.
    std::copy(src.data, src.data + N * 4, m_prev.begin());

    outputs.image.upload();
  }

private:
  std::vector<std::uint8_t> m_prev; // previous frame, RGBA8, m_pw * m_ph * 4
  int m_pw = 0, m_ph = 0;
};
}
