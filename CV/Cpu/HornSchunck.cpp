#include "HornSchunck.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cmath>

namespace cv
{
namespace
{
// Clamped fetch from a W*H row-major plane.
inline float at(const std::vector<float>& p, int W, int H, int x, int y) noexcept
{
  x = std::clamp(x, 0, W - 1);
  y = std::clamp(y, 0, H - 1);
  return p[static_cast<std::size_t>(y) * W + x];
}

// Standard Horn & Schunck local average: 1/6 on the 4-neighbours, 1/12 on the diagonals
// (the weights sum to 1). Border pixels replicate the edge.
inline float hs_average(const std::vector<float>& f, int W, int H, int x, int y) noexcept
{
  const float side = at(f, W, H, x - 1, y) + at(f, W, H, x + 1, y)
                     + at(f, W, H, x, y - 1) + at(f, W, H, x, y + 1);
  const float diag = at(f, W, H, x - 1, y - 1) + at(f, W, H, x + 1, y - 1)
                     + at(f, W, H, x - 1, y + 1) + at(f, W, H, x + 1, y + 1);
  return side * (1.f / 6.f) + diag * (1.f / 12.f);
}

// Write one flow plane out under the addon's r32f [0,1] contract, bipolar-encoded.
// Returns true if anything had to be clamped (only possible with a fixed scale).
bool emit_flow(const std::vector<float>& f, float* dst, std::size_t N, float scale) noexcept
{
  bool clipped = false;
  for(std::size_t i = 0; i < N; ++i)
  {
    const float v = std::isfinite(f[i]) ? f[i] : 0.f;
    if(std::abs(v) > scale)
      clipped = true;
    dst[i] = polar_codec::encode_signed01(v, scale);
  }
  return clipped;
}
}

void HornSchunck::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || in.width <= 0 || in.height <= 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const std::size_t N = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);

  // ------------------------------------------------------------------ luminance [0,1]
  const auto src = cv_support::as_rgba(in);
  m_cur.resize(N);
  for(std::size_t i = 0; i < N; ++i)
  {
    const std::uint8_t* p = src.data + i * 4;
    m_cur[i] = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }

  outputs.dx.create(W, H);
  outputs.dy.create(W, H);
  auto& tx = outputs.dx.texture;
  auto& ty = outputs.dy.texture;
  float* ox = reinterpret_cast<float*>(tx.bytes);
  float* oy = reinterpret_cast<float*>(ty.bytes);

  // ------------------------------------------------- first frame / dimension change
  // DEVIATION from cv.jit: it would compare this frame against its zero-cleared internal
  // matrix and emit a large spurious flow. We emit exactly zero instead.
  if(m_w != W || m_h != H || m_prev.size() != N)
  {
    m_w = W;
    m_h = H;
    m_prev = m_cur;
    m_u.assign(N, 0.f);
    m_v.assign(N, 0.f);
    // Zero flow is the middle of the bipolar swing, not 0.0 — decoding it gives exactly 0.
    std::fill_n(ox, N, 0.5f);
    std::fill_n(oy, N, 0.5f);
    outputs.iterations = 0;
    outputs.flow_scale = (inputs.flow_scale.value > 0.f) ? inputs.flow_scale.value : 1.f;
    outputs.clipped = false;
    tx.changed = true;
    ty.changed = true;
    return;
  }

  // ------------------------------------------------------------------- derivatives
  // Original Horn & Schunck (1981) 2x2x2 cube of forward differences: the three
  // derivatives are averaged over the four edges of the cube in each direction, which
  // keeps them consistent with each other. `P` = previous frame, `C` = current frame.
  m_ex.resize(N);
  m_ey.resize(N);
  m_et.resize(N);
  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      const float p00 = at(m_prev, W, H, x, y);
      const float p10 = at(m_prev, W, H, x + 1, y);
      const float p01 = at(m_prev, W, H, x, y + 1);
      const float p11 = at(m_prev, W, H, x + 1, y + 1);
      const float c00 = at(m_cur, W, H, x, y);
      const float c10 = at(m_cur, W, H, x + 1, y);
      const float c01 = at(m_cur, W, H, x, y + 1);
      const float c11 = at(m_cur, W, H, x + 1, y + 1);

      const std::size_t i = static_cast<std::size_t>(y) * W + x;
      m_ex[i] = 0.25f * ((p10 - p00) + (p11 - p01) + (c10 - c00) + (c11 - c01));
      m_ey[i] = 0.25f * ((p01 - p00) + (p11 - p10) + (c01 - c00) + (c11 - c10));
      m_et[i] = 0.25f * ((c00 - p00) + (c10 - p10) + (c01 - p01) + (c11 - p11));
    }
  }

  // -------------------------------------------------------------------- parameters
  const int maxiter = std::max(1, inputs.maxiter.value);
  const float lambda = std::max(0.f, inputs.lambda.value);
  const float lambda2 = lambda * lambda;
  const float eps = std::max(0.f, inputs.threshold.value);

  // cv.jit passes usePrevious = 0: the iteration always restarts from a zero field.
  if(!inputs.use_previous.value)
  {
    std::fill(m_u.begin(), m_u.end(), 0.f);
    std::fill(m_v.begin(), m_v.end(), 0.f);
  }
  m_un.resize(N);
  m_vn.resize(N);

  // --------------------------------------------------------------------- iterate
  int performed = 0;
  for(int it = 0; it < maxiter; ++it)
  {
    float maxdelta = 0.f;
    for(int y = 0; y < H; ++y)
    {
      for(int x = 0; x < W; ++x)
      {
        const std::size_t i = static_cast<std::size_t>(y) * W + x;
        const float ubar = hs_average(m_u, W, H, x, y);
        const float vbar = hs_average(m_v, W, H, x, y);
        const float ex = m_ex[i];
        const float ey = m_ey[i];
        const float denom = lambda2 + ex * ex + ey * ey;

        float nu = ubar;
        float nv = vbar;
        // Degenerate only when lambda == 0 AND the pixel is perfectly flat; then the
        // constraint carries no information and the pixel just takes the local average.
        if(denom > 1e-20f)
        {
          const float t = (ex * ubar + ey * vbar + m_et[i]) / denom;
          nu = ubar - ex * t;
          nv = vbar - ey * t;
        }
        if(!std::isfinite(nu))
          nu = 0.f;
        if(!std::isfinite(nv))
          nv = 0.f;

        maxdelta = std::max(maxdelta, std::abs(nu - m_u[i]));
        maxdelta = std::max(maxdelta, std::abs(nv - m_v[i]));
        m_un[i] = nu;
        m_vn[i] = nv;
      }
    }
    m_u.swap(m_un);
    m_v.swap(m_vn);
    ++performed;

    // CV_TERMCRIT_EPS: cv.jit hands `threshold` to cvTermCriteria as the epsilon. A
    // threshold of 0 (the default) disables the test, so all maxiter sweeps run.
    if(eps > 0.f && maxdelta < eps)
      break;
  }

  // ---------------------------------------------------------------------- output
  // The flow is signed px/frame; an r32f texture output carries [0,1] (see the contract at
  // the top of CV/Cpu/CartoPol.hpp), so it goes out bipolar-encoded against `Flow scale`.
  float scale = inputs.flow_scale.value;
  if(!(scale > 0.f))
  {
    // AUTO: normalise by this frame's peak component, so nothing is ever clipped.
    float peak = 0.f;
    for(std::size_t i = 0; i < N; ++i)
    {
      if(std::isfinite(m_u[i]))
        peak = std::max(peak, std::abs(m_u[i]));
      if(std::isfinite(m_v[i]))
        peak = std::max(peak, std::abs(m_v[i]));
    }
    scale = (peak > 0.f) ? peak : 1.f; // an all-zero field: any scale encodes it exactly
  }

  const bool cx = emit_flow(m_u, ox, N, scale);
  const bool cy = emit_flow(m_v, oy, N, scale);

  outputs.iterations = performed;
  outputs.flow_scale = scale;
  outputs.clipped = cx || cy;
  tx.changed = true;
  ty.changed = true;

  m_prev = m_cur;
}
}
