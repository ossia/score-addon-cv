#pragma once

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

namespace cv
{
// Multi-dimensional random number generator (cv.jit.noise). Emits a list of `Dimensions`
// floats every tick, each drawn from its own distribution, all sharing a single Mersenne
// Twister so that a given seed always reproduces the same stream.
//
// Two distributions are supported, selected per dimension: uniform (bounded by Min / Max)
// and normal (parameterised by Mean / Stddev), exactly as in cv.jit.noise.
//
// Per-dimension parameters are list inputs (Mode / Min / Max / Mean / Stddev). They follow
// cv.jit's rule: when fewer values are supplied than there are dimensions, the LAST supplied
// value repeats into the remaining slots -- so a one-element list configures every dimension.
// Growing `Dimensions` likewise replicates the last element of every stored parameter array
// into the newly created slots.
//
// Deviations from cv.jit.noise, all deliberate:
//  * Per-dimension parameters are list *inputs* rather than Max array attributes; there is no
//    other way to express a variable-length per-dimension parameter in a control-rate object.
//    An empty list leaves the stored values untouched (the constructor defaults being
//    uniform / min 0 / max 1 / mean 0 / stddev 1, i.e. cv.jit.noise with no creation args).
//  * Mode is a numeric list (0 = uniform, 1 = normal) since Max symbols have no equivalent.
//    cv.jit.noise has a bug in cv_jit_noise_set_mode -- it reads atom_getsym(argv) instead of
//    argv + j, so every dimension ends up with dimension 0's symbolic mode. That bug is NOT
//    reproduced here: Mode is honoured per dimension.
//  * The uniform bounds are sorted before building the distribution (cv.jit.noise only sorts
//    the creation arguments, so setting min > max through the attributes yields a malformed
//    std::uniform_real_distribution); Stddev is clamped to >= 0 for the same reason.
//  * `Seed` is a control: 0 (the default) means "seed from the wall clock", matching
//    cv.jit.noise's default; any other value reseeds deterministically whenever it changes.
//    The `seed` message reseeds from the wall clock on demand.
struct Noise
{
  halp_meta(name, "Random Noise");
  halp_meta(c_name, "cv_noise");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Multi-dimensional random number generator: emits a list of N floats per tick, each "
      "uniform (Min/Max) or normal (Mean/Stddev) per dimension, from a seedable generator.");
  halp_meta(uuid, "c1a70000-0003-4a00-9000-000000000003");

  // cv.jit.noise's MAX_DIMS.
  static constexpr int max_dims = 128;

  enum Distribution
  {
    Uniform = 0,
    Normal = 1
  };

  struct
  {
    halp::hslider_i32<"Dimensions", halp::range{1, max_dims, 1}> dims;

    // 0 = uniform, 1 = normal. Values are rounded, out-of-range values clamped.
    struct
    {
      halp_meta(name, "Mode") std::vector<float> value;
    } mode;
    struct
    {
      halp_meta(name, "Min") std::vector<float> value;
    } min;
    struct
    {
      halp_meta(name, "Max") std::vector<float> value;
    } max;
    struct
    {
      halp_meta(name, "Mean") std::vector<float> value;
    } mean;
    struct
    {
      halp_meta(name, "Stddev") std::vector<float> value;
    } stddev;

    // 0: seeded from the wall clock. Non-zero: deterministic; reseeds when it changes.
    halp::spinbox_i32<"Seed", halp::range{0, 1000000000, 0}> seed;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Values") std::vector<float> value;
    } values;
  } outputs;

  Noise()
  {
    reseed_from_clock();
    set_dims(1);
  }

  // Message: reseed the generator from the wall clock (cv.jit.noise's "seed" with no argument).
  void seed() { reseed_from_clock(); }

  halp_start_messages(Noise)
    halp_mem_fun(seed)
  halp_end_messages

  void operator()() noexcept
  {
    bool dirty = set_dims(inputs.dims.value);

    if(inputs.seed.value != m_seed)
    {
      m_seed = inputs.seed.value;
      if(m_seed != 0)
        m_gen.seed(static_cast<std::uint_fast32_t>(m_seed));
      else
        reseed_from_clock();
    }

    dirty |= assign(m_mode, inputs.mode.value);
    dirty |= assign(m_min, inputs.min.value);
    dirty |= assign(m_max, inputs.max.value);
    dirty |= assign(m_mean, inputs.mean.value);
    dirty |= assign(m_stddev, inputs.stddev.value);

    if(dirty)
      rebuild();

    const std::size_t n = static_cast<std::size_t>(m_dims);
    auto& out = outputs.values.value;
    out.resize(n);

    // One draw per dimension, in dimension order, from the single shared generator.
    for(std::size_t i = 0; i < n; i++)
    {
      const bool normal = m_mode[i] >= 0.5f;
      const double v = normal ? m_normal[i](m_gen) : m_uniform[i](m_gen);
      out[i] = static_cast<float>(v);
    }
  }

private:
  void reseed_from_clock()
  {
    const auto t = std::chrono::system_clock::now().time_since_epoch().count();
    m_gen.seed(static_cast<std::uint_fast32_t>(t));
  }

  // Grow/shrink a parameter array, replicating its last element into any new slot.
  static void resize_replicate(std::vector<float>& v, std::size_t n, float def)
  {
    v.resize(n, v.empty() ? def : v.back());
  }

  // Returns true if the dimension count actually changed.
  bool set_dims(int requested)
  {
    const int d = std::clamp(requested, 1, max_dims);
    if(d == m_dims)
      return false;

    m_dims = d;
    const std::size_t n = static_cast<std::size_t>(d);
    resize_replicate(m_mode, n, static_cast<float>(Uniform));
    resize_replicate(m_min, n, 0.f);
    resize_replicate(m_max, n, 1.f);
    resize_replicate(m_mean, n, 0.f);
    resize_replicate(m_stddev, n, 1.f);
    m_uniform.resize(n);
    m_normal.resize(n);
    return true;
  }

  // cv.jit's per-dimension array attribute semantics: the last supplied value repeats.
  // An empty list means "not driven": the stored values are left alone.
  static bool assign(std::vector<float>& dst, const std::vector<float>& src)
  {
    if(src.empty())
      return false;

    bool changed = false;
    for(std::size_t i = 0; i < dst.size(); i++)
    {
      const float v = src[std::min(i, src.size() - 1)];
      if(dst[i] != v)
      {
        dst[i] = v;
        changed = true;
      }
    }
    return changed;
  }

  void rebuild()
  {
    for(std::size_t i = 0; i < m_uniform.size(); i++)
    {
      const double lo = std::min(m_min[i], m_max[i]);
      const double hi = std::max(m_min[i], m_max[i]);
      m_uniform[i] = std::uniform_real_distribution<double>(lo, hi);
      m_normal[i] = std::normal_distribution<double>(
          m_mean[i], std::max(0.f, m_stddev[i]));
    }
  }

  std::mt19937 m_gen;
  std::vector<std::uniform_real_distribution<double>> m_uniform;
  std::vector<std::normal_distribution<double>> m_normal;

  std::vector<float> m_mode, m_min, m_max, m_mean, m_stddev;
  int m_dims = 0;
  int m_seed = 0;
};
}
