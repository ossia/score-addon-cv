#include "Label.hpp"

#include <CV/Cpu/ConnectedComponents.hpp>

#include <algorithm>
#include <numeric>

namespace cv
{
namespace
{
// Deterministic blob colour from a label id.
void label_color(std::int32_t id, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b)
{
  std::uint32_t h = static_cast<std::uint32_t>(id) * 2654435761u;
  r = static_cast<std::uint8_t>(60 + ((h) & 0x7F) + 0x40);
  g = static_cast<std::uint8_t>(60 + ((h >> 8) & 0x7F) + 0x40);
  b = static_cast<std::uint8_t>(60 + ((h >> 16) & 0x7F) + 0x40);
}

// cv.jit's char output is a byte: anything that does not fit is emitted as 0
// (cv.jit tests `> 255` and, where it does not, the `(char)` cast turns 256 into 0 too --
// so `> 255 -> 0` is the observable rule in every branch).
constexpr std::int32_t CHAR_LIMIT = 255;

struct RawLabels
{
  std::vector<std::int32_t> labels; // W*H, 0 = background, 1..count = raw blob id
  std::vector<std::int32_t> sizes;  // [0] unused, [l] = pixel count of blob l
  int count{};
  bool overflow{};
};

// Union-find labeling with selectable connectivity and a hard cap on the number of blobs.
//
// This is ConnectedComponents.hpp's algorithm (which is fixed at 8-connectivity and
// unbounded, and is READ-ONLY shared code) with the two extra degrees of freedom
// cv.jit.label needs. Ids are assigned in raster order of each blob's first pixel, which
// is also the order cv.jit's seed-fill discovers them in, so blob indices match.
//
// Cap semantics mirror cv.jit: its scan loop is `for(j = 0; j < width && ndx < 2048; j++)`,
// so once the cap is reached no further seed is ever started and the pixels of the
// undiscovered blobs simply stay background. Blobs found *before* the cap are unaffected.
RawLabels
label_raw(const cv_support::RgbaView& src, std::uint8_t thr, bool four, int cap)
{
  const int W = src.width, H = src.height;
  const std::size_t N = static_cast<std::size_t>(W) * H;

  std::vector<std::int32_t> labels(N, 0);
  std::vector<std::int32_t> parent(1, 0);
  std::int32_t next_label = 1;

  auto fg = [&](int x, int y) -> bool {
    const std::uint8_t* p = src.data + (static_cast<std::size_t>(y) * W + x) * 4;
    const float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    return luma >= thr;
  };

  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      if(!fg(x, y))
        continue;
      const std::size_t idx = static_cast<std::size_t>(y) * W + x;

      std::int32_t best = 0;
      auto consider = [&](int nx, int ny) {
        if(nx < 0 || ny < 0 || nx >= W || ny >= H)
          return;
        std::int32_t lbl = labels[static_cast<std::size_t>(ny) * W + nx];
        if(lbl == 0)
          return;
        if(best == 0)
          best = lbl;
        else
          cv_support::detail::uf_union(parent, best, lbl);
      };
      // 4-neighbourhood: only the already-scanned orthogonal neighbours.
      consider(x - 1, y);
      consider(x, y - 1);
      if(!four)
      {
        consider(x - 1, y - 1);
        consider(x + 1, y - 1);
      }

      if(best == 0)
      {
        best = next_label++;
        parent.push_back(best);
      }
      labels[idx] = best;
    }
  }

  // Resolve equivalences -> compact ids in raster order, accumulate sizes, apply the cap.
  // remap[root]: 0 = not seen yet, > 0 = compact id, -1 = dropped (cap reached).
  std::vector<std::int32_t> remap(parent.size(), 0);
  std::vector<std::int32_t> sizes(1, 0);
  std::int32_t final_count = 0;
  bool overflow = false;
  for(std::size_t i = 0; i < N; ++i)
  {
    std::int32_t lbl = labels[i];
    if(lbl == 0)
      continue;
    std::int32_t root = cv_support::detail::uf_find(parent, lbl);
    if(remap[static_cast<std::size_t>(root)] == 0)
    {
      if(final_count >= cap)
      {
        remap[static_cast<std::size_t>(root)] = -1;
        overflow = true;
      }
      else
      {
        remap[static_cast<std::size_t>(root)] = ++final_count;
        sizes.push_back(0);
      }
    }
    const std::int32_t fl = remap[static_cast<std::size_t>(root)];
    if(fl < 0)
    {
      labels[i] = 0;
      continue;
    }
    labels[i] = fl;
    sizes[static_cast<std::size_t>(fl)]++;
  }

  RawLabels R;
  R.labels = std::move(labels);
  R.sizes = std::move(sizes);
  R.count = final_count;
  R.overflow = overflow;
  return R;
}
}

void Label::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  const std::uint8_t thr = static_cast<std::uint8_t>(
      std::clamp(inputs.threshold.value, 0.f, 1.f) * 255.f + 0.5f);

  const bool four = (inputs.connectivity.value == Connectivity::Four);
  const int cap = std::max(1, inputs.max_blobs.value);

  auto R = label_raw(src, thr, four, cap);
  const int ndx = R.count;
  outputs.overflow = R.overflow;

  // ---------------------------------------------------------------- size filter
  // `index[l]` is the renumbered 1..kept id of blob l (0 = filtered out), which is both
  // cv.jit's mode-0 output and this port's visualization id.
  std::vector<std::int32_t> index(static_cast<std::size_t>(ndx) + 1, 0);
  std::vector<char> pass(static_cast<std::size_t>(ndx) + 1, 0);
  const int ms = inputs.min_size.value;
  const bool strict = (inputs.size_filter.value == SizeFilter::GreaterThan);
  std::int32_t kept = 0;
  for(int l = 1; l <= ndx; ++l)
  {
    const std::int32_t s = R.sizes[static_cast<std::size_t>(l)];
    const bool ok = strict ? (s > ms) : (s >= ms);
    pass[static_cast<std::size_t>(l)] = ok ? 1 : 0;
    if(ok)
      index[static_cast<std::size_t>(l)] = ++kept;
  }
  outputs.count = kept;

  // ----------------------------------------------------------- numeric label value
  std::vector<std::int32_t> value(static_cast<std::size_t>(ndx) + 1, 0);
  const bool charmode = inputs.charmode.value;
  const bool massmode = (inputs.mode.value == LabelMode::Mass);
  if(!charmode)
  {
    // Long output: mode 0 -> index, mode 1 -> mass.
    for(int l = 1; l <= ndx; ++l)
    {
      const std::size_t ul = static_cast<std::size_t>(l);
      value[ul] = massmode ? (pass[ul] ? R.sizes[ul] : 0) : index[ul];
    }
  }
  else if(!massmode)
  {
    // Char output, mode 0: index, clamped to a byte.
    for(int l = 1; l <= ndx; ++l)
    {
      const std::size_t ul = static_cast<std::size_t>(l);
      value[ul] = (index[ul] > CHAR_LIMIT) ? 0 : index[ul];
    }
  }
  else
  {
    // Char output, mode 1: SIZE RANK. cv.jit qsorts blobs[1..ndx] ascending by size and
    // writes `ndx - i + 1` for the blob at 1-based sorted position i, so the LARGEST blob
    // is rank 1 and the smallest is rank ndx. Ranks are computed over *all* blobs, so
    // filtered-out blobs still consume a rank; they just emit 0.
    std::vector<std::int32_t> order(static_cast<std::size_t>(ndx));
    std::iota(order.begin(), order.end(), 1);
    std::stable_sort(
        order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
          return R.sizes[static_cast<std::size_t>(a)]
                 < R.sizes[static_cast<std::size_t>(b)];
        });
    for(int i = 0; i < ndx; ++i)
    {
      const std::size_t ul = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
      const std::int32_t rank = ndx - i; // 0-based i <=> cv.jit's ndx - i + 1
      value[ul] = (!pass[ul] || rank > CHAR_LIMIT) ? 0 : rank;
    }
  }

  // ------------------------------------------------------------------------- output
  // The r32f label field carries [0,1] (see the LABEL FIELD SCALE block in Label.hpp), so
  // the values are divided by the largest one present and that divisor is published on
  // `Max label`. Computed over the whole value table first, so the scale is stable across
  // the pixel loop.
  std::int32_t vmax = 0;
  for(int l = 1; l <= ndx; ++l)
    vmax = std::max(vmax, value[static_cast<std::size_t>(l)]);
  outputs.max_label = static_cast<float>(vmax);
  // An all-background frame leaves every pixel at 0; the divisor is then irrelevant.
  const float inv_max = (vmax > 0) ? (1.f / static_cast<float>(vmax)) : 0.f;

  outputs.image.create(W, H);
  outputs.labels.create(W, H);
  auto& out = outputs.image.texture;
  auto& lblOut = outputs.labels.texture;
  float* lblData = reinterpret_cast<float*>(lblOut.bytes);
  const bool colorize = inputs.colorize.value;
  const std::size_t N = static_cast<std::size_t>(W) * H;
  for(std::size_t i = 0; i < N; ++i)
  {
    std::uint8_t* d = out.bytes + i * 4;
    const std::int32_t raw = R.labels[i];
    const std::int32_t lbl = raw ? index[static_cast<std::size_t>(raw)] : 0;

    // Numeric label field: the cv.jit output-matrix value (0 = background/filtered out),
    // normalised by `Max label`.
    lblData[i]
        = static_cast<float>(raw ? value[static_cast<std::size_t>(raw)] : 0) * inv_max;

    if(lbl == 0)
    {
      d[0] = d[1] = d[2] = 0;
      d[3] = 255;
    }
    else if(colorize)
    {
      label_color(lbl, d[0], d[1], d[2]);
      d[3] = 255;
    }
    else
    {
      std::uint8_t g = static_cast<std::uint8_t>(((lbl - 1) % 255) + 1);
      d[0] = d[1] = d[2] = g;
      d[3] = 255;
    }
  }
  out.changed = true;
  lblOut.changed = true;
}
}
