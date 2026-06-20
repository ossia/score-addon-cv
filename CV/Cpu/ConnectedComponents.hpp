#pragma once

// Shared OpenCV-free connected-components labeling (two-pass union-find, 8-connectivity).
// Used by Label (visualization) and BlobStats (per-blob measurements).

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cv_support
{

struct LabelResult
{
  std::vector<std::int32_t> labels; // width*height, 0 = background, 1..count = blob id
  std::vector<std::int32_t> sizes;  // [0] unused, [l] = pixel count of blob l
  int width{};
  int height{};
  int count{};
};

namespace detail
{
inline std::int32_t uf_find(std::vector<std::int32_t>& parent, std::int32_t x)
{
  std::int32_t root = x;
  while(parent[root] != root)
    root = parent[root];
  while(parent[x] != root)
  {
    std::int32_t next = parent[x];
    parent[x] = root;
    x = next;
  }
  return root;
}

inline void uf_union(std::vector<std::int32_t>& parent, std::int32_t a, std::int32_t b)
{
  std::int32_t ra = uf_find(parent, a);
  std::int32_t rb = uf_find(parent, b);
  if(ra != rb)
    parent[std::max(ra, rb)] = std::min(ra, rb);
}
}

// Label an RGBA8 image's luminance-thresholded foreground. min_size (0 = keep all) drops
// blobs smaller than the threshold and recompacts ids. Returns compact labels in [1,count].
inline LabelResult
label_connected(const RgbaView& src, std::uint8_t thr, int min_size = 0)
{
  LabelResult R;
  R.width = src.width;
  R.height = src.height;
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
          detail::uf_union(parent, best, lbl);
      };
      consider(x - 1, y);
      consider(x - 1, y - 1);
      consider(x, y - 1);
      consider(x + 1, y - 1);

      if(best == 0)
      {
        best = next_label++;
        parent.push_back(best);
      }
      labels[idx] = best;
    }
  }

  // Resolve equivalences -> compact ids, accumulate sizes.
  std::vector<std::int32_t> remap(parent.size(), 0);
  std::vector<std::int32_t> sizes(1, 0);
  std::int32_t final_count = 0;
  for(std::size_t i = 0; i < N; ++i)
  {
    std::int32_t lbl = labels[i];
    if(lbl == 0)
      continue;
    std::int32_t root = detail::uf_find(parent, lbl);
    if(remap[root] == 0)
    {
      remap[root] = ++final_count;
      sizes.push_back(0);
    }
    std::int32_t fl = remap[root];
    labels[i] = fl;
    sizes[static_cast<std::size_t>(fl)]++;
  }

  if(min_size > 0 && final_count > 0)
  {
    std::vector<std::int32_t> keep(static_cast<std::size_t>(final_count) + 1, 0);
    std::vector<std::int32_t> newSizes(1, 0);
    std::int32_t kept = 0;
    for(std::int32_t l = 1; l <= final_count; ++l)
    {
      if(sizes[static_cast<std::size_t>(l)] >= min_size)
      {
        keep[static_cast<std::size_t>(l)] = ++kept;
        newSizes.push_back(sizes[static_cast<std::size_t>(l)]);
      }
    }
    for(std::size_t i = 0; i < N; ++i)
      labels[i] = keep[static_cast<std::size_t>(labels[i])];
    final_count = kept;
    sizes.swap(newSizes);
  }

  R.labels = std::move(labels);
  R.sizes = std::move(sizes);
  R.count = final_count;
  return R;
}

} // namespace cv_support
