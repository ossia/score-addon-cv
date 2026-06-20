#include "BlobSort.hpp"

#include <CV/Cpu/ConnectedComponents.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cv
{
void BlobSort::operator()() noexcept
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

  auto R = cv_support::label_connected(src, thr, inputs.min_size.value);

  // Current-frame centroids + areas.
  struct Cur
  {
    double sx = 0, sy = 0, n = 0;
  };
  std::vector<Cur> cur(static_cast<std::size_t>(R.count) + 1);
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      std::int32_t l = R.labels[static_cast<std::size_t>(y) * W + x];
      if(l == 0)
        continue;
      Cur& c = cur[static_cast<std::size_t>(l)];
      c.sx += x;
      c.sy += y;
      c.n += 1;
    }

  const float invW = 1.f / W;
  const float invH = 1.f / H;

  struct CurBlob
  {
    float x, y, area;
  };
  std::vector<CurBlob> blobs;
  blobs.reserve(R.count);
  for(int l = 1; l <= R.count; ++l)
  {
    Cur& c = cur[static_cast<std::size_t>(l)];
    if(c.n <= 0)
      continue;
    blobs.push_back(
        {static_cast<float>(c.sx / c.n) * invW,
         static_cast<float>(c.sy / c.n) * invH,
         static_cast<float>(c.n) * invW * invH});
  }

  // Greedy nearest-neighbour assignment of current blobs to previous ids.
  const float maxd = inputs.max_distance.value;
  const float maxd2 = maxd * maxd;

  std::vector<int> assigned_id(blobs.size(), -1);
  std::vector<int> assigned_age(blobs.size(), 0);
  std::vector<bool> prev_used(m_prev.size(), false);

  // For each current blob, find the closest unused previous blob within range.
  // (O(n*m); blob counts are small in practice.)
  for(std::size_t i = 0; i < blobs.size(); ++i)
  {
    float best = maxd2;
    int bestj = -1;
    for(std::size_t j = 0; j < m_prev.size(); ++j)
    {
      if(prev_used[j])
        continue;
      const float dx = blobs[i].x - m_prev[j].x;
      const float dy = blobs[i].y - m_prev[j].y;
      const float d2 = dx * dx + dy * dy;
      if(d2 < best)
      {
        best = d2;
        bestj = static_cast<int>(j);
      }
    }
    if(bestj >= 0)
    {
      const auto& p = m_prev[static_cast<std::size_t>(bestj)];
      assigned_id[i] = p.id;
      assigned_age[i] = p.age + 1; // matched -> one frame older
      prev_used[static_cast<std::size_t>(bestj)] = true;
    }
  }

  // Previous blobs that were not matched have disappeared: recycle their ids.
  for(std::size_t j = 0; j < m_prev.size(); ++j)
    if(!prev_used[j])
      m_free_ids.push_back(m_prev[j].id);
  std::sort(m_free_ids.begin(), m_free_ids.end());

  // New blobs get fresh ids: prefer the lowest freed id, else grow m_next_id.
  // Age starts at 0 for a newly-seen blob.
  std::size_t free_cursor = 0;
  for(std::size_t i = 0; i < blobs.size(); ++i)
  {
    if(assigned_id[i] >= 0)
      continue;
    if(free_cursor < m_free_ids.size())
      assigned_id[i] = m_free_ids[free_cursor++];
    else
      assigned_id[i] = m_next_id++;
    assigned_age[i] = 0;
  }
  // Drop the freed ids we just reused.
  if(free_cursor > 0)
    m_free_ids.erase(m_free_ids.begin(), m_free_ids.begin() + free_cursor);

  // Emit + rebuild previous-frame state.
  outputs.blobs.value.clear();
  outputs.blobs.value.reserve(blobs.size());
  std::vector<Prev> nextPrev;
  nextPrev.reserve(blobs.size());
  for(std::size_t i = 0; i < blobs.size(); ++i)
  {
    tracked_blob tb;
    tb.centroid = {blobs[i].x, blobs[i].y};
    tb.area = blobs[i].area;
    tb.id = assigned_id[i];
    tb.age = assigned_age[i];
    outputs.blobs.value.push_back(tb);
    nextPrev.push_back({blobs[i].x, blobs[i].y, assigned_id[i], assigned_age[i]});
  }
  m_prev.swap(nextPrev);

  outputs.count = static_cast<int>(outputs.blobs.value.size());
}
}
