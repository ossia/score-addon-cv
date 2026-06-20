#include "FastCorners.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace cv
{
namespace
{
// Bresenham circle of radius 3 (the 16 FAST offsets), clockwise.
constexpr std::array<std::pair<int, int>, 16> kCircle{
    {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
     {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}}};

constexpr int kN = 9; // contiguous arc length for FAST-9
}

void FastCorners::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;
  if(!in.bytes || in.width == 0 || in.height == 0)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);
  const std::size_t N = static_cast<std::size_t>(W) * H;

  m_gray.resize(N);
  for(int i = 0; i < W * H; ++i)
  {
    const std::uint8_t* p = src.data + static_cast<std::size_t>(i) * 4;
    m_gray[static_cast<std::size_t>(i)]
        = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.f / 255.f);
  }

  const float t = inputs.threshold.value;
  m_scoremap.assign(N, 0.f);

  auto at = [&](int x, int y) -> float {
    return m_gray[static_cast<std::size_t>(y) * W + x];
  };

  // FAST-9 segment test: a corner if >= N contiguous circle pixels are all brighter than
  // (Ip + t) or all darker than (Ip - t). Score = sum of absolute differences over the arc.
  for(int y = 3; y < H - 3; ++y)
  {
    for(int x = 3; x < W - 3; ++x)
    {
      const float Ip = at(x, y);
      const float hi = Ip + t;
      const float lo = Ip - t;

      // Quick rejection using the 4 compass points.
      int brighter = 0, darker = 0;
      for(int k = 0; k < 16; k += 4)
      {
        float v = at(x + kCircle[k].first, y + kCircle[k].second);
        if(v > hi)
          ++brighter;
        else if(v < lo)
          ++darker;
      }
      if(brighter < 3 && darker < 3)
        continue;

      // Full contiguous-arc test (wrap-around over 16, looking for kN in a row).
      std::array<int, 16> cls; // +1 brighter, -1 darker, 0 similar
      for(int k = 0; k < 16; ++k)
      {
        float v = at(x + kCircle[k].first, y + kCircle[k].second);
        cls[static_cast<std::size_t>(k)] = (v > hi) ? 1 : (v < lo) ? -1 : 0;
      }

      auto hasArc = [&](int want) {
        int run = 0, best = 0;
        for(int k = 0; k < 16 + kN; ++k)
        {
          if(cls[static_cast<std::size_t>(k % 16)] == want)
          {
            ++run;
            best = std::max(best, run);
          }
          else
            run = 0;
        }
        return best >= kN;
      };

      if(!hasArc(1) && !hasArc(-1))
        continue;

      float score = 0.f;
      for(int k = 0; k < 16; ++k)
        score += std::abs(at(x + kCircle[k].first, y + kCircle[k].second) - Ip);
      m_scoremap[static_cast<std::size_t>(y) * W + x] = score;
    }
  }

  // Optional 3x3 non-maximum suppression on the score map.
  outputs.corners.value.clear();
  const float invW = 1.f / W;
  const float invH = 1.f / H;
  const bool suppress = inputs.suppress.value;
  const int cap = inputs.max_corners.value;
  const float minDist = inputs.min_distance.value;

  // Collect ALL surviving corners (in pixel space) with their scores. We must keep the
  // strongest by score (goodFeaturesToTrack semantics), not the first in raster order, so we
  // gather everything first and sort by score descending below.
  struct Cand
  {
    int x, y;
    float score;
  };
  std::vector<Cand> cands;
  for(int y = 3; y < H - 3; ++y)
  {
    for(int x = 3; x < W - 3; ++x)
    {
      const float s = m_scoremap[static_cast<std::size_t>(y) * W + x];
      if(s <= 0.f)
        continue;
      if(suppress)
      {
        bool isMax = true;
        for(int dy = -1; dy <= 1 && isMax; ++dy)
          for(int dx = -1; dx <= 1; ++dx)
          {
            if(dx == 0 && dy == 0)
              continue;
            if(m_scoremap[static_cast<std::size_t>(y + dy) * W + (x + dx)] > s)
            {
              isMax = false;
              break;
            }
          }
        if(!isMax)
          continue;
      }
      cands.push_back({x, y, s});
    }
  }

  // Sort candidates by descending strength so the greedy spacing pass and the cap both keep
  // the strongest corners.
  std::sort(
      cands.begin(), cands.end(),
      [](const Cand& a, const Cand& b) { return a.score > b.score; });

  // Min-distance de-clustering, in *pixel* space (min_distance is normalised by the image
  // diagonal-independent max(W,H) so it is roughly scale-invariant). When minDist == 0 this
  // accepts every candidate in strength order, i.e. just "strongest N".
  const float minDistPx = minDist * static_cast<float>(std::max(W, H));
  const float minDist2 = minDistPx * minDistPx;
  std::vector<Cand> accepted;
  accepted.reserve(cands.size());
  for(const auto& c : cands)
  {
    if(static_cast<int>(accepted.size()) >= cap)
      break;
    if(minDist2 > 0.f)
    {
      bool tooClose = false;
      for(const auto& a : accepted)
      {
        const float ddx = static_cast<float>(c.x - a.x);
        const float ddy = static_cast<float>(c.y - a.y);
        if(ddx * ddx + ddy * ddy < minDist2)
        {
          tooClose = true;
          break;
        }
      }
      if(tooClose)
        continue;
    }
    accepted.push_back(c);
  }

  outputs.corners.value.reserve(accepted.size());
  for(const auto& c : accepted)
  {
    corner_point cp;
    cp.position = {c.x * invW, c.y * invH};
    cp.score = c.score;
    outputs.corners.value.push_back(cp);
  }

  // Visualization: corners as white dots on black.
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  std::fill(out.bytes, out.bytes + N, std::uint8_t{0});
  for(auto& c : outputs.corners.value)
  {
    int x = static_cast<int>(c.position.x * W);
    int y = static_cast<int>(c.position.y * H);
    if(x >= 0 && y >= 0 && x < W && y < H)
      out.bytes[static_cast<std::size_t>(y) * W + x] = 255;
  }
  out.changed = true;

  outputs.count = static_cast<int>(outputs.corners.value.size());
}
}
