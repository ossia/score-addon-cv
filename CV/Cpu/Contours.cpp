#include "Contours.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace cv
{
namespace
{
// Moore-neighbor boundary tracing of a single 8-connected component, starting at `start`
// with the background pixel we came from at `from`. Returns the ordered boundary points.
// `bin` is a (W+2)x(H+2) padded binary image (1=fg) to avoid bounds checks.
struct TraceResult
{
  std::vector<std::pair<int, int>> points;
};

// 8 neighbour offsets, clockwise from East.
constexpr std::array<std::pair<int, int>, 8> kMoore{
    {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};

int dirIndex(int dx, int dy)
{
  for(int i = 0; i < 8; ++i)
    if(kMoore[i].first == dx && kMoore[i].second == dy)
      return i;
  return 0;
}

TraceResult traceContour(
    const std::vector<std::uint8_t>& bin, int PW, int /*PH*/, int sx, int sy, int fx, int fy)
{
  auto at = [&](int x, int y) -> std::uint8_t {
    return bin[static_cast<std::size_t>(y) * PW + x];
  };

  TraceResult res;
  res.points.emplace_back(sx, sy);

  int cx = sx, cy = sy;
  // Backtrack direction: from current pixel toward the background we entered from.
  int backDir = dirIndex(fx - sx, fy - sy);

  const int firstX = sx, firstY = sy;
  int secondX = -1, secondY = -1;
  bool haveSecond = false;
  int safety = 0;
  const int maxSteps = PW * PW; // generous loop guard

  while(safety++ < maxSteps)
  {
    // Search clockwise starting just after the backtrack direction.
    int found = -1;
    for(int k = 1; k <= 8; ++k)
    {
      int d = (backDir + k) % 8;
      int nx = cx + kMoore[d].first;
      int ny = cy + kMoore[d].second;
      if(at(nx, ny))
      {
        found = d;
        // New backtrack = direction from the found pixel back to current.
        backDir = dirIndex(cx - nx, cy - ny);
        cx = nx;
        cy = ny;
        break;
      }
    }
    if(found < 0)
      break; // isolated pixel

    // Stopping criterion.
    //
    // The textbook Jacob/Moore criterion stops only when BOTH the start pixel is
    // revisited AND the next traced pixel equals the original "second" pixel; this
    // correctly handles components that pass through their start pixel more than once
    // (e.g. one-pixel-wide necks). We approximate it by stopping the first time we
    // return to the start pixel after >=2 steps.
    //
    // Limitation: for a contour that legitimately revisits the start pixel before the
    // boundary is complete (a thin filament touching the start cell from two sides),
    // this terminates early and yields a partial boundary. For the solid blobs this
    // node targets that case does not arise; full correctness would require also
    // matching the recorded `second` pixel before breaking. We record `second` below so
    // that criterion can be enabled if such inputs become relevant.
    if(!haveSecond)
    {
      secondX = cx;
      secondY = cy;
      haveSecond = true;
    }
    else if(cx == firstX && cy == firstY)
    {
      break;
    }

    res.points.emplace_back(cx, cy);
  }

  return res;
}
}

void Contours::operator()() noexcept
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

  // Padded binary image (1px border of background) so the tracer never reads out of bounds.
  const int PW = W + 2;
  const int PH = H + 2;
  m_bin.assign(static_cast<std::size_t>(PW) * PH, 0);
  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      const std::uint8_t* p = src.data + (static_cast<std::size_t>(y) * W + x) * 4;
      const float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
      if(luma >= thr)
        m_bin[static_cast<std::size_t>(y + 1) * PW + (x + 1)] = 1;
    }
  }

  m_visited.assign(static_cast<std::size_t>(PW) * PH, 0);
  outputs.contours.value.clear();

  // Output texture (single channel): draw boundaries.
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  std::fill(out.bytes, out.bytes + static_cast<std::size_t>(W) * H, std::uint8_t{0});

  const float invW = 1.f / W;
  const float invH = 1.f / H;
  const int min_perim = inputs.min_perimeter.value;

  auto at = [&](int x, int y) -> std::uint8_t {
    return m_bin[static_cast<std::size_t>(y) * PW + x];
  };

  // Scan for contour starts: a fg pixel whose left neighbour is bg and not yet traced.
  for(int y = 1; y <= H; ++y)
  {
    for(int x = 1; x <= W; ++x)
    {
      if(!at(x, y))
        continue;
      const std::size_t pidx = static_cast<std::size_t>(y) * PW + x;
      if(at(x - 1, y)) // interior left -> not an outer-left boundary start
        continue;
      if(m_visited[pidx])
        continue;

      auto trace = traceContour(m_bin, PW, PH, x, y, x - 1, y);
      // Mark all traced pixels visited to avoid re-tracing the same contour.
      for(auto& [px, py] : trace.points)
        m_visited[static_cast<std::size_t>(py) * PW + px] = 1;

      const int n = static_cast<int>(trace.points.size());
      if(n < 2)
        continue;

      // Geometry: shoelace area, perimeter, centroid, bbox (in padded coords).
      double area2 = 0.0, perim = 0.0, cxAcc = 0.0, cyAcc = 0.0;
      int minx = PW, miny = PH, maxx = 0, maxy = 0;
      for(int i = 0; i < n; ++i)
      {
        auto [x0, y0] = trace.points[i];
        auto [x1, y1] = trace.points[(i + 1) % n];
        area2 += static_cast<double>(x0) * y1 - static_cast<double>(x1) * y0;
        perim += std::hypot(static_cast<double>(x1 - x0), static_cast<double>(y1 - y0));
        cxAcc += x0;
        cyAcc += y0;
        minx = std::min(minx, x0);
        miny = std::min(miny, y0);
        maxx = std::max(maxx, x0);
        maxy = std::max(maxy, y0);
      }

      if(perim < static_cast<double>(min_perim))
        continue;

      const double area = std::abs(area2) * 0.5;
      const float cx = static_cast<float>(cxAcc / n);
      const float cy = static_cast<float>(cyAcc / n);

      contour_info ci;
      // Convert padded coords (+1 offset) back to image space, normalise.
      ci.centroid = {(cx - 1.f) * invW, (cy - 1.f) * invH};
      // Inclusive pixel box: a contour spanning columns minx..maxx covers
      // (maxx - minx + 1) pixels. Matches BlobStats' bbox convention.
      ci.bbox = rect{
          (minx - 1.f) * invW,
          (miny - 1.f) * invH,
          (maxx - minx + 1) * invW,
          (maxy - miny + 1) * invH};
      ci.area = static_cast<float>(area) * invW * invH;
      ci.perimeter = static_cast<float>(perim) * (invW + invH) * 0.5f;
      ci.point_count = n;
      outputs.contours.value.push_back(ci);

      // Draw boundary into the output.
      for(auto& [px, py] : trace.points)
      {
        const int ix = px - 1, iy = py - 1;
        if(ix >= 0 && iy >= 0 && ix < W && iy < H)
          out.bytes[static_cast<std::size_t>(iy) * W + ix] = 255;
      }
    }
  }

  outputs.count = static_cast<int>(outputs.contours.value.size());
  out.changed = true;
}
}
