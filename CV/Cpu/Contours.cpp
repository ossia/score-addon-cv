#include "Contours.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <utility>

namespace cv
{
namespace
{
using Pt = std::pair<int, int>;

// 8 neighbour offsets, clockwise from East (image coordinates: y grows downwards).
constexpr std::array<Pt, 8> kMoore{
    {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};

int dirIndex(int dx, int dy)
{
  for(int i = 0; i < 8; ++i)
    if(kMoore[i].first == dx && kMoore[i].second == dy)
      return i;
  return 0;
}

// ------------------------------------------------------------------ border following
//
// Suzuki-Abe ("Topological structural analysis of digitized binary images by border
// following", 1985) is the algorithm behind OpenCV's findContours(), hence behind
// cv.jit.findcontours. Two pieces of bookkeeping are needed on top of plain Moore tracing:
//
//  * `nbd`   : 0 for background, 1 for foreground not yet seen on any border, and the id
//              (>= 2) of the last border that ran over the pixel. An OUTER border may only
//              start on a pixel that still reads 1, which is what stops a hole boundary
//              from being re-reported as a second outer contour.
//  * `hole_exit` : the sign bit of Suzuki-Abe's -NBD marking. It is set on a border pixel
//              when the East neighbour was one of the *examined* background cells during
//              the neighbourhood sweep, i.e. when the background lies east of the pixel on
//              the outside of that border. Those are exactly the pixels a hole border may
//              start on, so marking them stops a hole from being traced twice. Testing the
//              East neighbour directly instead would break the 1px-wide ring, whose outer
//              and inner borders run over the very same pixels.
struct BorderFollower
{
  const std::uint8_t* bin;
  std::int32_t* nbd;
  std::uint8_t* hole_exit;
  int PW;
  int id;
  int maxSteps;
  bool truncated{false};

  bool fg(int x, int y) const noexcept
  {
    return bin[static_cast<std::size_t>(y) * PW + x] != 0;
  }

  // Clockwise sweep of the 8-neighbourhood of (cx,cy) starting just after `backDir`.
  // Returns the direction of the first foreground neighbour, or -1 if there is none.
  // `eastBg` reports whether the East cell was among the examined background cells (the
  // backtrack cell itself counts as examined: it is the cell we came from).
  int sweep(int cx, int cy, int backDir, bool& eastBg) const noexcept
  {
    eastBg = false;
    if(backDir == 0 && !fg(cx + 1, cy))
      eastBg = true;

    for(int k = 1; k <= 8; ++k)
    {
      const int d = (backDir + k) % 8;
      const int nx = cx + kMoore[d].first;
      const int ny = cy + kMoore[d].second;
      if(fg(nx, ny))
        return d;
      if(d == 0)
        eastBg = true;
    }
    return -1;
  }

  void mark(int x, int y, bool eastBg) noexcept
  {
    const std::size_t i = static_cast<std::size_t>(y) * PW + x;
    if(eastBg) // Suzuki-Abe (3.4)(a): f <- -NBD
    {
      hole_exit[i] = 1;
      nbd[i] = id;
    }
    else if(nbd[i] == 1) // (3.4)(b): f <- NBD
    {
      nbd[i] = id;
    }
    // (3.4)(c): leave untouched
  }

  // Follow the border that starts at (sx,sy), entered from the background cell (fx,fy).
  // Returns the ordered boundary points.
  std::vector<Pt> follow(int sx, int sy, int fx, int fy)
  {
    std::vector<Pt> pts;

    bool eastBg = false;
    const int d0 = sweep(sx, sy, dirIndex(fx - sx, fy - sy), eastBg);
    mark(sx, sy, eastBg);
    pts.emplace_back(sx, sy);
    if(d0 < 0)
      return pts; // isolated pixel

    const int f1x = sx + kMoore[d0].first;
    const int f1y = sy + kMoore[d0].second;

    int px = sx, py = sy; // previous
    int cx = f1x, cy = f1y; // current
    int steps = 0;

    while(true)
    {
      bool eb = false;
      const int nd = sweep(cx, cy, dirIndex(px - cx, py - cy), eb);
      if(nd < 0)
        break; // unreachable: (px,py) is a foreground neighbour of (cx,cy)

      const int nx = cx + kMoore[nd].first;
      const int ny = cy + kMoore[nd].second;

      // Jacob's stopping criterion, proper form. The walk is a deterministic function of
      // the directed edge (previous -> current), so it is complete exactly when that state
      // repeats: when we leave the start pixel in the same direction as the very first
      // time. The cheap approximation ("stop as soon as the start pixel is reached again")
      // truncates every boundary that legitimately runs through its start pixel twice -
      // e.g. a bowtie pinched at its topmost-leftmost pixel, for which it reports only one
      // of the two lobes.
      if(cx == sx && cy == sy && nx == f1x && ny == f1y)
        break;

      mark(cx, cy, eb);
      pts.emplace_back(cx, cy);

      px = cx;
      py = cy;
      cx = nx;
      cy = ny;

      if(++steps >= maxSteps)
      {
        truncated = true;
        break;
      }
    }
    return pts;
  }
};

// ----------------------------------------------------------------- Douglas-Peucker
//
// Direct Ramer-Douglas-Peucker, iterative (an explicit stack rather than recursion: a
// pathological boundary can be O(W*H) points deep). Boost.Geometry has
// simplify()/douglas_peucker but it would mean converting to and from its point/ring types
// for ~30 lines of arithmetic, and it does not do the closed-curve split below.

double lineDist(Pt p, Pt a, Pt b) noexcept
{
  const double px = p.first, py = p.second;
  const double ax = a.first, ay = a.second;
  const double dx = static_cast<double>(b.first) - ax;
  const double dy = static_cast<double>(b.second) - ay;
  const double len2 = dx * dx + dy * dy;
  if(len2 <= 0.0)
    return std::hypot(px - ax, py - ay);
  return std::abs(dy * (px - ax) - dx * (py - ay)) / std::sqrt(len2);
}

double dist2(Pt a, Pt b) noexcept
{
  const double dx = static_cast<double>(a.first) - b.first;
  const double dy = static_cast<double>(a.second) - b.second;
  return dx * dx + dy * dy;
}

// Simplify the open polyline pts[idx[0]] .. pts[idx.back()]; both ends are always kept.
void rdpChain(
    const std::vector<Pt>& pts, const std::vector<int>& idx, double eps,
    std::vector<char>& keep)
{
  const int m = static_cast<int>(idx.size());
  if(m < 3)
    return;

  std::vector<std::pair<int, int>> stack; // (lo, hi) index ranges left to process
  stack.emplace_back(0, m - 1);
  while(!stack.empty())
  {
    const auto [lo, hi] = stack.back();
    stack.pop_back();
    if(hi <= lo + 1)
      continue;

    double best = -1.0;
    int besti = -1;
    for(int i = lo + 1; i < hi; ++i)
    {
      const double d = lineDist(pts[idx[i]], pts[idx[lo]], pts[idx[hi]]);
      if(d > best)
      {
        best = d;
        besti = i;
      }
    }
    if(besti >= 0 && best > eps)
    {
      keep[idx[besti]] = 1;
      stack.emplace_back(lo, besti);
      stack.emplace_back(besti, hi);
    }
  }
}

// approxPolyDP(..., closed = true): split the ring at its two most distant points, then
// run RDP on the two open chains. Retained points are emitted in traversal order, so the
// simplified polygon keeps the orientation and the starting point of the raw boundary.
std::vector<Pt> simplifyClosed(const std::vector<Pt>& pts, double eps)
{
  const int n = static_cast<int>(pts.size());
  if(eps <= 0.0 || n < 3)
    return pts;

  int i1 = 0;
  double best = -1.0;
  for(int i = 1; i < n; ++i)
  {
    const double d = dist2(pts[i], pts[0]);
    if(d > best)
    {
      best = d;
      i1 = i;
    }
  }
  int i2 = 0;
  best = -1.0;
  for(int i = 0; i < n; ++i)
  {
    const double d = dist2(pts[i], pts[i1]);
    if(d > best)
    {
      best = d;
      i2 = i;
    }
  }
  if(i1 == i2)
    return pts; // degenerate: every point identical

  const int a = std::min(i1, i2);
  const int b = std::max(i1, i2);

  std::vector<char> keep(static_cast<std::size_t>(n), 0);
  keep[a] = 1;
  keep[b] = 1;

  std::vector<int> idx;
  idx.reserve(static_cast<std::size_t>(n) + 1);
  for(int i = a; i <= b; ++i)
    idx.push_back(i);
  rdpChain(pts, idx, eps, keep);

  idx.clear();
  for(int i = b; i < n; ++i)
    idx.push_back(i);
  for(int i = 0; i <= a; ++i)
    idx.push_back(i);
  rdpChain(pts, idx, eps, keep);

  std::vector<Pt> out;
  out.reserve(static_cast<std::size_t>(n));
  for(int i = 0; i < n; ++i)
    if(keep[i])
      out.push_back(pts[i]);
  return out;
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
  const std::size_t PN = static_cast<std::size_t>(PW) * PH;
  m_bin.assign(PN, 0);
  m_nbd.assign(PN, 0);
  m_hole_exit.assign(PN, 0);
  for(int y = 0; y < H; ++y)
  {
    for(int x = 0; x < W; ++x)
    {
      const std::uint8_t* p = src.data + (static_cast<std::size_t>(y) * W + x) * 4;
      const float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
      if(luma >= thr)
      {
        const std::size_t i = static_cast<std::size_t>(y + 1) * PW + (x + 1);
        m_bin[i] = 1;
        m_nbd[i] = 1;
      }
    }
  }

  outputs.contours.value.clear();
  outputs.points.value.clear();

  // Output texture (single channel): draw boundaries.
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  std::fill(out.bytes, out.bytes + static_cast<std::size_t>(W) * H, std::uint8_t{0});

  const float invW = 1.f / W;
  const float invH = 1.f / H;
  const int min_perim = inputs.min_perimeter.value;
  const double eps = std::max(0.f, inputs.epsilon.value);
  const bool want_holes = inputs.find_holes.value;

  int maxSteps = m_max_trace_steps;
  if(maxSteps <= 0)
  {
    const long long guard = 8ll * W * H + 8ll;
    maxSteps = static_cast<int>(std::min<long long>(guard, INT_MAX));
  }

  // Border bookkeeping, indexed by border id. Id 1 is the image frame: a virtual hole
  // border that encloses every outermost contour, exactly as in the paper.
  std::vector<char> border_hole{0, 1};
  std::vector<int> border_parent{-1, -1};
  std::vector<int> border_out{-1, -1}; // index in the emitted list, -1 if not emitted
  int nbdCounter = 1;
  bool any_truncated = false;

  for(int y = 1; y <= H; ++y)
  {
    int LNBD = 1; // last border met on this scanline; the frame at the start of a row
    for(int x = 1; x <= W; ++x)
    {
      const std::size_t idx = static_cast<std::size_t>(y) * PW + x;
      if(m_nbd[idx] == 0)
        continue; // background

      bool start = false;
      bool isHole = false;
      int fx = 0, fy = 0;
      if(m_nbd[idx] == 1 && m_nbd[idx - 1] == 0)
      {
        // Outer border start: untouched foreground with background to the west.
        start = true;
        isHole = false;
        fx = x - 1;
        fy = y;
      }
      else if(m_nbd[idx] >= 1 && m_nbd[idx + 1] == 0 && !m_hole_exit[idx])
      {
        // Hole border start: foreground with background to the east, not already the
        // exit pixel of a border. Traced even when "Find holes" is off, because the
        // labelling it leaves behind is what keeps the hole from being reported as an
        // extra outer contour; it is simply not emitted in that case.
        start = true;
        isHole = true;
        fx = x + 1;
        fy = y;
        if(m_nbd[idx] > 1)
          LNBD = m_nbd[idx];
      }

      if(start)
      {
        const int id = ++nbdCounter;
        // Suzuki-Abe table 1: same type as the enclosing border -> share its parent,
        // different type -> the enclosing border *is* the parent.
        const bool lnbdHole = border_hole[static_cast<std::size_t>(LNBD)] != 0;
        const int par = (lnbdHole == isHole)
                            ? border_parent[static_cast<std::size_t>(LNBD)]
                            : LNBD;
        border_hole.push_back(isHole ? 1 : 0);
        border_parent.push_back(par);
        border_out.push_back(-1);

        BorderFollower bf{
            m_bin.data(), m_nbd.data(), m_hole_exit.data(), PW, id, maxSteps, false};
        const auto raw = bf.follow(x, y, fx, fy);
        if(bf.truncated)
          any_truncated = true;

        const bool emit = want_holes || !isHole;
        if(emit && raw.size() >= 2)
        {
          const auto poly = simplifyClosed(raw, eps);
          const int n = static_cast<int>(poly.size());

          // Geometry: shoelace area, perimeter, centroid, bbox (in padded coords).
          double area2 = 0.0, perim = 0.0, cxAcc = 0.0, cyAcc = 0.0;
          int minx = PW, miny = PH, maxx = 0, maxy = 0;
          for(int i = 0; i < n; ++i)
          {
            const auto [x0, y0] = poly[i];
            const auto [x1, y1] = poly[(i + 1) % n];
            area2 += static_cast<double>(x0) * y1 - static_cast<double>(x1) * y0;
            perim += std::hypot(
                static_cast<double>(x1 - x0), static_cast<double>(y1 - y0));
            cxAcc += x0;
            cyAcc += y0;
            minx = std::min(minx, x0);
            miny = std::min(miny, y0);
            maxx = std::max(maxx, x0);
            maxy = std::max(maxy, y0);
          }

          if(perim >= static_cast<double>(min_perim))
          {
            const double area = std::abs(area2) * 0.5;
            const float cx = static_cast<float>(cxAcc / n);
            const float cy = static_cast<float>(cyAcc / n);

            // Nearest emitted ancestor: a parent dropped by the min-perimeter filter must
            // not leave a dangling index behind. Ids only ever refer to earlier borders,
            // so this walk terminates.
            int p = border_parent[static_cast<std::size_t>(id)];
            while(p > 1 && border_out[static_cast<std::size_t>(p)] < 0)
              p = border_parent[static_cast<std::size_t>(p)];

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
            ci.is_hole = isHole ? 1 : 0;
            ci.parent = (p > 1) ? border_out[static_cast<std::size_t>(p)] : -1;

            const int ordinal = static_cast<int>(outputs.contours.value.size());
            outputs.contours.value.push_back(ci);
            border_out[static_cast<std::size_t>(id)] = ordinal;

            // Flat point list, cv.jit style: {x, y, contour ordinal} in traversal order.
            outputs.points.value.reserve(
                outputs.points.value.size() + static_cast<std::size_t>(n));
            for(const auto& [px, py] : poly)
              outputs.points.value.push_back(
                  contour_point{(px - 1.f) * invW, (py - 1.f) * invH, ordinal});

            // Draw the *raw* boundary into the output, so the texture stays a proper
            // outline even when Epsilon collapses the point list to a few vertices.
            for(const auto& [px, py] : raw)
            {
              const int ix = px - 1, iy = py - 1;
              if(ix >= 0 && iy >= 0 && ix < W && iy < H)
                out.bytes[static_cast<std::size_t>(iy) * W + ix] = 255;
            }
          }
        }
      }

      // Suzuki-Abe (4): remember the last border seen on this scanline.
      if(m_nbd[idx] != 1)
        LNBD = m_nbd[idx];
    }
  }

  outputs.count = static_cast<int>(outputs.contours.value.size());
  outputs.truncated = any_truncated;
  out.changed = true;
}
}
