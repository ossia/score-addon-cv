#pragma once

/* score-addon-cv — shared chessboard-corner detection (OpenCV-free, Eigen-based).
 *
 * Used by both CV/Cpu/ChessboardCorners and CV/Cpu/Calibration so the detection
 * logic lives in exactly one place.
 *
 * Pipeline (port of cv.jit.findchessboardcorners' idea, fully re-implemented):
 *   1. RGBA8 -> Rec.601 luma (float [0,1]).
 *   2. 3x3 separable smoothing to stabilise the Hessian on the continuous luma.
 *   3. Saddle-point detection: at each pixel compute the 2x2 Hessian of the
 *      smoothed luma. A checkerboard X-junction is a saddle: the Hessian has one
 *      strongly positive and one strongly negative eigenvalue, i.e. determinant
 *      < 0 with both |eigenvalues| large. Non-maximum suppression on the saddle
 *      response keeps one point per junction. The effective response gate is
 *      `saddle_frac` scaled by the user `threshold` control (see below), so the
 *      `threshold` input directly raises/lowers how strong a junction must be to
 *      survive: lower threshold -> permissive (more, weaker saddles kept), higher
 *      threshold -> strict (only the strongest junctions kept).
 *   4. Order the detected saddle points into an (cols x rows) inner-corner grid.
 *
 * LIMITATION (documented): step 4 is a best-effort topological ordering. We sort
 * the points into rows by projecting them onto the dominant grid axes (found via
 * PCA of the point cloud), bin them into `rows` bands along the minor axis, then
 * sort each band along the major axis. This recovers the correct ordering for a
 * roughly fronto-parallel / mildly perspective board, but does NOT implement a
 * full graph/topology-based ordering (as OpenCV's findChessboardCorners does) and
 * may mis-order under strong perspective or when spurious saddles survive. The
 * `found` flag is only set when exactly cols*rows points remain after detection.
 */

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cv_support
{

struct ChessboardResult
{
  // Inner corners, normalised to [0,1] (x = col/width, y = row/height), ordered
  // row-major: first row left-to-right, then next row, for a (cols x rows) grid.
  std::vector<std::array<float, 2>> corners;
  bool found = false; // true iff exactly cols*rows corners were ordered
};

struct ChessboardParams
{
  int cols = 7;     // inner corners per row
  int rows = 7;     // inner corners per column
  // Saddle-response gate, in [0,1]. Scales `saddle_frac` so it actually controls
  // detection strictness: a saddle survives iff its response >= effective_frac*max,
  // where effective_frac = saddle_frac * (0.25 + 1.5*threshold). At threshold=0.5
  // the multiplier is 1 (legacy behaviour); threshold->0 is permissive, ->1 strict.
  float threshold = 0.5f;
  int adaptive_radius = 5;
  float saddle_frac = 0.15f; // base response fraction, scaled by `threshold`
};

namespace detail
{
// Gaussian-ish 3x3 smoothing to stabilise the Hessian.
inline void smooth3(
    const std::vector<float>& in, std::vector<float>& out, int W, int H)
{
  out.assign(in.size(), 0.f);
  static constexpr float k[3] = {0.25f, 0.5f, 0.25f};
  std::vector<float> tmp(in.size(), 0.f);
  // horizontal
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      float s = 0.f;
      for(int dx = -1; dx <= 1; ++dx)
      {
        int xx = std::clamp(x + dx, 0, W - 1);
        s += k[dx + 1] * in[(std::size_t)y * W + xx];
      }
      tmp[(std::size_t)y * W + x] = s;
    }
  // vertical
  for(int y = 0; y < H; ++y)
    for(int x = 0; x < W; ++x)
    {
      float s = 0.f;
      for(int dy = -1; dy <= 1; ++dy)
      {
        int yy = std::clamp(y + dy, 0, H - 1);
        s += k[dy + 1] * tmp[(std::size_t)yy * W + x];
      }
      out[(std::size_t)y * W + x] = s;
    }
}
} // namespace detail

// Detect & order chessboard inner corners. Pure CPU, Eigen-free (only std math),
// but lives under cv_support to be shared. `gray` is filled as a side buffer.
inline ChessboardResult find_chessboard_corners(
    const RgbaView& src, const ChessboardParams& p)
{
  ChessboardResult R;
  if(!src.valid())
    return R;

  const int W = src.width;
  const int H = src.height;
  const std::size_t N = (std::size_t)W * H;

  // 1. luma
  std::vector<float> gray(N);
  to_gray<float>(src, gray.data());

  // 2. smooth
  std::vector<float> g;
  detail::smooth3(gray, g, W, H);

  auto at = [&](const std::vector<float>& v, int x, int y) -> float {
    return v[(std::size_t)y * W + x];
  };

  // 3. saddle response = -det(Hessian) when det < 0 (saddle), else 0.
  //    det(H) = Hxx*Hyy - Hxy^2 ; for a saddle det < 0. Magnitude |det| ranks it.
  std::vector<float> resp(N, 0.f);
  float maxResp = 0.f;
  const int b = 2; // border for finite differences
  for(int y = b; y < H - b; ++y)
  {
    for(int x = b; x < W - b; ++x)
    {
      // second derivatives (central differences)
      const float c = at(g, x, y);
      const float Hxx = at(g, x + 1, y) - 2.f * c + at(g, x - 1, y);
      const float Hyy = at(g, x, y + 1) - 2.f * c + at(g, x, y - 1);
      const float Hxy = 0.25f
                        * (at(g, x + 1, y + 1) - at(g, x + 1, y - 1)
                           - at(g, x - 1, y + 1) + at(g, x - 1, y - 1));

      const float det = Hxx * Hyy - Hxy * Hxy;
      if(det >= 0.f)
        continue; // not a saddle

      // eigenvalues of the 2x2 symmetric Hessian
      const float tr = Hxx + Hyy;
      const float disc = std::sqrt(std::max(0.f, tr * tr - 4.f * det));
      const float l1 = 0.5f * (tr + disc);
      const float l2 = 0.5f * (tr - disc);
      // saddle: one clearly positive, one clearly negative
      const float lo = std::min(l1, l2);
      const float hi = std::max(l1, l2);
      if(hi <= 0.f || lo >= 0.f)
        continue;
      const float strength = std::min(hi, -lo); // both must be strong
      resp[(std::size_t)y * W + x] = strength;
      maxResp = std::max(maxResp, strength);
    }
  }

  if(maxResp <= 0.f)
    return R;

  // 3b. non-maximum suppression (3x3) + threshold on response.
  // The user `threshold` control scales the base saddle fraction so it genuinely
  // gates detection: threshold=0.5 -> multiplier 1 (legacy), 0 -> 0.25 (permissive),
  // 1 -> 1.75 (strict). Clamp the effective fraction to a sane (0,1] range.
  const float thr = std::clamp(p.threshold, 0.f, 1.f);
  const float effFrac = std::clamp(p.saddle_frac * (0.25f + 1.5f * thr), 0.f, 1.f);
  const float respThr = effFrac * maxResp;
  struct Pt
  {
    float x, y, s;
  };
  std::vector<Pt> pts;
  const int win = std::max(2, std::min(W, H) / (4 * std::max(p.cols, p.rows)));
  for(int y = b; y < H - b; ++y)
  {
    for(int x = b; x < W - b; ++x)
    {
      const float s = resp[(std::size_t)y * W + x];
      if(s < respThr)
        continue;
      bool isMax = true;
      for(int dy = -win; dy <= win && isMax; ++dy)
        for(int dx = -win; dx <= win; ++dx)
        {
          int xx = std::clamp(x + dx, 0, W - 1);
          int yy = std::clamp(y + dy, 0, H - 1);
          if(resp[(std::size_t)yy * W + xx] > s)
          {
            isMax = false;
            break;
          }
        }
      if(isMax)
        pts.push_back({(float)x, (float)y, s});
    }
  }

  // If too many, keep the strongest cols*rows*K then trim during ordering.
  const int want = p.cols * p.rows;
  if((int)pts.size() > want)
  {
    std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& bb) {
      return a.s > bb.s;
    });
    // keep a generous superset to allow the grid sort to discard outliers
    const int keep = std::min((int)pts.size(), std::max(want, want * 3));
    pts.resize(keep);
  }

  // 4. order into a (cols x rows) grid via PCA axes -> banded sort.
  if((int)pts.size() < want)
  {
    // still emit what we have (unordered-ish), but not "found"
    R.corners.reserve(pts.size());
    std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& bb) {
      return (a.y < bb.y) || (a.y == bb.y && a.x < bb.x);
    });
    for(auto& pt : pts)
      R.corners.push_back({pt.x / (float)W, pt.y / (float)H});
    R.found = false;
    return R;
  }

  // centroid
  double mx = 0, my = 0;
  for(auto& pt : pts)
  {
    mx += pt.x;
    my += pt.y;
  }
  mx /= pts.size();
  my /= pts.size();
  // covariance
  double cxx = 0, cyy = 0, cxy = 0;
  for(auto& pt : pts)
  {
    const double dx = pt.x - mx, dy = pt.y - my;
    cxx += dx * dx;
    cyy += dy * dy;
    cxy += dx * dy;
  }
  cxx /= pts.size();
  cyy /= pts.size();
  cxy /= pts.size();
  // principal axes (2x2 symmetric eigvecs)
  const double tr = cxx + cyy;
  const double dd = std::sqrt(std::max(0.0, (cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy));
  const double lA = 0.5 * (tr + dd); // major
  // eigenvector for lA
  double ax, ay;
  if(std::abs(cxy) > 1e-9)
  {
    ax = lA - cyy;
    ay = cxy;
  }
  else
  {
    ax = (cxx >= cyy) ? 1.0 : 0.0;
    ay = (cxx >= cyy) ? 0.0 : 1.0;
  }
  const double an = std::hypot(ax, ay);
  ax /= an;
  ay /= an;
  // minor axis is perpendicular
  const double bx = -ay, by = ax;

  // Decide which axis spans the "rows" direction. We want `rows` bands. Project
  // onto both axes; the axis with the larger extent maps to the longer board
  // dimension. We band along the axis that should hold `rows` groups.
  // Heuristic: assume major axis = the larger of (cols,rows) dimension.
  int nMajor = std::max(p.cols, p.rows);
  int nMinor = std::min(p.cols, p.rows);
  const bool colsIsMajor = (p.cols >= p.rows);

  struct Proj
  {
    float maj, min;
    int idx;
  };
  std::vector<Proj> pr(pts.size());
  for(std::size_t i = 0; i < pts.size(); ++i)
  {
    const double dx = pts[i].x - mx, dy = pts[i].y - my;
    pr[i] = {(float)(dx * ax + dy * ay), (float)(dx * bx + dy * by), (int)i};
  }

  // band along minor axis into nMinor groups, each of nMajor points
  std::sort(pr.begin(), pr.end(), [](const Proj& a, const Proj& bb) {
    return a.min < bb.min;
  });
  // If we have a superset, keep the central want points by minor coordinate is
  // risky; instead trim by taking nMinor*nMajor best-fit: we already have >= want.
  // Greedily form bands of nMajor from the minor-sorted list.
  std::vector<std::array<float, 2>> ordered;
  ordered.reserve(want);
  // We may have more than want points; process in chunks of nMajor but we need
  // exactly nMinor bands. Re-trim to exactly want by discarding minor-extreme
  // extras evenly is complex; simplest robust choice: if more than want, drop the
  // weakest-response extras first (already sorted by strength earlier when >want).
  if((int)pr.size() > want)
  {
    // keep the `want` points whose minor coordinate is most central — but better:
    // since extras are spurious, drop those farthest from forming bands. Fallback:
    // keep the first `want` by minor order is wrong; instead cluster: take all,
    // band into nMinor by quantiles, then within each band keep nMajor strongest.
    std::vector<Proj> trimmed;
    trimmed.reserve(want);
    const int total = (int)pr.size();
    for(int bnd = 0; bnd < nMinor; ++bnd)
    {
      int lo = (int)((long long)bnd * total / nMinor);
      int hi = (int)((long long)(bnd + 1) * total / nMinor);
      std::vector<Proj> band(pr.begin() + lo, pr.begin() + hi);
      std::sort(band.begin(), band.end(), [](const Proj& a, const Proj& bb) {
        return a.maj < bb.maj;
      });
      if((int)band.size() > nMajor)
      {
        // keep nMajor strongest by response
        std::sort(band.begin(), band.end(), [&](const Proj& a, const Proj& bb) {
          return pts[a.idx].s > pts[bb.idx].s;
        });
        band.resize(nMajor);
        std::sort(band.begin(), band.end(), [](const Proj& a, const Proj& bb) {
          return a.maj < bb.maj;
        });
      }
      for(auto& q : band)
        trimmed.push_back(q);
    }
    pr.swap(trimmed);
  }

  if((int)pr.size() != want)
  {
    // ordering failed to land on exactly the grid size
    std::sort(pr.begin(), pr.end(), [](const Proj& a, const Proj& bb) {
      return a.min < bb.min;
    });
    for(auto& q : pr)
      R.corners.push_back({pts[q.idx].x / (float)W, pts[q.idx].y / (float)H});
    R.found = false;
    return R;
  }

  // Now pr has exactly want, minor-sorted. Form nMinor bands of nMajor.
  for(int bnd = 0; bnd < nMinor; ++bnd)
  {
    std::vector<Proj> band(
        pr.begin() + (std::size_t)bnd * nMajor,
        pr.begin() + (std::size_t)(bnd + 1) * nMajor);
    std::sort(band.begin(), band.end(), [](const Proj& a, const Proj& bb) {
      return a.maj < bb.maj;
    });
    for(auto& q : band)
      ordered.push_back({pts[q.idx].x / (float)W, pts[q.idx].y / (float)H});
  }

  // `ordered` is laid out band-by-band. If cols is the major axis, bands ARE rows
  // and each band has `cols` entries -> already row-major (cols x rows). If rows
  // is the major axis, bands are columns; transpose to row-major.
  if(colsIsMajor)
  {
    R.corners = std::move(ordered);
  }
  else
  {
    // ordered[col*nMajor + row] -> want corners[row*cols + col]
    R.corners.assign(want, {0.f, 0.f});
    const int cols = p.cols, rows = p.rows;
    for(int colB = 0; colB < nMinor; ++colB)       // nMinor == cols here
      for(int rowB = 0; rowB < nMajor; ++rowB)     // nMajor == rows here
        R.corners[(std::size_t)rowB * cols + colB]
            = ordered[(std::size_t)colB * nMajor + rowB];
  }

  R.found = true;
  return R;
}

} // namespace cv_support
