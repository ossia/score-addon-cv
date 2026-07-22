#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace cv
{
// One source -> destination correspondence for the arbitrary-N list input of Homography.
//
// cv.jit.findhomography carries the same information as *two* 1-plane float32 `n x 2`
// Jitter matrices (dim[0] = n points, dim[1] = 2, i.e. row 0 = x and row 1 = y), one for
// the source set and one for the destination set, and quietly does nothing when either has
// fewer than 4 points. Here the two matrices are interleaved into a single list element so
// that one score cable carries a complete, always-consistent correspondence set (it is
// impossible to feed 7 source points and 5 destination points, which cv.jit tolerates by
// bailing out silently).
struct homography_correspondence
{
  float sx, sy; // source point
  float dx, dy; // destination point

  halp_field_names(sx, sy, dx, dy);
};

// Estimator used to fit the homography.
enum class HomographyMethod
{
  // Plain Hartley-normalised least-squares DLT over *every* correspondence.
  // This is the cv.jit-parity mode and therefore the default: cv.jit.findhomography calls
  // cvFindHomography(&src, &dst, &out, 0, 0, NULL) -- method 0 == CV_LMEDS-free plain
  // least squares, no robust estimation whatsoever.
  LeastSquares,
  // Random sample consensus over minimal 4-point samples, followed by a least-squares
  // refit on the largest consensus set. Strictly opt-in: it is *not* what cv.jit does.
  RANSAC
};

// Homography / perspective transform (cv.jit.findhomography, cv.jit.getperspective).
// Computes the 3x3 matrix mapping source points to destination points via the Direct
// Linear Transform, solved as the null vector of a 2N x 9 system with Eigen SVD, on
// Hartley-normalised coordinates. With exactly 4 correspondences this is the classic
// exactly-determined DLT; with more it is an over-determined least-squares fit.
// Outputs the 9 coefficients (row-major) usable as a projective warp. Pure Eigen.
//
// N is arbitrary (N >= 4) through the "Points" list input. When that list is empty the
// object falls back to the legacy fixed spinbox pairs: the first 4 Src/Dst pairs are always
// used and Src/Dst 4..7 are added when their "Use point N" toggle is on (a toggle is
// required because the spinboxes cannot express "unused"). Existing patches therefore keep
// behaving exactly as before.
struct Homography
{
  halp_meta(name, "Homography (N points)");
  halp_meta(c_name, "cv_homography");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "3x3 perspective transform from N >= 4 point correspondences (least-squares DLT, "
      "optional RANSAC). Feed the Points list for arbitrary N; when it is empty the 4-8 "
      "Src/Dst spinbox pairs are used instead.");
  halp_meta(uuid, "8c1f6e0a-4b29-4d57-9e83-2a0c7b5d1f64");

  struct
  {
    halp::xy_spinboxes_f32<"Src 0"> src0;
    halp::xy_spinboxes_f32<"Src 1"> src1;
    halp::xy_spinboxes_f32<"Src 2"> src2;
    halp::xy_spinboxes_f32<"Src 3"> src3;
    halp::xy_spinboxes_f32<"Dst 0"> dst0;
    halp::xy_spinboxes_f32<"Dst 1"> dst1;
    halp::xy_spinboxes_f32<"Dst 2"> dst2;
    halp::xy_spinboxes_f32<"Dst 3"> dst3;
    halp::xy_spinboxes_f32<"Src 4"> src4;
    halp::xy_spinboxes_f32<"Src 5"> src5;
    halp::xy_spinboxes_f32<"Src 6"> src6;
    halp::xy_spinboxes_f32<"Src 7"> src7;
    halp::xy_spinboxes_f32<"Dst 4"> dst4;
    halp::xy_spinboxes_f32<"Dst 5"> dst5;
    halp::xy_spinboxes_f32<"Dst 6"> dst6;
    halp::xy_spinboxes_f32<"Dst 7"> dst7;
    halp::toggle<"Use point 4"> use4;
    halp::toggle<"Use point 5"> use5;
    halp::toggle<"Use point 6"> use6;
    halp::toggle<"Use point 7"> use7;

    // Arbitrary-N correspondence list. When non-empty it *replaces* the spinbox pairs
    // entirely (the toggles are ignored). N >= 4 is required, exactly like cv.jit.
    struct
    {
      halp_meta(name, "Points");
      std::vector<homography_correspondence> value;
    } points;

    // Default is LeastSquares == cv.jit behaviour.
    halp::enum_t<HomographyMethod, "Method"> method;
    // Maximum forward transfer distance ||H*s - d|| (in destination units) for a
    // correspondence to count as a RANSAC inlier. Ignored in LeastSquares mode.
    halp::hslider_f32<"RANSAC threshold", halp::range{0.001f, 100.f, 3.f}> ransac_threshold;
    // Upper bound on the number of minimal (4-point) samples drawn -- NOT the number
    // actually drawn: the loop stops as soon as `RANSAC confidence` is reached (see
    // solveRansac). Ignored in LeastSquares mode.
    halp::hslider_i32<"RANSAC iterations", halp::range{1, 10000, 2000}> ransac_iterations;
    // Probability that at least one all-inlier minimal sample has been drawn, i.e. the
    // `confidence` of cv::findHomography / cv::solvePnPRansac (same 0.99 default as
    // SolvePnP). Setting it to 0 disables the adaptive stop and runs all
    // `RANSAC iterations` samples. Ignored in LeastSquares mode.
    halp::hslider_f32<"RANSAC confidence", halp::range{0.f, 1.f, 0.99f}> ransac_confidence;
  } inputs;

  struct
  {
    halp::val_port<"Matrix", std::array<float, 9>> matrix;
    halp::val_port<"Valid", bool> valid;
    // Number of correspondences that agree with the reported matrix. In LeastSquares mode
    // every correspondence is used by construction, so this is simply N and the mask is
    // all-ones; in RANSAC mode it is the size of the winning consensus set.
    halp::val_port<"Inliers", int> inliers;
    struct
    {
      halp_meta(name, "Inlier mask");
      std::vector<int> value; // 1 per input correspondence, in input order
    } inlier_mask;
  } outputs;

  void operator()() noexcept
  {
    std::vector<Eigen::Vector2d> src, dst;
    gather(src, dst);

    const int N = static_cast<int>(src.size());
    if(N < 4)
    {
      // cv.jit.findhomography silently does nothing below 4 points; here that is reported
      // through Valid = false so the patch can react.
      fail(N);
      return;
    }

    std::vector<int> all(N);
    for(int i = 0; i < N; ++i)
      all[i] = i;

    Eigen::Matrix3d H;
    std::vector<std::uint8_t> mask;

    bool ok = false;
    if(inputs.method.value == HomographyMethod::RANSAC)
    {
      ok = solveRansac(src, dst, H, mask);
    }
    else
    {
      ok = solveDLT(src, dst, all, H);
      mask.assign(static_cast<std::size_t>(N), std::uint8_t{1});
    }

    if(!ok)
    {
      fail(N);
      return;
    }

    std::array<float, 9> m;
    for(int r = 0; r < 3; ++r)
      for(int c = 0; c < 3; ++c)
        m[r * 3 + c] = static_cast<float>(H(r, c));
    outputs.matrix = m;
    outputs.valid = true;

    int count = 0;
    outputs.inlier_mask.value.resize(static_cast<std::size_t>(N));
    for(int i = 0; i < N; ++i)
    {
      const int v = mask[static_cast<std::size_t>(i)] ? 1 : 0;
      outputs.inlier_mask.value[static_cast<std::size_t>(i)] = v;
      count += v;
    }
    outputs.inliers = count;
  }

private:
  // Collect the active correspondence set: the list input when non-empty, else the legacy
  // spinbox pairs (first 4 mandatory, 4..7 gated by their toggle).
  void gather(std::vector<Eigen::Vector2d>& src, std::vector<Eigen::Vector2d>& dst)
      const noexcept
  {
    const auto& list = inputs.points.value;
    if(!list.empty())
    {
      src.reserve(list.size());
      dst.reserve(list.size());
      for(const auto& c : list)
      {
        src.emplace_back(c.sx, c.sy);
        dst.emplace_back(c.dx, c.dy);
      }
      return;
    }

    using V2 = Eigen::Vector2d;
    const std::array<V2, 8> srcAll{
        V2{inputs.src0.value.x, inputs.src0.value.y},
        V2{inputs.src1.value.x, inputs.src1.value.y},
        V2{inputs.src2.value.x, inputs.src2.value.y},
        V2{inputs.src3.value.x, inputs.src3.value.y},
        V2{inputs.src4.value.x, inputs.src4.value.y},
        V2{inputs.src5.value.x, inputs.src5.value.y},
        V2{inputs.src6.value.x, inputs.src6.value.y},
        V2{inputs.src7.value.x, inputs.src7.value.y}};
    const std::array<V2, 8> dstAll{
        V2{inputs.dst0.value.x, inputs.dst0.value.y},
        V2{inputs.dst1.value.x, inputs.dst1.value.y},
        V2{inputs.dst2.value.x, inputs.dst2.value.y},
        V2{inputs.dst3.value.x, inputs.dst3.value.y},
        V2{inputs.dst4.value.x, inputs.dst4.value.y},
        V2{inputs.dst5.value.x, inputs.dst5.value.y},
        V2{inputs.dst6.value.x, inputs.dst6.value.y},
        V2{inputs.dst7.value.x, inputs.dst7.value.y}};

    const std::array<bool, 8> used{
        true, true, true, true, inputs.use4.value, inputs.use5.value, inputs.use6.value,
        inputs.use7.value};

    src.reserve(8);
    dst.reserve(8);
    for(int i = 0; i < 8; ++i)
    {
      if(used[i])
      {
        src.push_back(srcAll[i]);
        dst.push_back(dstAll[i]);
      }
    }
  }

  // Hartley normalisation of a point subset: translate to the centroid and scale so the
  // mean distance to the origin is sqrt(2). This conditions the DLT and is what makes the
  // over-determined (N>4) least-squares solve numerically accurate. Computed in double.
  static void normaliser(
      const std::vector<Eigen::Vector2d>& pts, const std::vector<int>& idx,
      Eigen::Matrix3d& T) noexcept
  {
    const double n = static_cast<double>(idx.size());
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for(int i : idx)
      c += pts[static_cast<std::size_t>(i)];
    c /= n;
    double meanDist = 0.0;
    for(int i : idx)
      meanDist += (pts[static_cast<std::size_t>(i)] - c).norm();
    meanDist /= n;
    const double s = (meanDist > 1e-12) ? (std::sqrt(2.0) / meanDist) : 1.0;
    T.setIdentity();
    T(0, 0) = s;
    T(1, 1) = s;
    T(0, 2) = -s * c.x();
    T(1, 2) = -s * c.y();
  }

  // Least-squares DLT over the correspondences selected by `idx` (any size >= 4).
  //
  // Returns false when the configuration is degenerate: the 2n x 9 DLT matrix must have
  // rank exactly 8 for the homography to be determined up to scale. Collinear point sets
  // (or repeated points) drop the rank below 8, which shows up as a *second* vanishing
  // singular value. Checking sigma_7 against sigma_0 catches all of them; simply looking
  // at whether the result is finite does not, since the SVD happily returns an arbitrary
  // vector out of a higher-dimensional null space.
  static bool solveDLT(
      const std::vector<Eigen::Vector2d>& src, const std::vector<Eigen::Vector2d>& dst,
      const std::vector<int>& idx, Eigen::Matrix3d& out) noexcept
  {
    const int n = static_cast<int>(idx.size());
    if(n < 4)
      return false;

    Eigen::Matrix3d Ts, Td;
    normaliser(src, idx, Ts);
    normaliser(dst, idx, Td);

    // Build the 2n x 9 DLT matrix from the normalised correspondences: two rows each. With
    // n=4 this is the classic exactly-determined 8x9 system; with n>4 it is over-determined
    // and the SVD null-vector yields the least-squares homography.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 9);
    for(int k = 0; k < n; ++k)
    {
      const std::size_t i = static_cast<std::size_t>(idx[static_cast<std::size_t>(k)]);
      const Eigen::Vector3d sn = Ts * Eigen::Vector3d{src[i].x(), src[i].y(), 1.0};
      const Eigen::Vector3d dn = Td * Eigen::Vector3d{dst[i].x(), dst[i].y(), 1.0};
      const double x = sn.x(), y = sn.y();
      const double u = dn.x(), v = dn.y();
      A.row(2 * k) << -x, -y, -1.0, 0, 0, 0, u * x, u * y, u;
      A.row(2 * k + 1) << 0, 0, 0, -x, -y, -1.0, v * x, v * y, v;
    }

    // Null vector = right singular vector of smallest singular value (last column of V).
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const auto& sv = svd.singularValues();
    if(sv.size() < 8)
      return false;
    if(!(sv(0) > 0.0))
      return false;
    // Rank must be >= 8: the 8th-largest singular value has to be appreciable.
    if(sv(7) < 1e-7 * sv(0))
      return false;

    const Eigen::Matrix<double, 9, 1> hv = svd.matrixV().col(8);

    // Denormalise: H = Td^-1 * Hn * Ts.
    Eigen::Matrix3d Hn;
    Hn << hv(0), hv(1), hv(2), hv(3), hv(4), hv(5), hv(6), hv(7), hv(8);
    Eigen::Matrix3d Hd = Td.inverse() * Hn * Ts;

    if(!Hd.allFinite())
      return false;
    if(!(std::abs(Hd(2, 2)) > 1e-12))
      return false;
    Hd /= Hd(2, 2); // normalise so H(2,2)=1
    if(!Hd.allFinite())
      return false;

    out = Hd;
    return true;
  }

  // Squared forward transfer error ||H*s / w - d||^2, +inf when the source point maps to
  // (or beyond) the horizon line of H.
  static double transferError2(
      const Eigen::Matrix3d& H, const Eigen::Vector2d& s, const Eigen::Vector2d& d) noexcept
  {
    const Eigen::Vector3d p = H * Eigen::Vector3d{s.x(), s.y(), 1.0};
    if(!(std::abs(p.z()) > 1e-12))
      return std::numeric_limits<double>::infinity();
    const Eigen::Vector2d q{p.x() / p.z(), p.y() / p.z()};
    const double e = (q - d).squaredNorm();
    return std::isfinite(e) ? e : std::numeric_limits<double>::infinity();
  }

  // RANSAC over minimal 4-point samples, then a least-squares refit on the winning
  // consensus set. The RNG is seeded per call with a fixed constant so the object is a
  // pure function of its inputs (identical inputs always give identical outputs), which
  // is what a score patch and the test-suite both expect.
  //
  // ADAPTIVE TERMINATION. `RANSAC iterations` (default 2000) is an upper bound, not a
  // count: after every improvement of the consensus set the standard rule
  //     N = log(1 - confidence) / log(1 - w^s),   w = inliers/total,  s = 4
  // re-estimates how many samples are still needed and shrinks the budget. Running the
  // full 2000 samples unconditionally cost 18.6 ms for N = 20 with 2 outliers (versus
  // 0.015 ms for LeastSquares) -- more than a whole 60 fps frame for one node. The same
  // rule is what CV/Cpu/SolvePnP.hpp's solveRansac already does.
  bool solveRansac(
      const std::vector<Eigen::Vector2d>& src, const std::vector<Eigen::Vector2d>& dst,
      Eigen::Matrix3d& out, std::vector<std::uint8_t>& mask) const noexcept
  {
    const int N = static_cast<int>(src.size());
    const double thr = std::max(1e-9f, inputs.ransac_threshold.value);
    const double thr2 = thr * thr;
    const int maxIters = std::max(1, inputs.ransac_iterations.value);
    const double conf
        = std::clamp(static_cast<double>(inputs.ransac_confidence.value), 0.0, 1.0);

    std::mt19937 rng(0x5eed1234u);
    std::vector<int> perm(static_cast<std::size_t>(N));
    for(int i = 0; i < N; ++i)
      perm[static_cast<std::size_t>(i)] = i;

    std::vector<int> sample(4);
    std::vector<std::uint8_t> cur(static_cast<std::size_t>(N));
    std::vector<std::uint8_t> best;
    int bestCount = 0;
    Eigen::Matrix3d bestH = Eigen::Matrix3d::Identity();
    bool haveBest = false;

    int trials = maxIters;
    for(int it = 0; it < trials && it < maxIters; ++it)
    {
      // Partial Fisher-Yates: 4 distinct indices in O(4).
      for(int k = 0; k < 4; ++k)
      {
        std::uniform_int_distribution<int> d(k, N - 1);
        const int j = d(rng);
        std::swap(perm[static_cast<std::size_t>(k)], perm[static_cast<std::size_t>(j)]);
        sample[static_cast<std::size_t>(k)] = perm[static_cast<std::size_t>(k)];
      }

      Eigen::Matrix3d Hc;
      if(!solveDLT(src, dst, sample, Hc)) // degenerate minimal sample -> skip
        continue;

      int count = 0;
      for(int i = 0; i < N; ++i)
      {
        const bool in
            = transferError2(
                  Hc, src[static_cast<std::size_t>(i)], dst[static_cast<std::size_t>(i)])
              <= thr2;
        cur[static_cast<std::size_t>(i)] = in ? 1 : 0;
        count += in ? 1 : 0;
      }

      if(count > bestCount)
      {
        bestCount = count;
        best = cur;
        bestH = Hc;
        haveBest = true;
        if(bestCount == N)
          break; // cannot do better

        // Adaptive stop: with an inlier ratio w, the chance that a 4-sample is
        // all-inlier is w^4, so log(1-conf)/log(1-w^4) samples suffice for the
        // requested confidence. conf == 0 keeps the full `maxIters` budget.
        const double w = double(bestCount) / double(N);
        const double p = w * w * w * w;
        if(p > 0.0 && p < 1.0 && conf > 0.0)
        {
          const double needed = std::log(1.0 - conf) / std::log(1.0 - p);
          if(std::isfinite(needed))
            trials = std::min<int>(maxIters, std::max(1, int(std::ceil(needed)) + 1));
        }
      }
    }

    if(!haveBest || bestCount < 4)
      return false;

    // Refit on the consensus set (this is what turns the minimal-sample hypothesis into a
    // proper least-squares estimate). If the refit is degenerate, keep the hypothesis.
    std::vector<int> inl;
    inl.reserve(static_cast<std::size_t>(bestCount));
    for(int i = 0; i < N; ++i)
      if(best[static_cast<std::size_t>(i)])
        inl.push_back(i);

    Eigen::Matrix3d Hr;
    if(solveDLT(src, dst, inl, Hr))
    {
      // Re-score with the refined model; never let the refit shrink the consensus set.
      int count = 0;
      std::vector<std::uint8_t> m2(static_cast<std::size_t>(N));
      for(int i = 0; i < N; ++i)
      {
        const bool in
            = transferError2(
                  Hr, src[static_cast<std::size_t>(i)], dst[static_cast<std::size_t>(i)])
              <= thr2;
        m2[static_cast<std::size_t>(i)] = in ? 1 : 0;
        count += in ? 1 : 0;
      }
      if(count >= bestCount)
      {
        out = Hr;
        mask = std::move(m2);
        return true;
      }
    }

    out = bestH;
    mask = std::move(best);
    return true;
  }

  void fail(int n) noexcept
  {
    outputs.matrix = std::array<float, 9>{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    outputs.valid = false;
    outputs.inliers = 0;
    outputs.inlier_mask.value.assign(static_cast<std::size_t>(std::max(0, n)), 0);
  }
};
}
