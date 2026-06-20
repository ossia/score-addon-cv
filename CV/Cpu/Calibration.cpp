#include "Calibration.hpp"

#include <CV/Support/Chessboard.hpp>
#include <CV/Support/EigenImage.hpp>

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <vector>

namespace cv
{
namespace
{
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Isotropic (Hartley) normalisation of a 2D point set: translate to centroid and
// scale so the mean distance to the origin is sqrt(2). Returns the 3x3 similarity T
// such that p_norm = T * p_homog, and fills `out` with the normalised points.
Eigen::Matrix3d hartley_normalise(
    const std::vector<std::array<double, 2>>& pts,
    std::vector<std::array<double, 2>>& out)
{
  const int n = (int)pts.size();
  double mx = 0, my = 0;
  for(const auto& p : pts)
  {
    mx += p[0];
    my += p[1];
  }
  mx /= n;
  my /= n;

  double meanDist = 0;
  for(const auto& p : pts)
    meanDist += std::hypot(p[0] - mx, p[1] - my);
  meanDist /= n;

  // Scale so the mean distance becomes sqrt(2). Guard against a degenerate cloud.
  const double s = (meanDist > 1e-12) ? (std::sqrt(2.0) / meanDist) : 1.0;

  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  T(0, 0) = s;
  T(1, 1) = s;
  T(0, 2) = -s * mx;
  T(1, 2) = -s * my;

  out.resize(n);
  for(int i = 0; i < n; ++i)
    out[i] = {s * (pts[i][0] - mx), s * (pts[i][1] - my)};
  return T;
}

// Per-view homography mapping planar object points (X,Y) -> image points (u,v)
// via the Direct Linear Transform, solved as the null vector of a 2N x 9 system
// (same technique as CV/Cpu/Homography.hpp, generalised to N points). Both point
// sets are isotropically (Hartley) normalised before the DLT for conditioning, then
// the resulting homography is denormalised: H = Timg^-1 * H_norm * Tobj. Returns
// row-major 3x3.
Eigen::Matrix3d compute_homography(
    const std::vector<std::array<double, 2>>& obj,
    const std::vector<std::array<double, 2>>& img)
{
  const int n = (int)obj.size();

  // Hartley normalisation of both clouds for numerical conditioning.
  std::vector<std::array<double, 2>> objn, imgn;
  const Eigen::Matrix3d Tobj = hartley_normalise(obj, objn);
  const Eigen::Matrix3d Timg = hartley_normalise(img, imgn);

  // Direct Linear Transform: H is the null vector of the 2N x 9 system, found as
  // the right singular vector of the smallest singular value (Eigen JacobiSVD).
  // Same technique as CV/Cpu/Homography.hpp. Element-wise matrix assignment is
  // used deliberately (comma-initialisers into dynamic .row() blocks are fragile).
  MatrixXd A = MatrixXd::Zero(2 * n, 9);
  for(int i = 0; i < n; ++i)
  {
    const double X = objn[i][0], Y = objn[i][1];
    const double u = imgn[i][0], v = imgn[i][1];
    A(2 * i, 0) = -X;
    A(2 * i, 1) = -Y;
    A(2 * i, 2) = -1;
    A(2 * i, 6) = u * X;
    A(2 * i, 7) = u * Y;
    A(2 * i, 8) = u;
    A(2 * i + 1, 3) = -X;
    A(2 * i + 1, 4) = -Y;
    A(2 * i + 1, 5) = -1;
    A(2 * i + 1, 6) = v * X;
    A(2 * i + 1, 7) = v * Y;
    A(2 * i + 1, 8) = v;
  }

  Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullV);
  VectorXd h = svd.matrixV().col(8);
  Eigen::Matrix3d Hn;
  Hn(0, 0) = h(0);
  Hn(0, 1) = h(1);
  Hn(0, 2) = h(2);
  Hn(1, 0) = h(3);
  Hn(1, 1) = h(4);
  Hn(1, 2) = h(5);
  Hn(2, 0) = h(6);
  Hn(2, 1) = h(7);
  Hn(2, 2) = h(8);

  // Denormalise: H maps raw object -> raw image points.
  Eigen::Matrix3d H = Timg.inverse() * Hn * Tobj;
  if(std::abs(H(2, 2)) > 1e-12)
    H /= H(2, 2);
  return H;
}

// Zhang's v_ij constraint row from a homography (columns h1,h2,h3).
Eigen::Matrix<double, 1, 6> v_ij(const Eigen::Matrix3d& H, int i, int j)
{
  // H columns are h1=col0, h2=col1
  const double hi0 = H(0, i), hi1 = H(1, i), hi2 = H(2, i);
  const double hj0 = H(0, j), hj1 = H(1, j), hj2 = H(2, j);
  Eigen::Matrix<double, 1, 6> v;
  v << hi0 * hj0, hi0 * hj1 + hi1 * hj0, hi1 * hj1, hi2 * hj0 + hi0 * hj2,
      hi2 * hj1 + hi1 * hj2, hi2 * hj2;
  return v;
}
} // namespace

void Calibration::operator()() noexcept
{
  auto& in = inputs.image.texture;

  // Reset.
  if(bool(inputs.reset.value))
  {
    m_views.clear();
    outputs.solved = false;
    outputs.rms = 0.f;
  }

  const bool haveImage = in.changed && in.bytes && in.width && in.height;

  // Capture: detect the board now and store its image corners.
  if(bool(inputs.capture.value) && haveImage)
  {
    const auto src = cv_support::as_rgba(in);
    cv_support::ChessboardParams p;
    p.cols = inputs.cols.value;
    p.rows = inputs.rows.value;
    p.threshold = inputs.threshold.value;
    const auto R = cv_support::find_chessboard_corners(src, p);
    if(R.found && (int)R.corners.size() == p.cols * p.rows)
    {
      m_cols = p.cols;
      m_rows = p.rows;
      m_imgW = in.width;
      m_imgH = in.height;
      View v;
      v.img.reserve(R.corners.size());
      for(const auto& c : R.corners)
        v.img.push_back({c[0] * m_imgW, c[1] * m_imgH}); // back to pixels
      m_views.push_back(std::move(v));
    }
  }

  outputs.views = static_cast<int>(m_views.size());

  // Solve.
  if(bool(inputs.solve.value) && m_views.size() >= 3)
  {
    const int cols = m_cols, rows = m_rows;
    const int npts = cols * rows;

    // Object points: planar grid, square size = 1 unit, row-major to match.
    std::vector<std::array<double, 2>> obj(npts);
    for(int r = 0; r < rows; ++r)
      for(int c = 0; c < cols; ++c)
        obj[(std::size_t)r * cols + c] = {(double)c, (double)r};

    // 1. homography per view.
    std::vector<Eigen::Matrix3d> Hs;
    Hs.reserve(m_views.size());
    for(const auto& v : m_views)
    {
      std::vector<std::array<double, 2>> img(npts);
      for(int i = 0; i < npts; ++i)
        img[i] = {(double)v.img[i][0], (double)v.img[i][1]};
      Hs.push_back(compute_homography(obj, img));
    }

    // 2. build V b = 0 for the image of the absolute conic.
    MatrixXd V(2 * Hs.size(), 6);
    for(std::size_t k = 0; k < Hs.size(); ++k)
    {
      V.row(2 * k) = v_ij(Hs[k], 0, 1);                       // v01
      V.row(2 * k + 1) = v_ij(Hs[k], 0, 0) - v_ij(Hs[k], 1, 1); // v00 - v11
    }
    Eigen::JacobiSVD<MatrixXd> svd(V, Eigen::ComputeFullV);
    VectorXd b = svd.matrixV().col(5);
    // b = [B11, B12, B22, B13, B23, B33]
    const double B11 = b(0), B12 = b(1), B22 = b(2), B13 = b(3), B23 = b(4),
                 B33 = b(5);

    // 3. recover intrinsics (Zhang closed form). NOTE: skew is NOT assumed zero —
    //    the general (5-parameter) closed form below computes a nonzero skew into
    //    K(0,1). On a well-conditioned planar target it comes out near zero, but it
    //    is solved for, not forced.
    const double denom = (B11 * B22 - B12 * B12);
    if(std::abs(denom) < 1e-20)
    {
      outputs.solved = false;
      return;
    }
    const double cy = (B12 * B13 - B11 * B23) / denom;
    const double lambda = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11;
    double fx2 = lambda / B11;
    double fy2 = lambda * B11 / denom;
    if(fx2 <= 0 || fy2 <= 0 || !std::isfinite(fx2) || !std::isfinite(fy2))
    {
      outputs.solved = false;
      return;
    }
    const double fx = std::sqrt(fx2);
    const double fy = std::sqrt(fy2);
    const double skew = -B12 * fx * fx * fy / lambda;
    const double cx = skew * cy / fy - B13 * fx * fx / lambda;

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = fx;
    K(0, 1) = skew;
    K(0, 2) = cx;
    K(1, 1) = fy;
    K(1, 2) = cy;

    // 4. extrinsics per view.
    Eigen::Matrix3d Kinv = K.inverse();
    struct Extr
    {
      Eigen::Matrix3d R;
      Eigen::Vector3d t;
    };
    std::vector<Extr> extr;
    extr.reserve(Hs.size());
    for(const auto& H : Hs)
    {
      Eigen::Vector3d h1 = H.col(0), h2 = H.col(1), h3 = H.col(2);
      double l = 1.0 / (Kinv * h1).norm();
      Eigen::Vector3d r1 = l * (Kinv * h1);
      Eigen::Vector3d r2 = l * (Kinv * h2);
      Eigen::Vector3d r3 = r1.cross(r2);
      Eigen::Vector3d t = l * (Kinv * h3);
      // orthonormalise R via SVD
      Eigen::Matrix3d Rm;
      Rm.col(0) = r1;
      Rm.col(1) = r2;
      Rm.col(2) = r3;
      Eigen::JacobiSVD<Eigen::Matrix3d> rsvd(
          Rm, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Rm = rsvd.matrixU() * rsvd.matrixV().transpose();
      extr.push_back({Rm, t});
    }

    // 5. optional radial distortion k1,k2 by linear least squares:
    //    (u - u0) = (u_ideal - u0)(1 + k1 r^2 + k2 r^4), same for v.
    //    Build D [k1 k2]^T = d, in normalised-then-pixel form.
    std::vector<double> Drows; // flattened 2 cols
    std::vector<double> drhs;
    for(std::size_t kview = 0; kview < m_views.size(); ++kview)
    {
      const auto& E = extr[kview];
      for(int i = 0; i < npts; ++i)
      {
        Eigen::Vector3d P(obj[i][0], obj[i][1], 0.0);
        Eigen::Vector3d cam = E.R * P + E.t;
        if(std::abs(cam.z()) < 1e-12)
          continue;
        const double xn = cam.x() / cam.z();
        const double yn = cam.y() / cam.z();
        const double r2 = xn * xn + yn * yn;
        const double r4 = r2 * r2;
        // ideal projection (no distortion)
        const double u = fx * xn + skew * yn + cx;
        const double v = fy * yn + cy;
        const double du = u - cx;
        const double dv = v - cy;
        // observed
        const double uo = m_views[kview].img[i][0];
        const double vo = m_views[kview].img[i][1];
        // rows: [du*r2, du*r4] * [k1 k2] = uo - u
        Drows.push_back(du * r2);
        Drows.push_back(du * r4);
        drhs.push_back(uo - u);
        Drows.push_back(dv * r2);
        Drows.push_back(dv * r4);
        drhs.push_back(vo - v);
      }
    }
    double k1 = 0, k2 = 0;
    if(drhs.size() >= 2)
    {
      const int m = (int)drhs.size();
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>> D(
          Drows.data(), m, 2);
      Eigen::Map<VectorXd> dd(drhs.data(), m);
      Eigen::Vector2d kk
          = D.colPivHouseholderQr().solve(dd.matrix());
      if(std::isfinite(kk(0)) && std::isfinite(kk(1)))
      {
        k1 = kk(0);
        k2 = kk(1);
      }
    }

    // 6. RMS reprojection error (with distortion applied).
    double sse = 0;
    int cnt = 0;
    for(std::size_t kview = 0; kview < m_views.size(); ++kview)
    {
      const auto& E = extr[kview];
      for(int i = 0; i < npts; ++i)
      {
        Eigen::Vector3d P(obj[i][0], obj[i][1], 0.0);
        Eigen::Vector3d cam = E.R * P + E.t;
        if(std::abs(cam.z()) < 1e-12)
          continue;
        const double xn = cam.x() / cam.z();
        const double yn = cam.y() / cam.z();
        const double r2 = xn * xn + yn * yn;
        const double rad = 1.0 + k1 * r2 + k2 * r2 * r2;
        const double xd = xn * rad;
        const double yd = yn * rad;
        const double u = fx * xd + skew * yd + cx;
        const double v = fy * yd + cy;
        const double uo = m_views[kview].img[i][0];
        const double vo = m_views[kview].img[i][1];
        sse += (u - uo) * (u - uo) + (v - vo) * (v - vo);
        ++cnt;
      }
    }
    const double rms = (cnt > 0) ? std::sqrt(sse / cnt) : 0.0;

    // Outputs (GOTCHA 1: assign array/xy via .value).
    std::array<float, 9> Karr{
        (float)K(0, 0), (float)K(0, 1), (float)K(0, 2),
        (float)K(1, 0), (float)K(1, 1), (float)K(1, 2),
        (float)K(2, 0), (float)K(2, 1), (float)K(2, 2)};
    outputs.K.value = Karr;
    outputs.focal.value = {(float)fx, (float)fy};
    outputs.center.value = {(float)cx, (float)cy};
    outputs.distortion.value = {(float)k1, (float)k2};
    outputs.rms = (float)rms;
    outputs.solved = true;
  }
}
}
