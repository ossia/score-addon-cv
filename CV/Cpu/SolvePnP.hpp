#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

namespace cv
{
// M_PI is not provided by MSVC without _USE_MATH_DEFINES; use the portable
// std::numbers constant so the object builds on every standalone back-end.
inline constexpr double cv_pi = std::numbers::pi_v<double>;

// One 3D <-> 2D correspondence for the arbitrary-N list input of SolvePnP.
//
// cv.jit.unproject takes two Jitter matrices: an image-point matrix with 2 or
// cvjit::KEYPOINT_FIELD_COUNT (6) planes, and a reference-point matrix with 2, 3 or 6
// planes (a 2-plane or 6-plane reference matrix is read as z = 0, so a keypoint matrix can
// be plugged straight in). It requires n >= 4 and silently bails out when the two matrices
// disagree on n. Here both are interleaved into one list element, which makes the
// mismatched-count case unrepresentable; a keypoint source simply leaves oz at 0.
struct pnp_correspondence
{
  float ox, oy, oz; // 3D object / reference point
  float ix, iy;     // observed 2D image point, in pixels

  halp_field_names(ox, oy, oz, ix, iy);
};

// cv.jit.unproject's `format` attribute. Selects what the single "Rotation" list outlet
// carries; the dedicated Euler / Quaternion / Axis-angle / Matrix ports are always all
// emitted regardless (that is strictly nicer than cv.jit's one variable-size outlet).
enum class PnpRotationFormat
{
  Euler,      // 3 values, degrees, cv.jit's bank/heading/attitude convention
  Quaternion, // 4 values, (x, y, z, w) -- cv.jit's ordering
  AxisAngle,  // 4 values, (theta_degrees, ax, ay, az)
  Matrix      // 9 values, COLUMN-major -- see the note on outputs.rotation
};

// Estimator used to fit the pose.
enum class PnpMethod
{
  // Multi-seed Gauss-Newton over *all* correspondences. Default, and what this object has
  // always done; keeps existing patches bit-for-bit unchanged.
  LeastSquares,
  // Random sample consensus over minimal 4-point samples, then a least-squares refit on
  // the consensus set. cv.jit.unproject always calls cv::solvePnPRansac, hence the
  // matching iterationsCount / reprojectionError / confidence controls, but here it is
  // opt-in so the plain solve stays the default.
  RANSAC
};

// Camera pose estimation (cv.jit.unproject / solvePnP). Given N >= 4 3D object points and
// their observed 2D image projections plus the pinhole intrinsics (fx, fy, cx, cy),
// recovers the rotation R and translation t such that the projection of (R*X + t) through
// K matches the observed image points. A DLT-style linear estimate seeds an iterative
// Gauss-Newton refinement that minimises the reprojection error directly on SO(3) x R^3.
//
// Rotation is reported as Euler angles (degrees, xyz), a quaternion (xyzw), an axis-angle
// 4-vector (theta in degrees, then the unit axis) and a 3x3 rotation matrix, matching
// cv.jit.unproject's four output formats -- except that all four are emitted at once on
// separate ports instead of a single re-sized outlet. The "Rotation" list port reproduces
// cv.jit's single-outlet behaviour, sized and ordered per the Format selector.
//
// N is arbitrary (N >= 4) through the "Points" list input. When that list is empty the
// object falls back to the legacy 4 fixed Obj/Img spinbox sets.
struct SolvePnP
{
  halp_meta(name, "Solve PnP");
  halp_meta(c_name, "cv_solve_pnp");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Object pose (R,t) from N >= 4 3D-2D point correspondences, optionally with RANSAC. "
      "Outputs Euler, quaternion, axis-angle and a row-major 3x3 rotation matrix. Feed the "
      "Points list for arbitrary N; when it is empty the 4 Obj/Img spinbox sets are used.");
  halp_meta(uuid, "ec2c1b78-b677-4ddf-ab7e-5eaa8ccb6073");

  struct
  {
    halp::xyz_spinboxes_f32<"Obj 0"> obj0;
    halp::xyz_spinboxes_f32<"Obj 1"> obj1;
    halp::xyz_spinboxes_f32<"Obj 2"> obj2;
    halp::xyz_spinboxes_f32<"Obj 3"> obj3;
    halp::xy_spinboxes_f32<"Img 0"> img0;
    halp::xy_spinboxes_f32<"Img 1"> img1;
    halp::xy_spinboxes_f32<"Img 2"> img2;
    halp::xy_spinboxes_f32<"Img 3"> img3;
    halp::hslider_f32<"fx", halp::range{0.1f, 4000.f, 1.f}> fx;
    halp::hslider_f32<"fy", halp::range{0.1f, 4000.f, 1.f}> fy;
    halp::hslider_f32<"cx", halp::range{-4000.f, 4000.f, 0.f}> cx;
    halp::hslider_f32<"cy", halp::range{-4000.f, 4000.f, 0.f}> cy;

    // Arbitrary-N correspondence list. When non-empty it *replaces* the spinbox sets.
    struct
    {
      halp_meta(name, "Points");
      std::vector<pnp_correspondence> value;
    } points;

    // cv.jit.unproject's `format` attribute; drives the "Rotation" list port only.
    halp::enum_t<PnpRotationFormat, "Format"> format;

    // Default is LeastSquares: RANSAC is opt-in here (cv.jit always ransacs).
    halp::enum_t<PnpMethod, "Method"> method;
    // The three cv::solvePnPRansac parameters cv.jit.unproject hard-codes, with cv.jit's
    // exact defaults: iterationsCount = 100, reprojectionError = 8.0, confidence = 0.99.
    halp::hslider_i32<"Iterations count", halp::range{1, 10000, 100}> iterations_count;
    halp::hslider_f32<"Reprojection error", halp::range{0.f, 100.f, 8.f}>
        reprojection_error;
    halp::hslider_f32<"Confidence", halp::range{0.f, 1.f, 0.99f}> confidence;
  } inputs;

  struct
  {
    halp::val_port<"Euler", halp::xyz_type<float>> euler;
    halp::val_port<"Quaternion", std::array<float, 4>> quat;
    // ROW-major 3x3 rotation matrix: element [r*3+c] is R(r, c).
    // NOTE: cv.jit.unproject's matrix format outlet is COLUMN-major (it emits
    // R(0,0) R(1,0) R(2,0) R(0,1) ...). This port deliberately keeps the row-major
    // convention it has always had; the cv.jit-compatible column-major ordering is
    // available explicitly on the "Rotation" list port with Format = Matrix.
    halp::val_port<"Rotation matrix", std::array<float, 9>> matrix;
    halp::val_port<"Translation", halp::xyz_type<float>> translation;
    halp::val_port<"Valid", bool> valid;

    // (theta in DEGREES, ax, ay, az), computed exactly like cv.jit.unproject:
    // theta = 2*acos(w), axis = (x,y,z)/sqrt(1-w*w), and when theta is not FP_NORMAL
    // (zero rotation, or NaN from acos of a w that rounded past 1) theta is forced to 0
    // and the axis to cv.jit's (0, 0, 1) fallback.
    halp::val_port<"Axis-angle", std::array<float, 4>> axis_angle;

    // cv.jit.unproject's single `rotation` outlet: 3 values for Euler, 4 for Quaternion
    // and AxisAngle, 9 for Matrix -- and for Matrix it is COLUMN-major, byte-for-byte
    // what cv.jit emits, unlike the row-major "Rotation matrix" port above.
    struct
    {
      halp_meta(name, "Rotation");
      std::vector<float> value;
    } rotation;

    // Correspondences that agree with the reported pose. In LeastSquares mode every
    // correspondence is used, so this is N and the mask is all-ones; in RANSAC mode it is
    // the winning consensus set.
    halp::val_port<"Inliers", int> inliers;
    struct
    {
      halp_meta(name, "Inlier mask");
      std::vector<int> value; // 1 per input correspondence, in input order
    } inlier_mask;
  } outputs;

  void operator()() noexcept
  {
    std::vector<Eigen::Vector3d> obj;
    std::vector<Eigen::Vector2d> img;
    gather(obj, img);

    const int N = static_cast<int>(obj.size());
    if(N < 4)
    {
      // cv.jit.unproject quietly returns below 4 points; here Valid = false reports it.
      fail(N);
      return;
    }

    const double fx = inputs.fx.value, fy = inputs.fy.value;
    const double cx = inputs.cx.value, cy = inputs.cy.value;

    // Degenerate focal length -> nothing meaningful to do.
    if(!(std::abs(fx) > 1e-6) || !(std::abs(fy) > 1e-6))
    {
      fail(N);
      return;
    }

    // Normalised image coordinates (remove intrinsics): these are bearing directions
    // m = ((u-cx)/fx, (v-cy)/fy, 1) up to scale. Pose then solves m ~ R*X + t.
    std::vector<Eigen::Vector2d> norm(static_cast<std::size_t>(N));
    for(int i = 0; i < N; ++i)
      norm[static_cast<std::size_t>(i)] = Eigen::Vector2d{
          (img[static_cast<std::size_t>(i)].x() - cx) / fx,
          (img[static_cast<std::size_t>(i)].y() - cy) / fy};

    // Guard against degenerate (collinear) object points.
    if(collinear(obj))
    {
      fail(N);
      return;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    std::vector<std::uint8_t> mask;

    bool ok = false;
    if(inputs.method.value == PnpMethod::RANSAC)
    {
      ok = solveRansac(obj, norm, fx, fy, R, t, mask);
    }
    else
    {
      ok = solvePose(obj, norm, R, t, /* thorough */ true, fx);
      mask.assign(static_cast<std::size_t>(N), std::uint8_t{1});
    }

    if(!ok || !R.allFinite() || !t.allFinite())
    {
      fail(N);
      return;
    }

    emit(R, t, mask);
  }

private:
  // Collect the active correspondence set: the list input when non-empty, else the legacy
  // 4 spinbox sets.
  void gather(std::vector<Eigen::Vector3d>& obj, std::vector<Eigen::Vector2d>& img)
      const noexcept
  {
    const auto& list = inputs.points.value;
    if(!list.empty())
    {
      obj.reserve(list.size());
      img.reserve(list.size());
      for(const auto& c : list)
      {
        obj.emplace_back(c.ox, c.oy, c.oz);
        img.emplace_back(c.ix, c.iy);
      }
      return;
    }

    obj = {
        Eigen::Vector3d{inputs.obj0.value.x, inputs.obj0.value.y, inputs.obj0.value.z},
        Eigen::Vector3d{inputs.obj1.value.x, inputs.obj1.value.y, inputs.obj1.value.z},
        Eigen::Vector3d{inputs.obj2.value.x, inputs.obj2.value.y, inputs.obj2.value.z},
        Eigen::Vector3d{inputs.obj3.value.x, inputs.obj3.value.y, inputs.obj3.value.z}};
    img = {
        Eigen::Vector2d{inputs.img0.value.x, inputs.img0.value.y},
        Eigen::Vector2d{inputs.img1.value.x, inputs.img1.value.y},
        Eigen::Vector2d{inputs.img2.value.x, inputs.img2.value.y},
        Eigen::Vector2d{inputs.img3.value.x, inputs.img3.value.y}};
  }

  static bool collinear(const std::vector<Eigen::Vector3d>& p) noexcept
  {
    // Rank of the centered point cloud must be >= 2.
    const int n = static_cast<int>(p.size());
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for(auto& v : p)
      c += v;
    c /= static_cast<double>(n);
    Eigen::MatrixXd M(n, 3);
    for(int i = 0; i < n; ++i)
      M.row(i) = (p[static_cast<std::size_t>(i)] - c).transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
    const auto& s = svd.singularValues();
    // Need at least two appreciable singular values for a non-degenerate config.
    return s(1) < 1e-9 * (s(0) + 1e-12);
  }

  // Linear pose seed. Builds a homogeneous DLT system from m ~ [R|t] X and extracts an
  // orthonormal rotation via SVD, then resolves the global scale/sign from depth.
  static bool initialPose(
      const std::vector<Eigen::Vector3d>& obj, const std::vector<Eigen::Vector2d>& norm,
      Eigen::Matrix3d& R, Eigen::Vector3d& t) noexcept
  {
    // For each point: x*(r3.X+t3) = (r1.X+t1), y*(r3.X+t3) = (r2.X+t2).
    // Unknowns p = [r1 t1 r2 t2 r3 t3] (12). Solve A p = 0 via SVD null space.
    const int n = static_cast<int>(obj.size());
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 12);
    for(int i = 0; i < n; ++i)
    {
      const std::size_t k = static_cast<std::size_t>(i);
      const double X = obj[k].x(), Y = obj[k].y(), Z = obj[k].z();
      const double x = norm[k].x(), y = norm[k].y();
      // row: x*(r3.X+t3) - (r1.X+t1) = 0
      A.block<1, 4>(2 * i, 0) << -X, -Y, -Z, -1.0;
      A.block<1, 4>(2 * i, 8) << x * X, x * Y, x * Z, x;
      // row: y*(r3.X+t3) - (r2.X+t2) = 0
      A.block<1, 4>(2 * i + 1, 4) << -X, -Y, -Z, -1.0;
      A.block<1, 4>(2 * i + 1, 8) << y * X, y * Y, y * Z, y;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 12, 1> p = svd.matrixV().col(11);

    Eigen::Matrix3d M;
    M.row(0) = p.segment<3>(0).transpose();
    M.row(1) = p.segment<3>(4).transpose();
    M.row(2) = p.segment<3>(8).transpose();
    Eigen::Vector3d tt{p(3), p(7), p(11)};

    // Recover scale: M should be a scaled rotation. Use the mean row norm.
    const double scale = (M.row(0).norm() + M.row(1).norm() + M.row(2).norm()) / 3.0;
    if(!(scale > 1e-12))
      return false;
    M /= scale;
    tt /= scale;

    // Project M onto SO(3).
    Eigen::JacobiSVD<Eigen::Matrix3d> rsvd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = rsvd.matrixU() * rsvd.matrixV().transpose();
    if(R.determinant() < 0)
    {
      Eigen::Matrix3d V = rsvd.matrixV();
      V.col(2) *= -1.0;
      R = rsvd.matrixU() * V.transpose();
      tt = -tt;
    }
    t = tt;

    // Fix the overall sign so that points sit in front of the camera (positive depth).
    int positive = 0;
    for(const auto& v : obj)
      if((R * v + t).z() > 0)
        ++positive;
    if(positive * 2 < n)
    {
      R = -R;
      t = -t;
      // R may now have det -1; re-orthonormalise sign by flipping a column pair.
      if(R.determinant() < 0)
        R.col(2) *= -1.0;
    }
    return R.allFinite() && t.allFinite();
  }

  static Eigen::Matrix3d skew(const Eigen::Vector3d& w) noexcept
  {
    Eigen::Matrix3d S;
    S << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
    return S;
  }

  // Gauss-Newton on twist [dw(3), dt(3)] minimising the 2*N normalized reprojection
  // residuals r_i = (project(R*X+t) - m_i).
  static bool refine(
      const std::vector<Eigen::Vector3d>& obj, const std::vector<Eigen::Vector2d>& norm,
      Eigen::Matrix3d& R, Eigen::Vector3d& t) noexcept
  {
    const int n = static_cast<int>(obj.size());
    double lambda = 1e-3;
    double prevErr = residualNorm(obj, norm, R, t);
    for(int iter = 0; iter < 100; ++iter)
    {
      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

      for(int i = 0; i < n; ++i)
      {
        const std::size_t k = static_cast<std::size_t>(i);
        const Eigen::Vector3d Pc = R * obj[k] + t;
        const double Z = Pc.z();
        if(std::abs(Z) < 1e-9)
          return false;
        const double invZ = 1.0 / Z;
        const Eigen::Vector2d proj{Pc.x() * invZ, Pc.y() * invZ};
        const Eigen::Vector2d r = proj - norm[k];

        // d(proj)/d(Pc)
        Eigen::Matrix<double, 2, 3> dp;
        dp << invZ, 0, -Pc.x() * invZ * invZ, 0, invZ, -Pc.y() * invZ * invZ;

        // d(Pc)/d(twist): dPc = [-skew(Pc) | I] * [dw; dt]
        Eigen::Matrix<double, 3, 6> dPc;
        dPc.block<3, 3>(0, 0) = -skew(Pc);
        dPc.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

        const Eigen::Matrix<double, 2, 6> J = dp * dPc;
        H += J.transpose() * J;
        g += J.transpose() * r;
      }

      // Levenberg-Marquardt: keep raising the damping until a step decreases the error.
      bool stepTaken = false;
      double stepSize = 0.0;
      for(int tries = 0; tries < 40 && lambda < 1e12; ++tries)
      {
        Eigen::Matrix<double, 6, 6> Hd = H;
        Hd.diagonal() += lambda * H.diagonal();
        Eigen::Matrix<double, 6, 1> delta = Hd.ldlt().solve(-g);
        if(!delta.allFinite())
        {
          lambda *= 10.0;
          continue;
        }

        const Eigen::Vector3d dw = delta.segment<3>(0);
        const Eigen::Vector3d dt = delta.segment<3>(3);
        const double th = dw.norm();
        Eigen::Matrix3d Rd = Eigen::Matrix3d::Identity();
        if(th > 1e-12)
          Rd = Eigen::AngleAxisd(th, dw / th).toRotationMatrix();

        Eigen::Matrix3d Rn = Rd * R;
        Eigen::Vector3d tn = t + dt;
        const double err = residualNorm(obj, norm, Rn, tn);
        if(err < prevErr)
        {
          R = Rn;
          t = tn;
          prevErr = err;
          lambda = std::max(lambda * 0.3, 1e-12);
          stepTaken = true;
          stepSize = delta.norm();
          break;
        }
        else
        {
          lambda *= 10.0;
        }
      }

      if(!stepTaken)
        break; // damping exhausted: at a (local) minimum
      if(prevErr < 1e-18 || stepSize < 1e-12)
        break;
    }
    return R.allFinite() && t.allFinite();
  }

  // Robust pose solve: refine from several seeds (DLT estimate, its camera-axis mirror,
  // and a systematic sampling of orientation space at a few depths) and keep the
  // in-front-of-camera solution with the smallest reprojection residual. The multi-seed
  // strategy escapes the local minima / two-fold ambiguities of coplanar 4-point PnP.
  //
  // `thorough` selects the full seed grid. The reduced grid is used for RANSAC hypothesis
  // generation, where hundreds of minimal solves are run and only their inlier count
  // matters; the winning consensus set is always re-solved with the full grid afterwards.
  //
  // EARLY EXIT. The full grid is 2 DLT seeds + 6 axes x 9 angles x 3 depths = 164 seeds,
  // each running up to 100 damped Gauss-Newton iterations: 8-9 ms for 6 coplanar points,
  // 43 ms for N = 60, against the 16.7 ms a 60 fps node has for EVERYTHING. The grid is
  // NOT redundant -- it is what resolves the coplanar two-fold ambiguity -- so it stays,
  // but as soon as a refined seed reproduces every observation to better than
  // `pixelTolerance` pixels there is nothing left to find and the remaining seeds are
  // skipped. `fx` converts that pixel budget into the normalised units residualNorm
  // works in; pass 0 to disable the early exit and always sweep the whole grid.
  static constexpr double pixelTolerance = 0.05;

  static bool solvePose(
      const std::vector<Eigen::Vector3d>& obj, const std::vector<Eigen::Vector2d>& norm,
      Eigen::Matrix3d& R, Eigen::Vector3d& t, bool thorough, double fx) noexcept
  {
    const double n = static_cast<double>(obj.size());
    Eigen::Vector3d oc = Eigen::Vector3d::Zero();
    for(auto& v : obj)
      oc += v;
    oc /= n;
    double oscale = 0.0;
    for(auto& v : obj)
      oscale = std::max(oscale, (v - oc).norm());
    if(oscale < 1e-9)
      oscale = 1.0;

    std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> seeds;
    {
      Eigen::Matrix3d R0;
      Eigen::Vector3d t0;
      if(initialPose(obj, norm, R0, t0))
      {
        seeds.emplace_back(R0, t0);
        Eigen::Matrix3d Mir = Eigen::Matrix3d::Identity();
        Mir(2, 2) = -1.0;
        Eigen::Matrix3d Rm = Mir * R0;
        if(Rm.determinant() < 0)
          Rm.col(0) *= -1.0;
        seeds.emplace_back(Rm, t0);
      }
    }

    const std::array<Eigen::Vector3d, 6> axes{
        Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1),
        Eigen::Vector3d(1, 1, 0).normalized(), Eigen::Vector3d(1, 0, 1).normalized(),
        Eigen::Vector3d(0, 1, 1).normalized()};

    static constexpr std::array<double, 9> anglesFull{
        0.0, 30.0, -30.0, 60.0, -60.0, 90.0, 120.0, 150.0, 180.0};
    static constexpr std::array<double, 5> anglesFast{0.0, 60.0, -60.0, 120.0, 180.0};

    const std::array<double, 3> depthsFull{oscale * 3.0, oscale * 6.0, oscale * 12.0};
    const std::array<double, 1> depthsFast{oscale * 6.0};

    auto addSeeds = [&](const double* depths, int nd, const double* angs, int na) {
      for(int d = 0; d < nd; ++d)
      {
        for(const Eigen::Vector3d& axis : axes)
        {
          for(int a = 0; a < na; ++a)
          {
            Eigen::Matrix3d Rs0
                = Eigen::AngleAxisd(angs[a] * cv_pi / 180.0, axis).toRotationMatrix();
            seeds.emplace_back(Rs0, Eigen::Vector3d(0, 0, depths[d]) - Rs0 * oc);
          }
        }
      }
    };

    if(thorough)
      addSeeds(depthsFull.data(), 3, anglesFull.data(), 9);
    else
      addSeeds(depthsFast.data(), 1, anglesFast.data(), 5);

    // Residual (in the normalised units residualNorm returns) below which a seed is
    // considered to have found the answer: `pixelTolerance` px on every one of the n
    // points. A non-positive fx disables the early exit.
    const double goodEnough
        = (fx > 0.0) ? (n * (pixelTolerance / fx) * (pixelTolerance / fx)) : -1.0;

    double best = std::numeric_limits<double>::infinity();
    bool found = false;
    for(auto& [Rs, ts] : seeds)
    {
      Eigen::Matrix3d Rr = Rs;
      Eigen::Vector3d tr = ts;
      if(!refine(obj, norm, Rr, tr))
        continue;
      if(tr.z() < 0.)
      {
        Rr = -Rr;
        tr = -tr;
      }
      bool inFront = true;
      for(auto& v : obj)
        if((Rr * v + tr).z() <= 0)
          inFront = false;
      if(!inFront)
        continue;
      const double e = residualNorm(obj, norm, Rr, tr);
      if(e < best && Rr.allFinite() && tr.allFinite())
      {
        best = e;
        R = Rr;
        t = tr;
        found = true;
        if(best <= goodEnough)
          break; // exact fit: no other seed can beat it, stop sweeping the grid
      }
    }
    return found;
  }

  static double residualNorm(
      const std::vector<Eigen::Vector3d>& obj, const std::vector<Eigen::Vector2d>& norm,
      const Eigen::Matrix3d& R, const Eigen::Vector3d& t) noexcept
  {
    double e = 0.0;
    const int n = static_cast<int>(obj.size());
    for(int i = 0; i < n; ++i)
    {
      const std::size_t k = static_cast<std::size_t>(i);
      const Eigen::Vector3d Pc = R * obj[k] + t;
      const double Z = Pc.z();
      if(std::abs(Z) < 1e-9)
        return std::numeric_limits<double>::infinity();
      const Eigen::Vector2d proj{Pc.x() / Z, Pc.y() / Z};
      e += (proj - norm[k]).squaredNorm();
    }
    return e;
  }

  // Squared reprojection error of one correspondence, in PIXELS: the residual is computed
  // in normalised coordinates and scaled back by (fx, fy), so it is directly comparable to
  // cv::solvePnPRansac's `reprojectionError` (which cv.jit.unproject hard-codes to 8.0).
  static double pixelError2(
      const Eigen::Vector3d& P, const Eigen::Vector2d& m, const Eigen::Matrix3d& R,
      const Eigen::Vector3d& t, double fx, double fy) noexcept
  {
    const Eigen::Vector3d Pc = R * P + t;
    const double Z = Pc.z();
    if(!(std::abs(Z) > 1e-9))
      return std::numeric_limits<double>::infinity();
    const double ex = fx * (Pc.x() / Z - m.x());
    const double ey = fy * (Pc.y() / Z - m.y());
    const double e = ex * ex + ey * ey;
    return std::isfinite(e) ? e : std::numeric_limits<double>::infinity();
  }

  // RANSAC with cv::solvePnPRansac's parameter semantics: `iterations_count` bounds the
  // number of minimal samples, `reprojection_error` is the pixel inlier threshold and
  // `confidence` drives the usual adaptive early stop. The RNG is seeded per call with a
  // fixed constant so identical inputs always give identical outputs.
  bool solveRansac(
      const std::vector<Eigen::Vector3d>& obj, const std::vector<Eigen::Vector2d>& norm,
      double fx, double fy, Eigen::Matrix3d& R, Eigen::Vector3d& t,
      std::vector<std::uint8_t>& mask) const noexcept
  {
    const int N = static_cast<int>(obj.size());
    const double thr = std::max(0.f, inputs.reprojection_error.value);
    const double thr2 = thr * thr;
    const int maxIters = std::max(1, inputs.iterations_count.value);
    const double conf
        = std::min(0.999999, std::max(0.0, double(inputs.confidence.value)));

    std::mt19937 rng(0x51e3d0eeu);
    std::vector<int> perm(static_cast<std::size_t>(N));
    for(int i = 0; i < N; ++i)
      perm[static_cast<std::size_t>(i)] = i;

    std::vector<Eigen::Vector3d> so(4);
    std::vector<Eigen::Vector2d> sn(4);
    std::vector<std::uint8_t> cur(static_cast<std::size_t>(N));
    std::vector<std::uint8_t> best;
    int bestCount = 0;
    Eigen::Matrix3d bestR = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bestT = Eigen::Vector3d::Zero();
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
        const std::size_t idx
            = static_cast<std::size_t>(perm[static_cast<std::size_t>(k)]);
        so[static_cast<std::size_t>(k)] = obj[idx];
        sn[static_cast<std::size_t>(k)] = norm[idx];
      }

      if(collinear(so))
        continue;

      Eigen::Matrix3d Rc;
      Eigen::Vector3d tc;
      if(!solvePose(so, sn, Rc, tc, /* thorough */ false, fx))
        continue;

      int count = 0;
      for(int i = 0; i < N; ++i)
      {
        const std::size_t k = static_cast<std::size_t>(i);
        const bool in = pixelError2(obj[k], norm[k], Rc, tc, fx, fy) <= thr2;
        cur[k] = in ? 1 : 0;
        count += in ? 1 : 0;
      }

      if(count > bestCount)
      {
        bestCount = count;
        best = cur;
        bestR = Rc;
        bestT = tc;
        haveBest = true;

        if(bestCount >= N)
          break;
        // Adaptive stop: with an inlier ratio w, the chance that a 4-sample is all-inlier
        // is w^4, so log(1-conf)/log(1-w^4) samples suffice for the requested confidence.
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

    // Refit on the consensus set with the full seed grid.
    std::vector<Eigen::Vector3d> io;
    std::vector<Eigen::Vector2d> in;
    io.reserve(static_cast<std::size_t>(bestCount));
    in.reserve(static_cast<std::size_t>(bestCount));
    for(int i = 0; i < N; ++i)
      if(best[static_cast<std::size_t>(i)])
      {
        io.push_back(obj[static_cast<std::size_t>(i)]);
        in.push_back(norm[static_cast<std::size_t>(i)]);
      }

    Eigen::Matrix3d Rr;
    Eigen::Vector3d tr;
    if(!collinear(io) && solvePose(io, in, Rr, tr, /* thorough */ true, fx) && Rr.allFinite()
       && tr.allFinite())
    {
      int count = 0;
      std::vector<std::uint8_t> m2(static_cast<std::size_t>(N));
      for(int i = 0; i < N; ++i)
      {
        const std::size_t k = static_cast<std::size_t>(i);
        const bool inl = pixelError2(obj[k], norm[k], Rr, tr, fx, fy) <= thr2;
        m2[k] = inl ? 1 : 0;
        count += inl ? 1 : 0;
      }
      if(count >= bestCount)
      {
        R = Rr;
        t = tr;
        mask = std::move(m2);
        return true;
      }
    }

    R = bestR;
    t = bestT;
    mask = std::move(best);
    return true;
  }

  // Quaternion (x,y,z,w) from a rotation matrix, matching cv.jit.unproject's ordering.
  static std::array<double, 4> toQuaternion(const Eigen::Matrix3d& m) noexcept
  {
    const double trace = m(0, 0) + m(1, 1) + m(2, 2);
    double w, x, y, z;
    if(trace > 0.0)
    {
      const double s = std::sqrt(trace + 1.0) * 2.0;
      w = 0.25 * s;
      x = (m(2, 1) - m(1, 2)) / s;
      y = (m(0, 2) - m(2, 0)) / s;
      z = (m(1, 0) - m(0, 1)) / s;
    }
    else if((m(0, 0) > m(1, 1)) && (m(0, 0) > m(2, 2)))
    {
      const double s = std::sqrt(1.0 + m(0, 0) - m(1, 1) - m(2, 2)) * 2.0;
      w = (m(2, 1) - m(1, 2)) / s;
      x = 0.25 * s;
      y = (m(0, 1) + m(1, 0)) / s;
      z = (m(0, 2) + m(2, 0)) / s;
    }
    else if(m(1, 1) > m(2, 2))
    {
      const double s = std::sqrt(1.0 + m(1, 1) - m(0, 0) - m(2, 2)) * 2.0;
      w = (m(0, 2) - m(2, 0)) / s;
      x = (m(0, 1) + m(1, 0)) / s;
      y = 0.25 * s;
      z = (m(1, 2) + m(2, 1)) / s;
    }
    else
    {
      const double s = std::sqrt(1.0 + m(2, 2) - m(0, 0) - m(1, 1)) * 2.0;
      w = (m(1, 0) - m(0, 1)) / s;
      x = (m(0, 2) + m(2, 0)) / s;
      y = (m(1, 2) + m(2, 1)) / s;
      z = 0.25 * s;
    }
    return {x, y, z, w};
  }

  // Axis-angle, EXACTLY cv.jit.unproject's formulation (including its quirks):
  //   theta = 2*acos(w) in radians, reported in DEGREES;
  //   axis  = (x,y,z) / sqrt(1 - w*w);
  //   if theta is not FP_NORMAL -- i.e. it is exactly 0 (w == 1), subnormal, or NaN
  //   (w rounded just past 1) -- theta is forced to 0 and the axis left at (0, 0, 1).
  // That last branch is what keeps the near-identity case free of NaN/inf: it is not an
  // accident, so do not "fix" it into a small-angle expansion.
  static std::array<double, 4> toAxisAngle(const std::array<double, 4>& q) noexcept
  {
    std::array<double, 3> axis{0.0, 0.0, 1.0};
    double theta = 2.0 * std::acos(q[3]);
    if(std::fpclassify(theta) != FP_NORMAL)
    {
      theta = 0.0;
    }
    else
    {
      const double a = std::sqrt(1.0 - q[3] * q[3]);
      axis[0] = q[0] / a;
      axis[1] = q[1] / a;
      axis[2] = q[2] / a;
      // A non-normal 1/a can still poison the axis if w is within an ulp of 1 while
      // theta stays normal; fall back to cv.jit's default axis rather than emit NaN.
      if(!(std::isfinite(axis[0]) && std::isfinite(axis[1]) && std::isfinite(axis[2])))
      {
        theta = 0.0;
        axis = {0.0, 0.0, 1.0};
      }
    }
    return {theta * 180.0 / cv_pi, axis[0], axis[1], axis[2]};
  }

  void emit(
      const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
      const std::vector<std::uint8_t>& mask) noexcept
  {
    const std::array<double, 4> q = toQuaternion(R);

    outputs.quat.value
        = {static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]),
           static_cast<float>(q[3])};

    // Euler angles (degrees), same heading/attitude/bank convention as cv.jit.unproject:
    // quat ordering is (x,y,z,w) = (q[0],q[1],q[2],q[3]).
    constexpr double rad2deg = 180.0 / cv_pi;
    const double a = q[0] * q[1] + q[2] * q[3];
    double rx, ry, rz;
    if(a > 0.4999)
    {
      rx = 0.0;
      ry = rad2deg * (2.0 * std::atan2(q[0], q[3]));
      rz = 90.0;
    }
    else if(a < -0.4999)
    {
      rx = 0.0;
      ry = rad2deg * (-2.0 * std::atan2(q[0], q[3]));
      rz = -90.0;
    }
    else
    {
      const double xx = q[0] * q[0];
      const double yy = q[1] * q[1];
      const double zz = q[2] * q[2];
      rx = rad2deg
           * std::atan2(2.0 * q[0] * q[3] - 2.0 * q[1] * q[2], 1.0 - 2.0 * xx - 2.0 * zz);
      ry = rad2deg
           * std::atan2(2.0 * q[1] * q[3] - 2.0 * q[0] * q[2], 1.0 - 2.0 * yy - 2.0 * zz);
      rz = rad2deg * std::asin(2.0 * a);
    }

    outputs.euler.value
        = {static_cast<float>(rx), static_cast<float>(ry), static_cast<float>(rz)};

    // Row-major 3x3 rotation matrix (cv.jit's own matrix outlet is column-major; see the
    // note on the port declaration).
    std::array<float, 9> rm;
    for(int row = 0; row < 3; ++row)
      for(int col = 0; col < 3; ++col)
        rm[row * 3 + col] = static_cast<float>(R(row, col));
    outputs.matrix.value = rm;

    const std::array<double, 4> aa = toAxisAngle(q);
    outputs.axis_angle.value
        = {static_cast<float>(aa[0]), static_cast<float>(aa[1]),
           static_cast<float>(aa[2]), static_cast<float>(aa[3])};

    outputs.translation.value
        = {static_cast<float>(t.x()), static_cast<float>(t.y()),
           static_cast<float>(t.z())};
    outputs.valid = true;

    emitRotationList(rx, ry, rz, q, aa, R);

    int count = 0;
    outputs.inlier_mask.value.resize(mask.size());
    for(std::size_t i = 0; i < mask.size(); ++i)
    {
      const int v = mask[i] ? 1 : 0;
      outputs.inlier_mask.value[i] = v;
      count += v;
    }
    outputs.inliers = count;
  }

  // cv.jit.unproject's single `rotation` outlet, sized and ordered by the format attribute.
  void emitRotationList(
      double rx, double ry, double rz, const std::array<double, 4>& q,
      const std::array<double, 4>& aa, const Eigen::Matrix3d& R) noexcept
  {
    auto& v = outputs.rotation.value;
    switch(inputs.format.value)
    {
      case PnpRotationFormat::Quaternion:
        v = {static_cast<float>(q[0]), static_cast<float>(q[1]),
             static_cast<float>(q[2]), static_cast<float>(q[3])};
        break;
      case PnpRotationFormat::AxisAngle:
        v = {static_cast<float>(aa[0]), static_cast<float>(aa[1]),
             static_cast<float>(aa[2]), static_cast<float>(aa[3])};
        break;
      case PnpRotationFormat::Matrix:
        // COLUMN-major, byte-for-byte cv.jit.unproject's ordering:
        // R(0,0) R(1,0) R(2,0) R(0,1) R(1,1) R(2,1) R(0,2) R(1,2) R(2,2).
        v.resize(9);
        for(int col = 0; col < 3; ++col)
          for(int row = 0; row < 3; ++row)
            v[static_cast<std::size_t>(col * 3 + row)] = static_cast<float>(R(row, col));
        break;
      case PnpRotationFormat::Euler:
      default:
        v = {static_cast<float>(rx), static_cast<float>(ry), static_cast<float>(rz)};
        break;
    }
  }

  void fail(int n) noexcept
  {
    outputs.euler.value = {0.f, 0.f, 0.f};
    outputs.quat.value = {0.f, 0.f, 0.f, 1.f};
    outputs.matrix.value = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    outputs.translation.value = {0.f, 0.f, 0.f};
    outputs.valid = false;
    // cv.jit's zero-rotation axis-angle: theta = 0 with its default (0,0,1) axis.
    outputs.axis_angle.value = {0.f, 0.f, 0.f, 1.f};
    emitRotationList(
        0.0, 0.0, 0.0, {0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 1.0},
        Eigen::Matrix3d::Identity());
    outputs.inliers = 0;
    outputs.inlier_mask.value.assign(static_cast<std::size_t>(std::max(0, n)), 0);
  }
};
}
