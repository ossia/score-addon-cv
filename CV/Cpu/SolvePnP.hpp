#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace cv
{
// Camera pose estimation (cv.jit.unproject / solvePnP). Given four 3D object points and
// their observed 2D image projections plus the pinhole intrinsics (fx, fy, cx, cy),
// recovers the rotation R and translation t such that the projection of (R*X + t) through
// K matches the observed image points. A DLT-style linear estimate seeds an iterative
// Gauss-Newton refinement that minimises the reprojection error directly on SO(3) x R^3.
// Rotation is reported as Euler angles (degrees, xyz), a quaternion (xyzw), and a row-major
// 3x3 rotation matrix (cv.jit.unproject has a matrix output mode), matching
// cv.jit.unproject's output conventions. Pure Eigen, no OpenCV.
//
// LIMITATION: this is a fixed 4-point solver. A general N-point PnP would require a list
// input for the correspondences, which the fixed spinbox ports cannot represent yet; an
// N-point list input is a future extension.
struct SolvePnP
{
  halp_meta(name, "Solve PnP");
  halp_meta(c_name, "cv_solve_pnp");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Object pose (R,t) from exactly 4 3D-2D point correspondences. Outputs Euler, "
      "quaternion and a row-major 3x3 rotation matrix. 4-point only: N-point would need "
      "a list input the port cannot take yet.");
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
  } inputs;

  struct
  {
    halp::val_port<"Euler", halp::xyz_type<float>> euler;
    halp::val_port<"Quaternion", std::array<float, 4>> quat;
    // Row-major 3x3 rotation matrix (cv.jit.unproject matrix output mode).
    halp::val_port<"Rotation matrix", std::array<float, 9>> matrix;
    halp::val_port<"Translation", halp::xyz_type<float>> translation;
    halp::val_port<"Valid", bool> valid;
  } outputs;

  void operator()() noexcept
  {
    const std::array<Eigen::Vector3d, 4> obj{
        Eigen::Vector3d{inputs.obj0.value.x, inputs.obj0.value.y, inputs.obj0.value.z},
        Eigen::Vector3d{inputs.obj1.value.x, inputs.obj1.value.y, inputs.obj1.value.z},
        Eigen::Vector3d{inputs.obj2.value.x, inputs.obj2.value.y, inputs.obj2.value.z},
        Eigen::Vector3d{inputs.obj3.value.x, inputs.obj3.value.y, inputs.obj3.value.z}};
    const std::array<Eigen::Vector2d, 4> img{
        Eigen::Vector2d{inputs.img0.value.x, inputs.img0.value.y},
        Eigen::Vector2d{inputs.img1.value.x, inputs.img1.value.y},
        Eigen::Vector2d{inputs.img2.value.x, inputs.img2.value.y},
        Eigen::Vector2d{inputs.img3.value.x, inputs.img3.value.y}};

    const double fx = inputs.fx.value, fy = inputs.fy.value;
    const double cx = inputs.cx.value, cy = inputs.cy.value;

    // Degenerate focal length -> nothing meaningful to do.
    if(!(std::abs(fx) > 1e-6) || !(std::abs(fy) > 1e-6))
    {
      fail();
      return;
    }

    // Normalised image coordinates (remove intrinsics): these are bearing directions
    // m = ((u-cx)/fx, (v-cy)/fy, 1) up to scale. Pose then solves m ~ R*X + t.
    std::array<Eigen::Vector2d, 4> norm;
    for(int i = 0; i < 4; ++i)
      norm[i] = Eigen::Vector2d{(img[i].x() - cx) / fx, (img[i].y() - cy) / fy};

    // Guard against degenerate (collinear) object points.
    if(collinear(obj))
    {
      fail();
      return;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    if(!solvePose(obj, norm, R, t))
    {
      fail();
      return;
    }

    if(!R.allFinite() || !t.allFinite())
    {
      fail();
      return;
    }

    emit(R, t);
  }

private:
  static bool collinear(const std::array<Eigen::Vector3d, 4>& p) noexcept
  {
    // Rank of the centered point cloud must be >= 2.
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for(auto& v : p)
      c += v;
    c /= 4.0;
    Eigen::Matrix<double, 4, 3> M;
    for(int i = 0; i < 4; ++i)
      M.row(i) = (p[i] - c).transpose();
    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 3>> svd(M);
    const auto& s = svd.singularValues();
    // Need at least two appreciable singular values for a non-degenerate config.
    return s(1) < 1e-9 * (s(0) + 1e-12);
  }

  // Linear pose seed. Builds a homogeneous DLT system from m ~ [R|t] X and extracts an
  // orthonormal rotation via SVD, then resolves the global scale/sign from depth.
  static bool initialPose(
      const std::array<Eigen::Vector3d, 4>& obj,
      const std::array<Eigen::Vector2d, 4>& norm, Eigen::Matrix3d& R,
      Eigen::Vector3d& t) noexcept
  {
    // For each point: x*(r3.X+t3) = (r1.X+t1), y*(r3.X+t3) = (r2.X+t2).
    // Unknowns p = [r1 t1 r2 t2 r3 t3] (12). Solve A p = 0 via SVD null space.
    Eigen::Matrix<double, 8, 12> A = Eigen::Matrix<double, 8, 12>::Zero();
    for(int i = 0; i < 4; ++i)
    {
      const double X = obj[i].x(), Y = obj[i].y(), Z = obj[i].z();
      const double x = norm[i].x(), y = norm[i].y();
      // row: x*(r3.X+t3) - (r1.X+t1) = 0
      A.block<1, 4>(2 * i, 0) << -X, -Y, -Z, -1.0;
      A.block<1, 4>(2 * i, 8) << x * X, x * Y, x * Z, x;
      // row: y*(r3.X+t3) - (r2.X+t2) = 0
      A.block<1, 4>(2 * i + 1, 4) << -X, -Y, -Z, -1.0;
      A.block<1, 4>(2 * i + 1, 8) << y * X, y * Y, y * Z, y;
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 8, 12>> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 12, 1> p = svd.matrixV().col(11);

    Eigen::Matrix3d M;
    M.row(0) = p.segment<3>(0).transpose();
    M.row(1) = p.segment<3>(4).transpose();
    M.row(2) = p.segment<3>(8).transpose();
    Eigen::Vector3d tt{p(3), p(7), p(11)};

    // Recover scale: M should be a scaled rotation. Use the mean row norm.
    const double scale
        = (M.row(0).norm() + M.row(1).norm() + M.row(2).norm()) / 3.0;
    if(!(scale > 1e-12))
      return false;
    M /= scale;
    tt /= scale;

    // Project M onto SO(3).
    Eigen::JacobiSVD<Eigen::Matrix3d> rsvd(
        M, Eigen::ComputeFullU | Eigen::ComputeFullV);
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
    for(int i = 0; i < 4; ++i)
      if((R * obj[i] + t).z() > 0)
        ++positive;
    if(positive < 2)
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
      const std::array<Eigen::Vector3d, 4>& obj,
      const std::array<Eigen::Vector2d, 4>& norm, Eigen::Matrix3d& R,
      Eigen::Vector3d& t) noexcept
  {
    double lambda = 1e-3;
    double prevErr = residualNorm(obj, norm, R, t);
    for(int iter = 0; iter < 100; ++iter)
    {
      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

      for(int i = 0; i < 4; ++i)
      {
        const Eigen::Vector3d Pc = R * obj[i] + t;
        const double Z = Pc.z();
        if(std::abs(Z) < 1e-9)
          return false;
        const double invZ = 1.0 / Z;
        const Eigen::Vector2d proj{Pc.x() * invZ, Pc.y() * invZ};
        const Eigen::Vector2d r = proj - norm[i];

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
  static bool solvePose(
      const std::array<Eigen::Vector3d, 4>& obj,
      const std::array<Eigen::Vector2d, 4>& norm, Eigen::Matrix3d& R,
      Eigen::Vector3d& t) noexcept
  {
    Eigen::Vector3d oc = Eigen::Vector3d::Zero();
    for(auto& v : obj)
      oc += v;
    oc /= 4.0;
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
        Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
        Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 1, 0).normalized(),
        Eigen::Vector3d(1, 0, 1).normalized(),
        Eigen::Vector3d(0, 1, 1).normalized()};
    for(double depth : {oscale * 3.0, oscale * 6.0, oscale * 12.0})
    {
      for(const Eigen::Vector3d& axis : axes)
      {
        for(double ang :
            {0.0, 30.0, -30.0, 60.0, -60.0, 90.0, 120.0, 150.0, 180.0})
        {
          Eigen::Matrix3d Rs0
              = Eigen::AngleAxisd(ang * M_PI / 180.0, axis).toRotationMatrix();
          seeds.emplace_back(Rs0, Eigen::Vector3d(0, 0, depth) - Rs0 * oc);
        }
      }
    }

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
      }
    }
    return found;
  }

  static double residualNorm(
      const std::array<Eigen::Vector3d, 4>& obj,
      const std::array<Eigen::Vector2d, 4>& norm, const Eigen::Matrix3d& R,
      const Eigen::Vector3d& t) noexcept
  {
    double e = 0.0;
    for(int i = 0; i < 4; ++i)
    {
      const Eigen::Vector3d Pc = R * obj[i] + t;
      const double Z = Pc.z();
      if(std::abs(Z) < 1e-9)
        return std::numeric_limits<double>::infinity();
      const Eigen::Vector2d proj{Pc.x() / Z, Pc.y() / Z};
      e += (proj - norm[i]).squaredNorm();
    }
    return e;
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

  void emit(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) noexcept
  {
    const std::array<double, 4> q = toQuaternion(R);

    outputs.quat.value = {
        static_cast<float>(q[0]), static_cast<float>(q[1]),
        static_cast<float>(q[2]), static_cast<float>(q[3])};

    // Euler angles (degrees), same heading/attitude/bank convention as cv.jit.unproject:
    // quat ordering is (x,y,z,w) = (q[0],q[1],q[2],q[3]).
    constexpr double rad2deg = 180.0 / M_PI;
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
           * std::atan2(
               2.0 * q[0] * q[3] - 2.0 * q[1] * q[2], 1.0 - 2.0 * xx - 2.0 * zz);
      ry = rad2deg
           * std::atan2(
               2.0 * q[1] * q[3] - 2.0 * q[0] * q[2], 1.0 - 2.0 * yy - 2.0 * zz);
      rz = rad2deg * std::asin(2.0 * a);
    }

    outputs.euler.value
        = {static_cast<float>(rx), static_cast<float>(ry), static_cast<float>(rz)};

    // Row-major 3x3 rotation matrix.
    std::array<float, 9> rm;
    for(int row = 0; row < 3; ++row)
      for(int col = 0; col < 3; ++col)
        rm[row * 3 + col] = static_cast<float>(R(row, col));
    outputs.matrix.value = rm;

    outputs.translation.value = {
        static_cast<float>(t.x()), static_cast<float>(t.y()),
        static_cast<float>(t.z())};
    outputs.valid = true;
  }

  void fail() noexcept
  {
    outputs.euler.value = {0.f, 0.f, 0.f};
    outputs.quat.value = {0.f, 0.f, 0.f, 1.f};
    outputs.matrix.value = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    outputs.translation.value = {0.f, 0.f, 0.f};
    outputs.valid = false;
  }
};
}
