#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace cv
{
// Homography / perspective transform (cv.jit.getperspective). Computes the 3x3 matrix
// mapping source points to destination points via the Direct Linear Transform, solved as
// the null vector of a 2N x 9 system with Eigen SVD. With exactly 4 correspondences this is
// the classic exactly-determined DLT; with more it is an over-determined least-squares fit.
// Outputs the 9 coefficients (row-major) usable as a projective warp. Pure Eigen.
//
// The first 4 source/destination pairs are always used. Four further optional pairs
// (Src 4..7 / Dst 4..7) are included only when their matching "Use point N" toggle is on,
// allowing up to 8 correspondences with a least-squares solve. N is therefore currently
// capped at the 8 provided point-pair inputs; a real arbitrary-N list input is a future
// extension. (A toggle is required because the spinboxes default to a non-zero value, so a
// "non-zero" sentinel cannot reliably distinguish unused pairs.)
struct Homography
{
  halp_meta(name, "Homography (4-8 points)");
  halp_meta(c_name, "cv_homography");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "3x3 perspective transform from 4 to 8 point correspondences (least-squares DLT). "
      "The first 4 pairs are always used; Src/Dst 4..7 are used when the source is "
      "non-zero. N is capped at the 8 provided pairs (a list input is a future extension).");
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
  } inputs;

  struct
  {
    halp::val_port<"Matrix", std::array<float, 9>> matrix;
    halp::val_port<"Valid", bool> valid;
  } outputs;

  void operator()() noexcept
  {
    using V2 = Eigen::Vector2f;
    // The first 4 correspondences are mandatory; the next 4 are optional and only used
    // when their source point is non-zero (the spinboxes default to (0,0)).
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
        true, true, true, true, inputs.use4.value, inputs.use5.value,
        inputs.use6.value, inputs.use7.value};

    std::vector<V2> src, dst;
    src.reserve(8);
    dst.reserve(8);
    for(int i = 0; i < 8; ++i)
    {
      // Always include the first 4; include the rest only when their toggle is on.
      if(used[i])
      {
        src.push_back(srcAll[i]);
        dst.push_back(dstAll[i]);
      }
    }

    const int N = static_cast<int>(src.size());

    // Hartley normalisation: translate each point set to its centroid and scale so the mean
    // distance to the origin is sqrt(2). This conditions the DLT and is what makes the
    // over-determined (N>4) least-squares solve numerically accurate. Computed in double.
    auto normaliser = [N](const std::vector<V2>& pts, Eigen::Matrix3d& T) {
      Eigen::Vector2d c = Eigen::Vector2d::Zero();
      for(int i = 0; i < N; ++i)
        c += Eigen::Vector2d{pts[i].x(), pts[i].y()};
      c /= N;
      double meanDist = 0.0;
      for(int i = 0; i < N; ++i)
        meanDist += (Eigen::Vector2d{pts[i].x(), pts[i].y()} - c).norm();
      meanDist /= N;
      const double s = (meanDist > 1e-12) ? (std::sqrt(2.0) / meanDist) : 1.0;
      T.setIdentity();
      T(0, 0) = s;
      T(1, 1) = s;
      T(0, 2) = -s * c.x();
      T(1, 2) = -s * c.y();
    };

    Eigen::Matrix3d Ts, Td;
    normaliser(src, Ts);
    normaliser(dst, Td);

    // Build the 2N x 9 DLT matrix from the normalised correspondences: two rows each. With
    // N=4 this is the classic exactly-determined 8x9 system; with N>4 it is over-determined
    // and the SVD null-vector yields the least-squares homography.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * N, 9);
    for(int i = 0; i < N; ++i)
    {
      const Eigen::Vector3d sn = Ts * Eigen::Vector3d{src[i].x(), src[i].y(), 1.0};
      const Eigen::Vector3d dn = Td * Eigen::Vector3d{dst[i].x(), dst[i].y(), 1.0};
      const double x = sn.x(), y = sn.y();
      const double u = dn.x(), v = dn.y();
      A.row(2 * i) << -x, -y, -1.0, 0, 0, 0, u * x, u * y, u;
      A.row(2 * i + 1) << 0, 0, 0, -x, -y, -1.0, v * x, v * y, v;
    }

    // Null vector = right singular vector of smallest singular value (last column of V).
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> hv = svd.matrixV().col(8);

    // Denormalise: H = Td^-1 * Hn * Ts.
    Eigen::Matrix3d Hn;
    Hn << hv(0), hv(1), hv(2), hv(3), hv(4), hv(5), hv(6), hv(7), hv(8);
    Eigen::Matrix3d Hd = Td.inverse() * Hn * Ts;

    if(std::abs(Hd(2, 2)) > 1e-12)
      Hd /= Hd(2, 2); // normalise so H(2,2)=1

    std::array<float, 9> m;
    for(int r = 0; r < 3; ++r)
      for(int c = 0; c < 3; ++c)
        m[r * 3 + c] = static_cast<float>(Hd(r, c));
    outputs.matrix = m;
    outputs.valid = Hd.allFinite() && std::abs(Hd(2, 2)) > 1e-12;
  }
};
}
