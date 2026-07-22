#include "BlobStats.hpp"

#include <CV/Cpu/ConnectedComponents.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cv
{
namespace
{
constexpr double PI = 3.14159265358979323846;

// Per-blob raw moment accumulators (cv.jit.blobs.moments accumulates exactly these).
// Kept as doubles: every increment is an exact small integer, so up to 2^53 the sums are
// bit-exact, which matters for the `nu20 == nu02` equality test below.
struct Acc
{
  double m00 = 0;                    // pixel count
  double m10 = 0, m01 = 0;           // sum x, sum y
  double m20 = 0, m02 = 0, m11 = 0;  // second-order raw moments
  double m30 = 0, m03 = 0, m21 = 0, m12 = 0; // third-order raw moments
  int minx = 0, miny = 0, maxx = 0, maxy = 0;
  bool init = false;
};

// cv.jit.blobs.direction's quadrant-banded sign test on the 3rd-order normalised moments.
// Transcribed 1:1 from cv.jit/source/projects/cv.jit.blobs.direction.
double cvjit_direction(double theta, double nu30, double nu03) noexcept
{
  if(theta > PI * 0.25 && theta < PI * 0.75)
  {
    if(nu03 < 0)
      theta += PI;
  }
  else if(theta > PI * 0.75)
  {
    if(nu30 > 0)
      theta += PI;
  }
  else
  {
    if(nu30 < 0)
      theta += PI;
  }
  return theta;
}

// cv.jit.blobs.orientation. NOTE: atan (not atan2) plus explicit quadrant fix-ups, and an
// exact-zero result when the two second-order moments are equal. Range [0, pi).
double cvjit_orientation(double nu20, double nu02, double nu11) noexcept
{
  const double d = nu20 - nu02;
  if(d == 0.0)
    return 0.0;

  const double c = std::atan((nu11 * 2.0) / d) * 0.5;
  if(nu20 > nu02)
    return (c < 0.0) ? (c + PI) : c;
  return c + PI * 0.5;
}
}

void BlobStats::operator()() noexcept
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

  outputs.blobs.value.clear();
  outputs.count = R.count;
  if(R.count == 0)
    return;

  std::vector<Acc> acc(static_cast<std::size_t>(R.count) + 1);

  for(int y = 0; y < H; ++y)
  {
    const double dy = y;
    const double dyy = dy * dy;
    for(int x = 0; x < W; ++x)
    {
      std::int32_t l = R.labels[static_cast<std::size_t>(y) * W + x];
      if(l == 0)
        continue;
      Acc& a = acc[static_cast<std::size_t>(l)];
      const double dx = x;
      const double dxx = dx * dx;
      a.m00 += 1;
      a.m10 += dx;
      a.m01 += dy;
      a.m20 += dxx;
      a.m02 += dyy;
      a.m11 += dx * dy;
      a.m30 += dxx * dx;
      a.m03 += dyy * dy;
      a.m21 += dxx * dy;
      a.m12 += dyy * dx;
      if(!a.init)
      {
        a.minx = a.maxx = x;
        a.miny = a.maxy = y;
        a.init = true;
      }
      else
      {
        a.minx = std::min(a.minx, x);
        a.maxx = std::max(a.maxx, x);
        a.miny = std::min(a.miny, y);
        a.maxy = std::max(a.maxy, y);
      }
    }
  }

  const float invW = 1.f / W;
  const float invH = 1.f / H;
  const bool cvjit = (inputs.formula.value == BlobFormula::CvJit);
  const bool doFlip = inputs.flip.value;

  outputs.blobs.value.reserve(R.count);
  for(int l = 1; l <= R.count; ++l)
  {
    Acc& a = acc[static_cast<std::size_t>(l)];
    if(a.m00 <= 0)
      continue;

    const double m00 = a.m00;
    const double m10 = a.m10, m01 = a.m01;
    const double cx = m10 / m00;
    const double cy = m01 / m00;

    // Central moments, in cv.jit.blobs.moments' exact expression order so that the
    // cancellations (and therefore the `nu20 == nu02` equality) reproduce bit-for-bit.
    const double mu20 = a.m20 - m10 * cx;
    const double mu02 = a.m02 - m01 * cy;
    const double mu11 = a.m11 - m10 * cy;
    const double mu30 = a.m30 - cx * (3.0 * mu20 + cx * m10);
    const double mu03 = a.m03 - cy * (3.0 * mu02 + cy * m01);
    const double mu21 = a.m21 - cx * (2.0 * mu11 + cx * m01) - cy * mu20;
    const double mu12 = a.m12 - cy * (2.0 * mu11 + cy * m10) - cx * mu02;

    // Normalised central moments nu_pq (== the standard eta_pq used by Hu):
    //   nu_pq = mu_pq / m00^(1 + (p+q)/2)  ->  / m00^2 for p+q==2, / m00^2.5 for p+q==3.
    const double scale2 = m00 * m00;
    const double scale3 = std::pow(m00, 2.5);
    const double nu20 = mu20 / scale2;
    const double nu02 = mu02 / scale2;
    const double nu11 = mu11 / scale2;
    const double nu21 = mu21 / scale3;
    const double nu12 = mu12 / scale3;
    const double nu30 = mu30 / scale3;
    const double nu03 = mu03 / scale3;

    double theta = 0.0;
    double direction = 0.0;
    double elong = 0.0;

    if(cvjit)
    {
      // ------------------------------------------------------------------ cv.jit mode
      theta = cvjit_orientation(nu20, nu02, nu11);
      direction = cvjit_direction(theta, nu30, nu03);
      if(doFlip)
      {
        // cv.jit.blobs.direction `flip`: shift by exactly pi, sign chosen so the result
        // stays close to the original range.
        direction += (direction > 0.0) ? -PI : PI;
      }

      // cv.jit.blobs.elongation. Deliberately NOT an eigenvalue ratio: it is 0 for a
      // circular/square blob and diverges as the blob thins out. A blob that is exactly
      // one pixel wide has nu20 == 0 (or nu02 == 0) so the quotient is +inf; a single
      // pixel has 0/0 -> NaN. Branch explicitly rather than dividing so no sanitizer can
      // trip on it, and so the degenerate results are documented at the point of use.
      const double num = (nu20 - nu02) * (nu20 - nu02) + 4.0 * nu11 * nu11;
      const double den = nu20 * nu02; // both factors are >= 0
      if(den != 0.0)
        elong = num / den;
      else if(num == 0.0)
        elong = std::numeric_limits<double>::quiet_NaN();
      else
        elong = std::numeric_limits<double>::infinity();
    }
    else
    {
      // -------------------------------------------------------------- Normalized mode
      // Second/third central moments divided by m00 only (the port's historical scale).
      const double c20 = mu20 / m00, c02 = mu02 / m00, c11 = mu11 / m00;
      const double c30 = mu30 / m00, c03 = mu03 / m00;
      const double c21 = mu21 / m00, c12 = mu12 / m00;

      theta = 0.5 * std::atan2(2.0 * c11, c20 - c02);

      // Project the third-order moment onto the principal axis: its sign tells which of
      // the two opposite directions holds the "heavy/long tail".
      const double cs = std::cos(theta);
      const double sn = std::sin(theta);
      const double proj = c30 * cs * cs * cs + 3.0 * c21 * cs * cs * sn
                          + 3.0 * c12 * cs * sn * sn + c03 * sn * sn * sn;
      direction = (proj < 0.0) ? (theta + PI) : theta;
      if(doFlip)
        direction += PI;
      direction = std::fmod(direction, 2.0 * PI);
      if(direction < 0.0)
        direction += 2.0 * PI;

      const double common
          = std::sqrt(std::max(0.0, (c20 - c02) * (c20 - c02) + 4.0 * c11 * c11));
      const double l1 = 0.5 * (c20 + c02 + common);
      const double l2 = 0.5 * (c20 + c02 - common);
      elong = (l2 > 1e-9) ? std::sqrt(l1 / l2) : 1.0;
    }

    // Hu's seven moment invariants, from the same normalised central moments (nu == eta),
    // so they are identical in both formula modes.
    const double ha = nu30 + nu12, hb = nu21 + nu03;
    const double ha2 = ha * ha, hb2 = hb * hb;
    std::array<float, 7> hu{};
    hu[0] = static_cast<float>(nu20 + nu02);
    hu[1] = static_cast<float>((nu20 - nu02) * (nu20 - nu02) + 4.0 * nu11 * nu11);
    hu[2] = static_cast<float>(
        (nu30 - 3.0 * nu12) * (nu30 - 3.0 * nu12)
        + (3.0 * nu21 - nu03) * (3.0 * nu21 - nu03));
    hu[3] = static_cast<float>(ha2 + hb2);
    hu[4] = static_cast<float>(
        (nu30 - 3.0 * nu12) * ha * (ha2 - 3.0 * hb2)
        + (3.0 * nu21 - nu03) * hb * (3.0 * ha2 - hb2));
    hu[5] = static_cast<float>((nu20 - nu02) * (ha2 - hb2) + 4.0 * nu11 * ha * hb);
    hu[6] = static_cast<float>(
        (3.0 * nu21 - nu03) * ha * (ha2 - 3.0 * hb2)
        - (nu30 - 3.0 * nu12) * hb * (3.0 * ha2 - hb2));

    const bool deg = inputs.degrees.value;
    const double rad2deg = 180.0 / PI;

    blob_info bi;
    bi.centroid = {static_cast<float>(cx) * invW, static_cast<float>(cy) * invH};
    bi.bbox = rect{
        a.minx * invW, a.miny * invH, (a.maxx - a.minx + 1) * invW,
        (a.maxy - a.miny + 1) * invH};
    bi.area = static_cast<float>(m00) * invW * invH;
    bi.mass = static_cast<float>(m00);
    bi.orientation = static_cast<float>(deg ? theta * rad2deg : theta);
    bi.direction = static_cast<float>(deg ? direction * rad2deg : direction);
    bi.elongation = static_cast<float>(elong);
    bi.hu = hu;
    bi.nu20 = static_cast<float>(nu20);
    bi.nu02 = static_cast<float>(nu02);
    bi.nu11 = static_cast<float>(nu11);
    bi.nu21 = static_cast<float>(nu21);
    bi.nu12 = static_cast<float>(nu12);
    bi.nu30 = static_cast<float>(nu30);
    bi.nu03 = static_cast<float>(nu03);
    bi.id = l;
    outputs.blobs.value.push_back(bi);
  }

  outputs.count = static_cast<int>(outputs.blobs.value.size());
}
}
