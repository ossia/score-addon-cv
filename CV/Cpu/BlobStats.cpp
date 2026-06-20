#include "BlobStats.hpp"

#include <CV/Cpu/ConnectedComponents.hpp>

#include <algorithm>
#include <cmath>

namespace cv
{
namespace
{
constexpr double PI = 3.14159265358979323846;

// Per-blob raw accumulators for centroid / bbox / second moments.
struct Acc
{
  double n = 0;            // pixel count (m00)
  double sx = 0, sy = 0;   // sum x, sum y
  double sxx = 0, syy = 0, sxy = 0;
  double sxxx = 0, syyy = 0, sxxy = 0, sxyy = 0; // third-order sums
  int minx = 0, miny = 0, maxx = 0, maxy = 0;
  bool init = false;
};
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
    for(int x = 0; x < W; ++x)
    {
      std::int32_t l = R.labels[static_cast<std::size_t>(y) * W + x];
      if(l == 0)
        continue;
      Acc& a = acc[static_cast<std::size_t>(l)];
      a.n += 1;
      a.sx += x;
      a.sy += y;
      a.sxx += static_cast<double>(x) * x;
      a.syy += static_cast<double>(y) * y;
      a.sxy += static_cast<double>(x) * y;
      a.sxxx += static_cast<double>(x) * x * x;
      a.syyy += static_cast<double>(y) * y * y;
      a.sxxy += static_cast<double>(x) * x * y;
      a.sxyy += static_cast<double>(x) * y * y;
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

  outputs.blobs.value.reserve(R.count);
  for(int l = 1; l <= R.count; ++l)
  {
    Acc& a = acc[static_cast<std::size_t>(l)];
    if(a.n <= 0)
      continue;

    const double cx = a.sx / a.n;
    const double cy = a.sy / a.n;

    // Central second moments.
    const double mu20 = a.sxx / a.n - cx * cx;
    const double mu02 = a.syy / a.n - cy * cy;
    const double mu11 = a.sxy / a.n - cx * cy;

    const double theta = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02);

    // Central third moments, used to disambiguate the pointing direction.
    const double mu30 = a.sxxx / a.n - 3.0 * cx * (a.sxx / a.n) + 2.0 * cx * cx * cx;
    const double mu03 = a.syyy / a.n - 3.0 * cy * (a.syy / a.n) + 2.0 * cy * cy * cy;
    const double mu21 = a.sxxy / a.n - 2.0 * cx * (a.sxy / a.n) - cy * (a.sxx / a.n)
                        + 2.0 * cx * cx * cy;
    const double mu12 = a.sxyy / a.n - 2.0 * cy * (a.sxy / a.n) - cx * (a.syy / a.n)
                        + 2.0 * cy * cy * cx;

    // Project the third-order moment onto the principal axis: its sign tells
    // which of the two opposite directions holds the "heavy/long tail".
    const double dirAxis = theta;
    const double c = std::cos(dirAxis);
    const double s = std::sin(dirAxis);
    const double proj = mu30 * c * c * c + 3.0 * mu21 * c * c * s
                        + 3.0 * mu12 * c * s * s + mu03 * s * s * s;
    double direction = (proj < 0.0) ? (dirAxis + PI) : dirAxis;
    direction = std::fmod(direction, 2.0 * PI);
    if(direction < 0.0)
      direction += 2.0 * PI;

    const double common
        = std::sqrt(std::max(0.0, (mu20 - mu02) * (mu20 - mu02) + 4.0 * mu11 * mu11));
    const double l1 = 0.5 * (mu20 + mu02 + common);
    const double l2 = 0.5 * (mu20 + mu02 - common);
    const double elong = (l2 > 1e-9) ? std::sqrt(l1 / l2) : 1.0;

    // Hu's seven moment invariants from normalised central moments.
    // mu20/mu02/... above are central moments divided by m00 (n); recover the raw
    // central moments (M_pq = n * c_pq) to apply eta_pq = M_pq / m00^(1+(p+q)/2).
    const double m00 = a.n;
    auto eta = [&](double cpq, int pq) {
      // cpq is the central moment already divided by m00, so M_pq = m00 * cpq.
      return (m00 * cpq) / std::pow(m00, 1.0 + pq / 2.0);
    };
    const double n20 = eta(mu20, 2), n02 = eta(mu02, 2), n11 = eta(mu11, 2);
    const double n30 = eta(mu30, 3), n03 = eta(mu03, 3);
    const double n21 = eta(mu21, 3), n12 = eta(mu12, 3);

    const double ha = n30 + n12, hb = n21 + n03;
    const double ha2 = ha * ha, hb2 = hb * hb;
    std::array<float, 7> hu{};
    hu[0] = static_cast<float>(n20 + n02);
    hu[1] = static_cast<float>((n20 - n02) * (n20 - n02) + 4.0 * n11 * n11);
    hu[2] = static_cast<float>(
        (n30 - 3.0 * n12) * (n30 - 3.0 * n12)
        + (3.0 * n21 - n03) * (3.0 * n21 - n03));
    hu[3] = static_cast<float>(ha2 + hb2);
    hu[4] = static_cast<float>(
        (n30 - 3.0 * n12) * ha * (ha2 - 3.0 * hb2)
        + (3.0 * n21 - n03) * hb * (3.0 * ha2 - hb2));
    hu[5] = static_cast<float>((n20 - n02) * (ha2 - hb2) + 4.0 * n11 * ha * hb);
    hu[6] = static_cast<float>(
        (3.0 * n21 - n03) * ha * (ha2 - 3.0 * hb2)
        - (n30 - 3.0 * n12) * hb * (3.0 * ha2 - hb2));

    const bool deg = inputs.degrees.value;
    const double rad2deg = 180.0 / PI;

    blob_info bi;
    bi.centroid = {static_cast<float>(cx) * invW, static_cast<float>(cy) * invH};
    bi.bbox = rect{
        a.minx * invW, a.miny * invH, (a.maxx - a.minx + 1) * invW,
        (a.maxy - a.miny + 1) * invH};
    bi.area = static_cast<float>(a.n) * invW * invH;
    bi.mass = static_cast<float>(a.n);
    bi.orientation = static_cast<float>(deg ? theta * rad2deg : theta);
    bi.direction = static_cast<float>(deg ? direction * rad2deg : direction);
    bi.elongation = static_cast<float>(elong);
    bi.hu = hu;
    bi.id = l;
    outputs.blobs.value.push_back(bi);
  }

  outputs.count = static_cast<int>(outputs.blobs.value.size());
}
}
