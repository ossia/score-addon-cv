#pragma once

// -----------------------------------------------------------------------------------
// The cv.jit "blobs" chain, as separate, user-composable objects.
//
// cv.jit's model is
//
//     cv.jit.label -> cv.jit.blobs.moments -> { orientation, elongation, direction,
//                                               bounds, centroids, recon, sort }
//
// where cv.jit.blobs.moments emits ONE ROW PER LABEL of a 17-plane float32 matrix and
// every downstream object is an independent Max object that reads planes 0..6 of that row
// (nu20, nu02, nu11, nu21, nu12, nu30, nu03) and emits one number per row. Because the
// matrix is *label-indexed*, a label that is absent from the image still occupies a row --
// a "gap row" filled with a sentinel.
//
// This port originally collapsed all of that into the single monolithic `cv::BlobStats`
// texture object, so none of the chain could be re-composed by a user. `BlobStats` now
// exposes the normalised central moments on its `Blobs` list output, and the objects
// below consume that list, one object per cv.jit object:
//
//     cv::BlobStats  ->  std::vector<cv::blob_info>  ->  BlobsOrientation
//                                                    ->  BlobsElongation
//                                                    ->  BlobsDirection
//                                                    ->  BlobsBounds
//                                                    ->  BlobsCentroids
//
// The formulas are transcribed from `CV/Cpu/BlobStats.cpp` (which itself is a 1:1
// transcription of the cv.jit sources) rather than re-derived, so the chained result is
// identical to what BlobStats reports directly -- there is exactly one definition of
// "orientation" in this addon. See `CV/Cpu/BlobStats.hpp` for the (deliberately quirky)
// semantics of each quantity in each BlobFormula mode.
// -----------------------------------------------------------------------------------

#include <CV/Cpu/BlobStats.hpp> // cv::blob_info, cv::BlobFormula

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cv
{
namespace blobs_chain
{
inline constexpr double PI = 3.14159265358979323846;

// Guard against an absurd allocation if a caller hands us a garbage blob id while
// label-indexed (gap-row) output is on. cv.jit's own tables are 256 entries; cv.jit.label
// caps at 2048. 65536 is far past both and still trivially small.
inline constexpr int MAX_LABEL_INDEX = 65536;

// cv.jit.blobs.orientation, verbatim: atan (NOT atan2) of 2*nu11/(nu20-nu02), halved, then
// explicit quadrant fix-ups; range [0, pi). Returns EXACTLY 0 when nu20 == nu02.
inline double cvjit_orientation(double nu20, double nu02, double nu11) noexcept
{
  const double d = nu20 - nu02;
  if(d == 0.0)
    return 0.0;

  const double c = std::atan((nu11 * 2.0) / d) * 0.5;
  if(nu20 > nu02)
    return (c < 0.0) ? (c + PI) : c;
  return c + PI * 0.5;
}

// cv.jit.blobs.direction's quadrant-banded sign test on the 3rd-order normalised moments.
inline double cvjit_direction(double theta, double nu30, double nu03) noexcept
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

// cv.jit.blobs.elongation: ((nu20-nu02)^2 + 4*nu11^2) / (nu20*nu02).
// NOT an eigenvalue ratio and NOT >= 1: it is 0 for a circular/square blob, grows without
// bound as the blob thins, is +inf when the blob is exactly one pixel wide (nu20 or nu02
// is 0) and NaN for a single pixel (0/0). Branch explicitly so no sanitizer trips on it.
inline double cvjit_elongation(double nu20, double nu02, double nu11) noexcept
{
  const double num = (nu20 - nu02) * (nu20 - nu02) + 4.0 * nu11 * nu11;
  const double den = nu20 * nu02; // both factors are >= 0
  if(den != 0.0)
    return num / den;
  if(num == 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return std::numeric_limits<double>::infinity();
}

// ---- "Normalized" (non-cv.jit) variants, matching BlobStats' BlobFormula::Normalized ----
//
// BlobStats computes those from mu_pq/m00; here we only have nu_pq = mu_pq/m00^k. Every
// one of the three is invariant under a common POSITIVE rescaling of its inputs:
//   * orientation uses atan2(2*c11, c20-c02) -- a uniform scale of both arguments leaves
//     atan2 unchanged;
//   * elongation is a ratio of eigenvalues of the same 2x2 matrix -- scale cancels;
//   * direction only uses the SIGN of the third-order projection.
// so feeding nu_pq instead of mu_pq/m00 yields bit-identical results.
inline double norm_orientation(double n20, double n02, double n11) noexcept
{
  return 0.5 * std::atan2(2.0 * n11, n20 - n02);
}

inline double norm_elongation(double n20, double n02, double n11) noexcept
{
  const double common = std::sqrt(std::max(0.0, (n20 - n02) * (n20 - n02) + 4.0 * n11 * n11));
  const double l1 = 0.5 * (n20 + n02 + common);
  const double l2 = 0.5 * (n20 + n02 - common);
  return (l2 > 1e-9) ? std::sqrt(l1 / l2) : 1.0;
}

inline double norm_direction(
    double theta, double n30, double n03, double n21, double n12) noexcept
{
  const double cs = std::cos(theta);
  const double sn = std::sin(theta);
  const double proj = n30 * cs * cs * cs + 3.0 * n21 * cs * cs * sn
                      + 3.0 * n12 * cs * sn * sn + n03 * sn * sn * sn;
  return (proj < 0.0) ? (theta + PI) : theta;
}
}

// A blob bounding box in PIXEL coordinates, inclusive on all four sides, in the plane
// order cv.jit.blobs.bounds emits: left, top, right, bottom.
struct blob_bounds
{
  int left{};
  int top{};
  int right{};
  int bottom{};

  halp_field_names(left, top, right, bottom);
};

// A blob centroid in PIXEL coordinates plus its mass, in the plane order
// cv.jit.blobs.centroids emits: x, y, mass.
struct blob_centroid
{
  float x{};
  float y{};
  float mass{};

  halp_field_names(x, y, mass);
};

// -------------------------------------------------------------------- BlobsOrientation
// cv.jit.blobs.orientation. Consumes a blob list (BlobStats' `Blobs` outlet) and emits the
// principal-axis angle of every blob, in list order.
struct BlobsOrientation
{
  halp_meta(name, "Blobs orientation");
  halp_meta(c_name, "cv_blobs_orientation");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob principal axis angle from normalised central moments.");
  halp_meta(uuid, "c1a70000-0030-4a00-9000-000000000001");

  struct
  {
    struct
    {
      halp_meta(name, "Blobs")
      std::vector<blob_info> value;
    } blobs;
    halp::toggle<"Degrees"> degrees;
    halp::enum_t<BlobFormula, "Formula"> formula;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Orientation")
      std::vector<float> value;
    } orientation;
  } outputs;

  void operator()() noexcept
  {
    auto& out = outputs.orientation.value;
    out.clear();
    out.reserve(inputs.blobs.value.size());

    const bool cvjit = (inputs.formula.value == BlobFormula::CvJit);
    const double k = inputs.degrees.value ? (180.0 / blobs_chain::PI) : 1.0;

    for(const blob_info& b : inputs.blobs.value)
    {
      const double t = cvjit ? blobs_chain::cvjit_orientation(b.nu20, b.nu02, b.nu11)
                             : blobs_chain::norm_orientation(b.nu20, b.nu02, b.nu11);
      out.push_back(static_cast<float>(t * k));
    }
  }
};

// --------------------------------------------------------------------- BlobsElongation
// cv.jit.blobs.elongation. See blobs_chain::cvjit_elongation: this is NOT an aspect ratio.
struct BlobsElongation
{
  halp_meta(name, "Blobs elongation");
  halp_meta(c_name, "cv_blobs_elongation");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob elongation from normalised central moments.");
  halp_meta(uuid, "c1a70000-0030-4a00-9000-000000000002");

  struct
  {
    struct
    {
      halp_meta(name, "Blobs")
      std::vector<blob_info> value;
    } blobs;
    halp::enum_t<BlobFormula, "Formula"> formula;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Elongation")
      std::vector<float> value;
    } elongation;
  } outputs;

  void operator()() noexcept
  {
    auto& out = outputs.elongation.value;
    out.clear();
    out.reserve(inputs.blobs.value.size());

    const bool cvjit = (inputs.formula.value == BlobFormula::CvJit);
    for(const blob_info& b : inputs.blobs.value)
    {
      const double e = cvjit ? blobs_chain::cvjit_elongation(b.nu20, b.nu02, b.nu11)
                             : blobs_chain::norm_elongation(b.nu20, b.nu02, b.nu11);
      out.push_back(static_cast<float>(e));
    }
  }
};

// ---------------------------------------------------------------------- BlobsDirection
// cv.jit.blobs.direction: the orientation, disambiguated into a pointing direction using
// the third-order moments, with cv.jit's `flip` attribute.
struct BlobsDirection
{
  halp_meta(name, "Blobs direction");
  halp_meta(c_name, "cv_blobs_direction");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob pointing direction from normalised central moments.");
  halp_meta(uuid, "c1a70000-0030-4a00-9000-000000000003");

  struct
  {
    struct
    {
      halp_meta(name, "Blobs")
      std::vector<blob_info> value;
    } blobs;
    halp::toggle<"Degrees"> degrees;
    halp::enum_t<BlobFormula, "Formula"> formula;
    // cv.jit.blobs.direction's `flip`: shift the result by exactly pi.
    halp::toggle<"Flip"> flip;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Direction")
      std::vector<float> value;
    } direction;
  } outputs;

  void operator()() noexcept
  {
    auto& out = outputs.direction.value;
    out.clear();
    out.reserve(inputs.blobs.value.size());

    const bool cvjit = (inputs.formula.value == BlobFormula::CvJit);
    const bool doFlip = inputs.flip.value;
    const double k = inputs.degrees.value ? (180.0 / blobs_chain::PI) : 1.0;

    for(const blob_info& b : inputs.blobs.value)
    {
      double d = 0.0;
      if(cvjit)
      {
        const double t = blobs_chain::cvjit_orientation(b.nu20, b.nu02, b.nu11);
        d = blobs_chain::cvjit_direction(t, b.nu30, b.nu03);
        if(doFlip)
          d += (d > 0.0) ? -blobs_chain::PI : blobs_chain::PI;
      }
      else
      {
        const double t = blobs_chain::norm_orientation(b.nu20, b.nu02, b.nu11);
        d = blobs_chain::norm_direction(t, b.nu30, b.nu03, b.nu21, b.nu12);
        if(doFlip)
          d += blobs_chain::PI;
        d = std::fmod(d, 2.0 * blobs_chain::PI);
        if(d < 0.0)
          d += 2.0 * blobs_chain::PI;
      }
      out.push_back(static_cast<float>(d * k));
    }
  }
};

// ------------------------------------------------------------------------ BlobsBounds
// cv.jit.blobs.bounds: per-blob {left, top, right, bottom} in pixels, inclusive.
//
// `blob_info::bbox` is normalised to [0,1] by the source image dimensions, so `Width` /
// `Height` MUST be set to the dimensions of the image BlobStats analysed for the pixel
// coordinates to come out right.
//
// `Label indexed` reproduces cv.jit's addressing: the output has one row per label id from
// 1 to the largest id present, and an absent label gets cv.jit's sentinel gap row
// {0x7FFFFFFF, 0x7FFFFFFF, 0, 0} (its "empty" initialiser: left/top = INT_MAX, right/
// bottom = 0). Off (the default) the output is a compact list in input order.
struct BlobsBounds
{
  halp_meta(name, "Blobs bounds");
  halp_meta(c_name, "cv_blobs_bounds");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob bounding boxes in pixel coordinates.");
  halp_meta(uuid, "c1a70000-0030-4a00-9000-000000000004");

  struct
  {
    struct
    {
      halp_meta(name, "Blobs")
      std::vector<blob_info> value;
    } blobs;
    // Dimensions of the image the blobs were measured on (blob_info is normalised).
    halp::hslider_i32<"Width", halp::range{1, 8192, 640}> width;
    halp::hslider_i32<"Height", halp::range{1, 8192, 480}> height;
    halp::toggle<"Label indexed"> label_indexed;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Bounds")
      std::vector<blob_bounds> value;
    } bounds;
  } outputs;

  // cv.jit.blobs.bounds' "no such blob" row.
  static constexpr blob_bounds sentinel() noexcept
  {
    return blob_bounds{
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 0, 0};
  }

  void operator()() noexcept
  {
    auto& out = outputs.bounds.value;
    out.clear();

    const int W = inputs.width.value;
    const int H = inputs.height.value;

    auto to_px = [&](const blob_info& b) {
      blob_bounds r;
      r.left = static_cast<int>(std::lround(static_cast<double>(b.bbox.x) * W));
      r.top = static_cast<int>(std::lround(static_cast<double>(b.bbox.y) * H));
      const int w = static_cast<int>(std::lround(static_cast<double>(b.bbox.w) * W));
      const int h = static_cast<int>(std::lround(static_cast<double>(b.bbox.h) * H));
      r.right = r.left + (w > 0 ? w - 1 : 0);
      r.bottom = r.top + (h > 0 ? h - 1 : 0);
      return r;
    };

    if(!inputs.label_indexed.value)
    {
      out.reserve(inputs.blobs.value.size());
      for(const blob_info& b : inputs.blobs.value)
        out.push_back(to_px(b));
      return;
    }

    int maxid = 0;
    for(const blob_info& b : inputs.blobs.value)
      if(b.id > maxid && b.id <= blobs_chain::MAX_LABEL_INDEX)
        maxid = b.id;
    if(maxid <= 0)
      return;

    out.assign(static_cast<std::size_t>(maxid), sentinel());
    for(const blob_info& b : inputs.blobs.value)
      if(b.id >= 1 && b.id <= maxid)
        out[static_cast<std::size_t>(b.id - 1)] = to_px(b);
  }
};

// --------------------------------------------------------------------- BlobsCentroids
// cv.jit.blobs.centroids: per-blob {x, y, mass} in pixels.
//
// `Label indexed` uses cv.jit's sentinel gap row {-1, -1, 0}: cv.jit writes
// (-1, -1, mass) when mass == 0, i.e. exactly (-1, -1, 0) for a label with no pixels.
struct BlobsCentroids
{
  halp_meta(name, "Blobs centroids");
  halp_meta(c_name, "cv_blobs_centroids");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob centroid and mass in pixel coordinates.");
  halp_meta(uuid, "c1a70000-0030-4a00-9000-000000000005");

  struct
  {
    struct
    {
      halp_meta(name, "Blobs")
      std::vector<blob_info> value;
    } blobs;
    halp::hslider_i32<"Width", halp::range{1, 8192, 640}> width;
    halp::hslider_i32<"Height", halp::range{1, 8192, 480}> height;
    halp::toggle<"Label indexed"> label_indexed;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Centroids")
      std::vector<blob_centroid> value;
    } centroids;
  } outputs;

  // cv.jit.blobs.centroids' "no such blob" row.
  static constexpr blob_centroid sentinel() noexcept
  {
    return blob_centroid{-1.f, -1.f, 0.f};
  }

  void operator()() noexcept
  {
    auto& out = outputs.centroids.value;
    out.clear();

    const float W = static_cast<float>(inputs.width.value);
    const float H = static_cast<float>(inputs.height.value);

    auto to_px = [&](const blob_info& b) {
      return blob_centroid{b.centroid.x * W, b.centroid.y * H, b.mass};
    };

    if(!inputs.label_indexed.value)
    {
      out.reserve(inputs.blobs.value.size());
      for(const blob_info& b : inputs.blobs.value)
        out.push_back(to_px(b));
      return;
    }

    int maxid = 0;
    for(const blob_info& b : inputs.blobs.value)
      if(b.id > maxid && b.id <= blobs_chain::MAX_LABEL_INDEX)
        maxid = b.id;
    if(maxid <= 0)
      return;

    out.assign(static_cast<std::size_t>(maxid), sentinel());
    for(const blob_info& b : inputs.blobs.value)
      if(b.id >= 1 && b.id <= maxid)
        out[static_cast<std::size_t>(b.id - 1)] = to_px(b);
  }
};
}
