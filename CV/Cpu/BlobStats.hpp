#pragma once

#include <CV/Geometry.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <vector>

namespace cv
{
// Which set of formulas to use for orientation / direction / elongation.
//
// CvJit (default, first enumerator so a default-constructed object matches cv.jit):
//   Bit-for-bit the definitions used by cv.jit.blobs.orientation / .direction /
//   .elongation, fed from the *normalised* central moments nu_pq that
//   cv.jit.blobs.moments emits (nu_pq = mu_pq / m00^2 for p+q==2, / m00^2.5 for p+q==3).
//     orientation: c = 0.5*atan(2*nu11 / (nu20-nu02)), with explicit quadrant fix-ups,
//                  giving [0, pi). Returns EXACTLY 0 when nu20 == nu02.
//     direction:   orientation, then a quadrant-banded sign test on nu30 / nu03.
//     elongation:  ((nu20-nu02)^2 + 4*nu11^2) / (nu20*nu02). NOTE: this is NOT an
//                  eigenvalue ratio and is not >= 1. It is 0 for a circular/square blob,
//                  grows without bound as the blob gets thinner, and is +/-inf (or NaN
//                  for a single pixel) when nu20 or nu02 is 0 -- i.e. for a blob that is
//                  exactly one pixel wide. That is cv.jit's behaviour; reproduce it here.
//
// Normalized:
//   The better-behaved definitions this port used before the cv.jit fidelity fix.
//     orientation: 0.5*atan2(2*mu11, mu20-mu02), range [-pi/2, pi/2].
//     direction:   orientation disambiguated by projecting the full 3rd-order central
//                  moment tensor on the principal axis, wrapped into [0, 2pi).
//     elongation:  sqrt(lambda1/lambda2) of the 2nd-order central moment matrix, i.e. the
//                  aspect ratio of the equivalent ellipse. Always >= 1, guarded so a
//                  degenerate (1-pixel-wide or 1-pixel) blob reports 1 instead of inf/NaN.
//   These use mu_pq/m00 (un-normalised by m00^k); the nu* outputs below are unaffected by
//   this choice and are always the cv.jit-normalised values.
enum class BlobFormula
{
  CvJit,
  Normalized
};

struct blob_info
{
  halp::xy_type<float> centroid; // normalised [0,1]
  rect bbox;                     // normalised
  float area;                    // fraction of image
  float mass;                    // raw pixel count of the blob (m00)
  // Angles: radians unless the "Degrees" toggle is on. See BlobFormula above for the
  // exact definition of each of the three quantities in each mode.
  float orientation; // principal axis (undirected)
  float direction;   // pointing direction
  float elongation;
  std::array<float, 7> hu; // Hu's seven moment invariants (mode-independent)
  // Normalised central moments, exactly as cv.jit.blobs.moments emits them on planes 0..6:
  //   nu_pq = mu_pq / m00^2    for p+q == 2   (nu20, nu02, nu11)
  //   nu_pq = mu_pq / m00^2.5  for p+q == 3   (nu21, nu12, nu30, nu03)
  // These are what every downstream cv.jit blobs.* object consumes, so they are exposed
  // directly; they are identical in both BlobFormula modes.
  float nu20;
  float nu02;
  float nu11;
  float nu21;
  float nu12;
  float nu30;
  float nu03;
  int id;

  halp_field_names(
      centroid, bbox, area, mass, orientation, direction, elongation, hu, nu20, nu02,
      nu11, nu21, nu12, nu30, nu03, id);
};

// Connected-components + per-blob measurements (cv.jit.blobs.* family rolled into one).
// Thresholds to binary, labels, and reports geometry per blob. Replaces the whole
// blobs.moments/bounds/centroids/orientation/direction/elongation family.
struct BlobStats
{
  halp_meta(name, "Blob stats");
  halp_meta(c_name, "cv_blob_stats");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob centroid, bounding box, area, moments, orientation, elongation.");
  halp_meta(uuid, "3f8a0b51-9d2c-4e76-8a4f-1b6e2c9d0a73");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::hslider_i32<"Min size", halp::range{0, 10000, 4}> min_size;
    halp::toggle<"Degrees"> degrees;
    // halp::enum_t (magic_enum-backed) value-initialises `value` to the first enumerator,
    // so a default-constructed object is always valid (no UBSAN invalid-enum load) and
    // defaults to the cv.jit-compatible formulas.
    halp::enum_t<BlobFormula, "Formula"> formula;
    // cv.jit.blobs.direction's `flip` attribute: add +/-pi to the reported direction after
    // disambiguation (cv.jit subtracts pi when direction > 0, adds pi otherwise; either
    // way the direction is shifted by exactly pi). Absent from this port until now.
    halp::toggle<"Flip"> flip;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Blobs");
      std::vector<blob_info> value;
    } blobs;
  } outputs;

  void operator()() noexcept;
};
}
