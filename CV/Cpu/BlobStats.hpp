#pragma once

#include <CV/Geometry.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <vector>

namespace cv
{
struct blob_info
{
  halp::xy_type<float> centroid; // normalised [0,1]
  rect bbox;                     // normalised
  float area;                    // fraction of image
  float mass;                    // raw pixel count of the blob (m00)
  // Orientation of the principal axis from atan2(2*mu11, mu20-mu02)/2.
  // Range is [-pi/2, pi/2] (radians), or degrees when the "Degrees" toggle is on.
  // NOTE: these are the standard image-moment definitions and intentionally differ
  // from cv.jit: cv.jit uses an atan-based orientation and a ratio (lambda1/lambda2)
  // elongation, whereas we use sqrt(lambda1/lambda2) for elongation.
  float orientation;             // principal axis (undirected), radians or degrees
  float direction;               // pointing direction [0,2pi) radians, or degrees
  float elongation;              // sqrt(lambda1/lambda2) >= 1
  std::array<float, 7> hu;       // Hu's seven moment invariants
  int id;

  halp_field_names(
      centroid, bbox, area, mass, orientation, direction, elongation, hu, id);
};

// Connected-components + per-blob measurements (cv.jit.blobs.* family rolled into one).
// Thresholds to binary, labels, and reports geometry per blob. Replaces the whole
// blobs.bounds/centroids/orientation/elongation family.
struct BlobStats
{
  halp_meta(name, "Blob stats");
  halp_meta(c_name, "cv_blob_stats");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Per-blob centroid, bounding box, area, orientation, elongation.");
  halp_meta(uuid, "3f8a0b51-9d2c-4e76-8a4f-1b6e2c9d0a73");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::hslider_i32<"Min size", halp::range{0, 10000, 4}> min_size;
    halp::toggle<"Degrees"> degrees;
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
