#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <vector>

namespace cv
{
struct tracked_blob
{
  halp::xy_type<float> centroid; // normalised
  float area;                    // fraction of image
  int id;                        // persistent across frames
  int age;                       // frames since first seen

  halp_field_names(centroid, area, id, age);
};

// Temporally-stable blob IDs (cv.jit.blobs.sort). Labels each frame, then greedily matches
// current blob centroids to the previous frame's within a distance threshold, recycling
// freed IDs. Stateful -> Path A.
struct BlobSort
{
  halp_meta(name, "Blob sort");
  halp_meta(c_name, "cv_blob_sort");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Assign persistent IDs to blobs across frames (nearest-neighbour).");
  halp_meta(uuid, "0d5b3a27-6e91-4c84-8f2a-1b7c0e9d4a36");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::hslider_i32<"Min size", halp::range{0, 10000, 4}> min_size;
    halp::hslider_f32<"Max distance", halp::range{0.f, 1.f, 0.1f}> max_distance;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Blobs");
      std::vector<tracked_blob> value;
    } blobs;
  } outputs;

  void operator()() noexcept;

private:
  struct Prev
  {
    float x, y;
    int id;
    int age;
  };
  std::vector<Prev> m_prev;
  std::vector<int> m_free_ids; // recycled ids, kept sorted ascending (lowest first)
  int m_next_id = 1;
};
}
