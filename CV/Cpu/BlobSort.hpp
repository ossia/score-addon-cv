#pragma once

#include <CV/Cpu/BlobStats.hpp>

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

// Units for the latching distance.
//
// Normalized (first enumerator == default): "Max distance" is compared against the
//   distance between centroids expressed in [0,1] image-relative units, i.e.
//   dx = (x1-x0), dy = (y1-y0) with both already normalised. This is the port's original
//   behaviour and stays the default.
//
// Pixels (cv.jit): "Max distance (px)" is compared against the true pixel distance,
//   dx_px = (x1-x0)*width, dy_px = (y1-y0)*height. cv.jit.blobs.sort's `threshold`
//   attribute is a pixel distance and defaults to 10, which is what the pixel slider
//   defaults to. The image dimensions used are those of the most recent valid frame; if
//   no frame has ever been seen (pure list-driven use with nothing upstream providing an
//   image) the object falls back to treating the threshold as normalised.
enum class BlobDistanceUnits
{
  Normalized,
  Pixels
};

// Temporally-stable blob IDs (cv.jit.blobs.sort). Greedily matches current blob centroids
// to the previous frame's within a distance threshold, recycling freed IDs. Stateful ->
// Path A.
//
// Two ways to drive it:
//  - the "Blobs in" list, which is what cv.jit does: cv.jit.blobs.sort performs *no*
//    connectivity analysis at all, it consumes the centroid matrix of
//    cv.jit.blobs.centroids (3 planes, cx/cy at plane offset 0) or the moment matrix of
//    cv.jit.blobs.moments (17 planes, cx/cy at plane offset 14). Here that is a
//    `std::vector<cv::blob_info>` and chains straight from cv::BlobStats' "Blobs" output,
//    or from any producer of that element type (only `centroid` and `area` are read, so a
//    plain centroid list with the rest left at zero works).
//  - the "In" texture, kept from the original port: when the list is empty the object
//    labels the image itself, so patches built before the list inlet existed still work.
// The list wins whenever it is non-empty.
//
// Divergences from cv.jit, all deliberate and all pre-existing:
//  - DOUBLE-LATCH BUG NOT REPRODUCED. cv.jit latches each incoming blob to the nearest
//    previous blob within threshold *without marking it as taken*, so two current blobs
//    that are both closest to the same previous blob both receive that previous blob's id
//    (`x->ndx.data[id]`) and the second overwrites the first's stored position. The result
//    is duplicate ids in one frame. This port marks a previous blob as used once matched,
//    so ids are unique within a frame; the loser is treated as a new blob.
//  - LOWEST FREED ID FIRST. cv.jit keeps an `available` list that it qsorts *after* the
//    cleanup pass, i.e. one frame late, so the id handed out is not necessarily the lowest
//    free one. This port keeps the free list sorted before allocating.
//  - `age` and the `area` field are additions; cv.jit only emits the id.
// The one cv.jit quirk that *is* reproduced is the empty-image special case: see
// operator().
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
    // Appended after the pre-existing ports so port indices of saved projects are
    // preserved.
    // Chains from cv::BlobStats "Blobs" (or any centroid list). Overrides the texture
    // input when non-empty.
    struct
    {
      halp_meta(name, "Blobs in");
      std::vector<blob_info> value;
    } blobs_in;
    // cv.jit.blobs.sort's `threshold` attribute: a latching distance in pixels, default 10.
    halp::hslider_f32<"Max distance (px)", halp::range{0.f, 1000.f, 10.f}> max_distance_px;
    halp::enum_t<BlobDistanceUnits, "Distance units"> units;
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
  // Dimensions of the most recent valid frame, used to convert normalised centroid
  // distances into pixels. 0 when nothing has been seen yet.
  int m_w = 0;
  int m_h = 0;
};
}
