#pragma once

#include <CV/Geometry.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Per-contour summary (one entry per detected blob boundary).
struct contour_info
{
  halp::xy_type<float> centroid; // normalised [0,1]
  rect bbox;                     // normalised [0,1]
  float area;                    // normalised (fraction of image)
  float perimeter;               // normalised
  int point_count;

  halp_field_names(centroid, bbox, area, perimeter, point_count);
};

// Contour finder (cv.jit.findcontours equivalent), OpenCV-free.
// Thresholds to binary, traces outer boundaries with Moore-neighbor border following,
// and reports per-contour geometry (centroid, bounding box, area, perimeter). Also draws
// the traced boundaries into the output texture.
struct Contours
{
  halp_meta(name, "Contours");
  halp_meta(c_name, "cv_contours");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Trace blob contours and report per-contour geometry.");
  halp_meta(uuid, "1d7e8c40-5a2b-49f6-8e31-6c0b9a4d2f57");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::hslider_i32<"Min perimeter", halp::range{0, 4000, 8}> min_perimeter;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Contours");
      std::vector<contour_info> value;
    } contours;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<std::uint8_t> m_bin;     // binary mask
  std::vector<std::uint8_t> m_visited; // boundary-start bookkeeping
};
}
