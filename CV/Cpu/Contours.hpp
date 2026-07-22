#pragma once

#include <CV/Geometry.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Per-contour summary (one entry per traced boundary).
//
// All geometry is computed from the *simplified* polygon, i.e. after the Douglas-Peucker
// pass controlled by the "Epsilon" input. With the default epsilon = 0 no simplification
// happens and the values are those of the raw traced boundary.
struct contour_info
{
  halp::xy_type<float> centroid; // normalised [0,1]
  rect bbox;                     // normalised [0,1]
  float area;                    // normalised (fraction of image), always positive
  float perimeter;               // normalised
  int point_count;               // number of points this contour contributes to "Points"
  int is_hole;                   // 0 = outer boundary, 1 = inner (hole) boundary
  int parent;                    // index of the enclosing contour in this list, -1 if none

  halp_field_names(centroid, bbox, area, perimeter, point_count, is_hole, parent);
};

// One boundary point of the flat point list.
//
// This mirrors cv.jit.findcontours' primary output, which is a 1 x total float32 matrix
// with 3 planes {x, y, contourIndex}: every contour's points are concatenated in traversal
// order, each tagged with the ordinal of the contour it belongs to. Ordinals are contiguous
// from 0 and index directly into the "Contours" summary list.
//
// DIFFERENCE FROM cv.jit: cv.jit emits absolute pixel coordinates; this port emits
// coordinates normalised to [0,1] (x / width, y / height) like every other geometry output
// of this addon. Multiply by the image size to recover pixels.
struct contour_point
{
  float x; // normalised [0,1]
  float y; // normalised [0,1]
  int contour;

  halp_field_names(x, y, contour);
};

// Contour finder (cv.jit.findcontours equivalent), OpenCV-free.
//
// Thresholds to binary, then follows borders with Suzuki-Abe style topological analysis
// (Moore-neighbour tracing + border labelling), which yields both outer boundaries and
// inner (hole) boundaries together with the nesting hierarchy, equivalent to OpenCV's
// RETR_TREE mode used by cv.jit.findcontours. Outputs:
//  - the raw boundary points as a flat {x, y, contour} list  (the primary cv.jit output),
//  - a per-contour summary (centroid / bbox / area / perimeter / point count / hole flag /
//    parent),
//  - the traced boundaries drawn into an r8 texture (this port's own addition; the
//    equivalent drawContours() call is commented out in the cv.jit source).
//
// Notes on the cv.jit original:
//  - its "level" attribute is dead: the only use was the commented-out drawContours() call,
//    so it is deliberately not ported;
//  - its "epsilon" attribute (default 0 = no simplification) is ported as "Epsilon".
//
// Boundary orientation: outer boundaries are traced clockwise in image coordinates
// (positive shoelace with y pointing down), hole boundaries counter-clockwise (negative
// shoelace), so the two can be told apart from the point list alone.
struct Contours
{
  halp_meta(name, "Contours");
  halp_meta(c_name, "cv_contours");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Trace blob contours and report boundary points and geometry.");
  halp_meta(uuid, "1d7e8c40-5a2b-49f6-8e31-6c0b9a4d2f57");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::hslider_i32<"Min perimeter", halp::range{0, 4000, 8}> min_perimeter;

    // Douglas-Peucker tolerance, in PIXELS (like cv.jit's epsilon, even though the emitted
    // coordinates are normalised). 0 = no simplification, which is cv.jit's default.
    halp::hslider_f32<"Epsilon", halp::range{0.f, 64.f, 0.f}> epsilon;

    // On (default, matching cv.jit's RETR_TREE): inner boundaries of holes are reported as
    // contours of their own, with is_hole = 1 and parent pointing at the enclosing contour.
    // Off: only outer boundaries are reported - the previous behaviour of this object.
    // Hole borders are still traced internally when off, otherwise they would be
    // mis-reported as additional outer contours.
    halp::toggle<"Find holes", halp::toggle_setup{.init = true}> find_holes;
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
    struct
    {
      halp_meta(name, "Points");
      std::vector<contour_point> value;
    } points;
    // True when at least one boundary hit the internal step guard and was cut short: the
    // point list for that contour is incomplete. Never silently truncated.
    halp::val_port<"Truncated", bool> truncated;
  } outputs;

  void operator()() noexcept;

  // Testing hook: overrides the per-contour step guard (0 = automatic, 8 * pixel count).
  // Not a port; only used to exercise the truncation reporting path.
  void set_max_trace_steps(int n) noexcept { m_max_trace_steps = n; }

private:
  std::vector<std::uint8_t> m_bin;       // binary mask, padded by 1px
  std::vector<std::int32_t> m_nbd;       // 0 = bg, 1 = untouched fg, >=2 = border id
  std::vector<std::uint8_t> m_hole_exit; // Suzuki-Abe's -NBD marking
  int m_max_trace_steps{0};
};
}
