#pragma once

#include <CompVis/Geometry.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>
#include <opencv2/core/types.hpp>
namespace CompVis
{
struct Contours
{
public:
  halp_meta(name, "Contour detector");
  halp_meta(c_name, "contour_detector");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "OpenCV");
  halp_meta(description, "Detect contours in an image.");
  halp_meta(uuid, "dac1ef5d-db42-463d-82ed-8519ff25d982");

  struct
  {
    halp::texture_input<"In"> image;

    halp::hslider_f32<"Threshold", halp::range{0, 255, 50}> threshold;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;
    halp::val_port<"Contours", int> contours;
    halp::val_port<"Density", int> density;

  } outputs;

  Contours() noexcept;
  ~Contours();

  void operator()();

private:
};
}
