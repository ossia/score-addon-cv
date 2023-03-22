#pragma once

#include <CompVis/Geometry.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#undef READ
#undef WRITE
#undef CONSTANT

#include <opencv2/core/mat.hpp>
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
    halp::texture_output<"Out", halp::r8_texture> image;
    halp::val_port<"Contours", int> contours;
    halp::val_port<"Density", int> density;

  } outputs;

  Contours() noexcept;
  ~Contours();

  void operator()();

private:
  cv::Mat img_source;
  cv::Mat canny_output;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
};
}
