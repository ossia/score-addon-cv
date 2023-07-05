#pragma once

#include <CompVis/CV.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>
namespace CompVis
{
struct Contrast
{
public:
  halp_meta(name, "Contrast measure");
  halp_meta(c_name, "contrast_measure");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "OpenCV");
  halp_meta(description, "Measure the contrast of an image.");
  halp_meta(uuid, "4273179f-b68f-4965-adf2-8144de9b5fee");

  struct
  {
    halp::texture_input<"In"> image;
    struct
    {
      halp__enum("Mode", RMS, RMS, Michelson);
    } mode{};
  } inputs;

  struct
  {
    halp::val_port<"Contrast", float> contrast;

  } outputs;

  Contrast() noexcept;
  ~Contrast();

  void operator()();
};
}
