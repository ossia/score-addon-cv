#pragma once

#include <halp/meta.hpp>
#include <halp/texture.hpp>

namespace cv
{
// Phase-0 anchor object: RGBA8 in -> single-channel (r8) luma out.
// Exercises the full texture-in / texture-out path and CV/Support/EigenImage.hpp,
// with zero external dependencies. Replaces nothing yet; it is the seed the rest builds on.
struct Luminance
{
  halp_meta(name, "Luminance");
  halp_meta(c_name, "cv_luminance");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Convert an image to single-channel luminance (Rec.601).");
  halp_meta(uuid, "8b2f1d4e-6c0a-4f9b-9e2a-1c7d5f3b0a64");

  struct
  {
    halp::texture_input<"In"> image;
  } inputs;

  struct
  {
    halp::texture_output<"Out", halp::r8_texture> image;
  } outputs;

  void operator()() noexcept;
};
}
