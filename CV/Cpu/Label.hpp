#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Connected-components labeling (cv.jit.label equivalent), OpenCV-free.
// Thresholds the input to a binary mask, then runs a classic two-pass union-find
// (8-connectivity) to assign a distinct label to each blob. Outputs a visualization
// (each blob a distinct grey/colour) and the blob count. The label field itself can be
// consumed by downstream per-blob analysis.
struct Label
{
  halp_meta(name, "Connected components");
  halp_meta(c_name, "cv_label");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Label connected blobs in a binarized image (union-find).");
  halp_meta(uuid, "f4a9c2d1-83b6-4e07-9c5a-2b1e6d0f7a38");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.5f}> threshold;
    halp::hslider_i32<"Min size", halp::range{0, 10000, 0}> min_size;
    halp::toggle<"Colorize"> colorize;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image; // RGBA8 visualization (grey/colour per blob)
    // Single-channel label field: pixel value = blob id (0 = background, 1..count).
    // r32f carries the exact id with no wrapping, so downstream objects can consume it.
    halp::texture_output<"Labels", halp::r32f_texture> labels;
    halp::val_port<"Count", int> count;
  } outputs;

  void operator()() noexcept;
};
}
