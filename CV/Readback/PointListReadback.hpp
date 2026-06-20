#pragma once

#include <halp/buffer.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace cv
{
// Decodes a point-list SSBO of the form { uint count; float coords[2*N]; } (as produced by
// Corners.cs and other "atomic-append point" analyses) into a clean std::vector of (x,y)
// points. The leading count is a uint; coords are interleaved floats.
struct PointListReadback
{
  halp_meta(name, "Point list readback");
  halp_meta(c_name, "cv_pointlist_readback");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Decode a {count, coords[]} SSBO into a list of (x,y) points.");
  halp_meta(uuid, "6a0d3f92-1c47-4b8e-95a2-0f7c3e1d8b64");

  struct
  {
    halp::cpu_buffer_input<"In"> buffer;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Points");
      std::vector<halp::xy_type<float>> value;
    } points;
    halp::val_port<"Count", int> count;
  } outputs;

  void operator()() noexcept
  {
    outputs.points.value.clear();
    outputs.count = 0;

    const auto& raw = inputs.buffer.buffer;
    if(!raw.raw_data || raw.byte_size < static_cast<std::int64_t>(sizeof(std::uint32_t)))
      return;

    // First 4 bytes = count (uint), then interleaved floats.
    std::uint32_t n = 0;
    std::memcpy(&n, raw.raw_data, sizeof(std::uint32_t));

    const float* coords = reinterpret_cast<const float*>(raw.raw_data + sizeof(std::uint32_t));
    const std::int64_t avail_floats
        = (raw.byte_size - static_cast<std::int64_t>(sizeof(std::uint32_t))) / sizeof(float);
    const std::uint32_t max_pairs = static_cast<std::uint32_t>(avail_floats / 2);
    if(n > max_pairs)
      n = max_pairs; // never read past the buffer (overflow was clamped on the GPU too)

    outputs.points.value.reserve(n);
    for(std::uint32_t i = 0; i < n; ++i)
      outputs.points.value.push_back({coords[i * 2], coords[i * 2 + 1]});

    outputs.count = static_cast<int>(n);
  }
};
}
