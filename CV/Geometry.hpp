#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

namespace cv
{
// Axis-aligned rectangle, fields exposed by name for list/struct outlets.
struct rect
{
  float x{}, y{}, w{}, h{};
  halp_field_names(x, y, w, h);
};
}
