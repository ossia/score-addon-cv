#pragma once

// Shared helpers for the score-addon-cv unit tests: build synthetic RGBA8 images and wire
// them into a halp texture_input, plus small assertion utilities. Tests construct an object,
// set its inputs, call operator(), and assert on outputs — no score/engine harness needed.

#include <halp/texture.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace cvtest
{

// An owned RGBA8 image buffer (row-major, 4 bytes/pixel) kept alive for the duration of a
// test; the texture_input wraps its bytes zero-copy.
struct Image
{
  int width{};
  int height{};
  std::vector<std::uint8_t> px; // width*height*4

  Image() = default;
  Image(int w, int h, std::uint8_t fill = 0)
      : width{w}, height{h}, px(static_cast<std::size_t>(w) * h * 4, fill)
  {
    // default alpha = 255
    for(std::size_t i = 3; i < px.size(); i += 4)
      px[i] = 255;
  }

  void set(int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255)
  {
    auto* p = &px[(static_cast<std::size_t>(y) * width + x) * 4];
    p[0] = r;
    p[1] = g;
    p[2] = b;
    p[3] = a;
  }

  void setGray(int x, int y, std::uint8_t v) { set(x, y, v, v, v, 255); }

  void fillRect(int x0, int y0, int w, int h, std::uint8_t v)
  {
    for(int y = y0; y < y0 + h && y < height; ++y)
      for(int x = x0; x < x0 + w && x < width; ++x)
        if(x >= 0 && y >= 0)
          setGray(x, y, v);
  }

  void fillRectRGB(
      int x0, int y0, int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b)
  {
    for(int y = y0; y < y0 + h && y < height; ++y)
      for(int x = x0; x < x0 + w && x < width; ++x)
        if(x >= 0 && y >= 0)
          set(x, y, r, g, b, 255);
  }
};

// Wire an Image into an object's halp texture_input (rgba_texture), marking it changed.
template <typename TextureInput>
void feed(TextureInput& in, Image& img)
{
  in.texture.bytes = img.px.data();
  in.texture.width = img.width;
  in.texture.height = img.height;
  in.texture.changed = true;
}

// Rec.601 luma of an output r8 / rgba pixel buffer at index.
inline float luma8(std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
  return (0.299f * r + 0.587f * g + 0.114f * b);
}

} // namespace cvtest
