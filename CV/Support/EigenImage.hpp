#pragma once

/* score-addon-cv — shared CPU image helpers (OpenCV-free).
 *
 * Zero-copy bridges between avendish/halp textures and Eigen / xtensor views, plus the
 * small primitives every Path-A object needs: grayscale conversion and an optional
 * vertical flip.
 *
 * TEXTURE ORIGIN: score delivers texture_input bytes **top-left origin** — row 0 is the top
 * row. (Verified against the working addons: score-addon-onnx wraps the bytes straight into
 * a QImage/model with no flip, and score-addon-bendage processes them top-to-bottom.) So
 * objects index y directly and must NOT flip. `flip_vertical` below is a utility for the rare
 * case of interop with a bottom-left consumer; it is intentionally not called by our objects.
 *
 * Conventions:
 *  - Input textures are RGBA8 (halp::rgba_texture): tightly packed, 4 bytes/pixel,
 *    row-major, `bytes` points at width*height*4 unsigned chars.
 *  - Single-channel outputs are r8 (uint8) or r32f (float).
 *  - These helpers never allocate for the *input*; they wrap the existing buffer. Output
 *    buffers are owned by the caller's `halp::texture_output` (via `create()`).
 */

#include <Eigen/Core>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace cv_support
{

// ---- Row-major Eigen typedefs (images are row-major) --------------------------------------
template <typename T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMatrixMap = Eigen::Map<RowMatrix<T>>;

template <typename T>
using ConstRowMatrixMap = Eigen::Map<const RowMatrix<T>>;

// A pixel is 4 interleaved channels; we view the whole image as (height) x (width*4) bytes
// when we want raw access, or per-channel via strided maps.
struct RgbaView
{
  const std::uint8_t* data{};
  int width{};
  int height{};

  [[nodiscard]] bool valid() const noexcept
  {
    return data && width > 0 && height > 0;
  }

  // pixel (x,y), channel c in [0..3]
  [[nodiscard]] std::uint8_t at(int x, int y, int c) const noexcept
  {
    return data[(static_cast<std::size_t>(y) * width + x) * 4 + c];
  }
};

// Wrap a halp rgba_texture-like input (anything exposing .bytes/.width/.height) zero-copy.
template <typename Tex>
[[nodiscard]] RgbaView as_rgba(const Tex& t) noexcept
{
  return RgbaView{
      .data = reinterpret_cast<const std::uint8_t*>(t.bytes),
      .width = t.width,
      .height = t.height};
}

// ---- Grayscale (luma) -----------------------------------------------------------------------
// Rec.601 luma. Writes a width*height row-major buffer of `Out` (uint8_t or float in [0,1]
// for float / [0,255] for uint8_t). `dst` must hold width*height elements.
template <typename Out>
void to_gray(const RgbaView& src, Out* dst) noexcept
{
  static_assert(
      std::is_same_v<Out, std::uint8_t> || std::is_same_v<Out, float>,
      "to_gray supports uint8_t or float output");

  const int n = src.width * src.height;
  for(int i = 0; i < n; ++i)
  {
    const std::uint8_t* p = src.data + static_cast<std::size_t>(i) * 4;
    const float y = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    if constexpr(std::is_same_v<Out, std::uint8_t>)
      dst[i] = static_cast<std::uint8_t>(y + 0.5f);
    else
      dst[i] = y * (1.0f / 255.0f);
  }
}

// Grayscale directly into an Eigen row-major matrix (height x width), zero-extra-copy view.
template <typename T>
void to_gray(const RgbaView& src, RowMatrix<T>& dst)
{
  dst.resize(src.height, src.width);
  to_gray<T>(src, dst.data());
}

// ---- Vertical flip --------------------------------------------------------------------------
// Utility for interop with a bottom-left consumer. NOTE: score's texture_input is already
// top-left origin, so our objects do NOT need this — see the TEXTURE ORIGIN note at the top.
// Flips rows in place for a tightly-packed image with `bytes_per_pixel` channels.
inline void flip_vertical(
    std::uint8_t* data, int width, int height, int bytes_per_pixel) noexcept
{
  const std::size_t stride = static_cast<std::size_t>(width) * bytes_per_pixel;
  if(stride == 0 || height < 2)
    return;

  // Swap row y with row (height-1-y). Use a small stack buffer per chunk to avoid alloc.
  constexpr std::size_t chunk = 4096;
  std::uint8_t tmp[chunk];
  for(int y = 0; y < height / 2; ++y)
  {
    std::uint8_t* a = data + static_cast<std::size_t>(y) * stride;
    std::uint8_t* b = data + static_cast<std::size_t>(height - 1 - y) * stride;
    std::size_t remaining = stride;
    while(remaining > 0)
    {
      const std::size_t k = remaining < chunk ? remaining : chunk;
      std::memcpy(tmp, a, k);
      std::memcpy(a, b, k);
      std::memcpy(b, tmp, k);
      a += k;
      b += k;
      remaining -= k;
    }
  }
}

} // namespace cv_support
