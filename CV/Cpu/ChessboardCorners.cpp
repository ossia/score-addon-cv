#include "ChessboardCorners.hpp"

#include <CV/Support/Chessboard.hpp>
#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace cv
{
namespace
{
/* --- minimal RGBA8 rasteriser -------------------------------------------------
 * cv.jit's first outlet is the input image with cvDrawChessboardCorners painted on
 * it. We reproduce that annotation with integer Bresenham primitives: no
 * anti-aliasing, no dependencies, a handful of writes per corner.
 */
struct Canvas
{
  std::uint8_t* px;
  int W, H;

  void plot(int x, int y, const std::array<std::uint8_t, 3>& c) noexcept
  {
    if(x < 0 || y < 0 || x >= W || y >= H)
      return;
    auto* p = px + ((std::size_t)y * W + x) * 4;
    p[0] = c[0];
    p[1] = c[1];
    p[2] = c[2];
    p[3] = 255;
  }

  void line(int x0, int y0, int x1, int y1, const std::array<std::uint8_t, 3>& c) noexcept
  {
    const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    // Bounded by the Manhattan span, so it always terminates.
    for(;;)
    {
      plot(x0, y0, c);
      if(x0 == x1 && y0 == y1)
        break;
      const int e2 = 2 * err;
      if(e2 >= dy)
      {
        err += dy;
        x0 += sx;
      }
      if(e2 <= dx)
      {
        err += dx;
        y0 += sy;
      }
    }
  }

  // Midpoint circle outline.
  void circle(int cx, int cy, int r, const std::array<std::uint8_t, 3>& c) noexcept
  {
    if(r <= 0)
    {
      plot(cx, cy, c);
      return;
    }
    int x = r, y = 0, err = 1 - r;
    while(x >= y)
    {
      plot(cx + x, cy + y, c);
      plot(cx + y, cy + x, c);
      plot(cx - y, cy + x, c);
      plot(cx - x, cy + y, c);
      plot(cx - x, cy - y, c);
      plot(cx - y, cy - x, c);
      plot(cx + y, cy - x, c);
      plot(cx + x, cy - y, c);
      ++y;
      if(err < 0)
        err += 2 * y + 1;
      else
      {
        --x;
        err += 2 * (y - x) + 1;
      }
    }
  }

  void cross(int cx, int cy, int r, const std::array<std::uint8_t, 3>& c) noexcept
  {
    line(cx - r, cy - r, cx + r, cy + r, c);
    line(cx - r, cy + r, cx + r, cy - r, c);
  }
};

// cvDrawChessboardCorners' per-row palette, converted from its BGR Scalars to RGB.
constexpr std::array<std::array<std::uint8_t, 3>, 7> kRowColors{
    {{255, 0, 0},
     {255, 128, 0},
     {200, 200, 0},
     {0, 255, 0},
     {0, 200, 200},
     {0, 0, 255},
     {255, 0, 255}}};

constexpr std::array<std::uint8_t, 3> kNotFoundColor{255, 0, 0}; // red, as in cv.jit
} // namespace

void ChessboardCorners::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || !in.width || !in.height)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  cv_support::ChessboardParams p;
  p.cols = inputs.cols.value;
  p.rows = inputs.rows.value;
  p.threshold = inputs.threshold.value;
  p.subpixel = bool(inputs.subpixel.value);
  p.win_w = inputs.window_size.value.x;
  p.win_h = inputs.window_size.value.y;
  p.zero_w = inputs.zero_zone.value.x;
  p.zero_h = inputs.zero_zone.value.y;

  const auto R = cv_support::find_chessboard_corners(src, p);

  // Corners are emitted whether or not the full grid was recognised — cv.jit sends
  // whatever cvFindChessboardCorners produced (sub-pixel refined) on its corner
  // outlet and reports the match separately, on `Found`.
  outputs.corners.value.clear();
  outputs.corners.value.reserve(R.corners.size());
  for(const auto& c : R.corners)
  {
    chessboard_corner cc;
    cc.position = {c[0], c[1]};
    outputs.corners.value.push_back(cc);
  }
  outputs.count = static_cast<int>(R.corners.size());
  outputs.found = R.found;

  // --- annotated output (cv.jit's first outlet) -------------------------------
  outputs.image.create(W, H);
  auto& out = outputs.image.texture;
  std::memcpy(out.bytes, in.bytes, (std::size_t)W * H * 4);

  Canvas canvas{out.bytes, W, H};
  const int r = 4; // cvDrawChessboardCorners' marker radius

  auto marker_at = [&](std::size_t i) {
    // corners_px is sub-pixel; round to the nearest raster position for drawing.
    return std::array<int, 2>{
        (int)std::lround(R.corners_px[i][0]), (int)std::lround(R.corners_px[i][1])};
  };

  if(R.found && (int)R.corners_px.size() == p.cols * p.rows)
  {
    // Colour zig-zag polyline through the ordered corners + a cross/circle marker,
    // the row colour cycling through the palette exactly as cv.jit does.
    std::array<int, 2> prev{0, 0};
    std::size_t i = 0;
    for(int row = 0; row < p.rows; ++row)
    {
      const auto& col = kRowColors[(std::size_t)row % kRowColors.size()];
      for(int c = 0; c < p.cols; ++c, ++i)
      {
        const auto q = marker_at(i);
        if(i != 0)
          canvas.line(prev[0], prev[1], q[0], q[1], col);
        canvas.cross(q[0], q[1], r, col);
        canvas.circle(q[0], q[1], r + 1, col);
        prev = q;
      }
    }
  }
  else
  {
    // Not found: distinct markers and NO connecting polyline — plain red circles.
    for(std::size_t i = 0; i < R.corners_px.size(); ++i)
    {
      const auto q = marker_at(i);
      canvas.circle(q[0], q[1], r, kNotFoundColor);
    }
  }
  out.changed = true;
}
}
