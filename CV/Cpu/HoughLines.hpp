#pragma once

#include <CV/Cpu/Extrema.hpp> // cv::hough_peak

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <cmath>
#include <vector>

namespace cv
{
// OpenCV's cvRound(), for ports of OpenCV code that call it.
//
// cvRound compiles to cvtsd2si / lrint, i.e. it rounds to nearest with TIES TO EVEN under
// the default FE_TONEAREST mode: cvRound(0.5) == 0 and cvRound(1.5) == 2. std::lround is
// round-half-AWAY-from-zero and gives 1 and 2, so the two disagree on every exact .5 --
// which in a Hough accumulator is exactly a bin boundary. Use this wherever the ported
// source said cvRound, and std::lround only where it said round() (cv.jit.hough does say
// round(), so CV/Cpu/HoughTransform.cpp is deliberately NOT using this helper).
//
// Lives here because CV/Cpu/Lines.hpp already includes this header for cv::line_segment.
inline int cv_round(double v) noexcept
{
  // nearbyint, not rint: same result, without raising FE_INEXACT.
  return static_cast<int>(std::nearbyint(v));
}

// A drawable line segment in source-image pixel coordinates.
struct line_segment
{
  float x1{};
  float y1{};
  float x2{};
  float y2{};

  halp_field_names(x1, y1, x2, y2);
};

// Where the accumulator's rho axis is measured FROM. The two accumulator sources in this
// addon disagree, and decoding one with the other's convention is a gross error, not a
// rounding one -- see the note on `Rho origin` below.
enum class HoughRhoOrigin
{
  ImageCorner, // cv::HoughTransform / cv.jit.hough: rho = x*cos + y*sin, x,y from (0,0)
  ImageCentre  // CV/Shaders/Analysis/Hough.cs: rho measured from the image centre
};

// Hough accumulator peaks -> drawable line segments (the cv.jit.hough2lines abstraction).
//
// For a peak at bin (rho_index, theta_index), with `Rho origin` = Image corner:
//
//   theta     = theta_index * dTheta
//   rho       = (rho_index - (accW - 1) / 2) * rho_scale        [INTEGER division]
//
// and the segment spanning the full image width is
//
//   (x1, y1) = (0, rho / sin(theta))
//   (x2, y2) = (w, (rho - w * cos(theta)) / sin(theta))
//
// dTheta and rho_scale ARE INPUTS (`Theta step`, `Rho step`), because they cannot be
// recovered from the accumulator size. cv::HoughTransform builds the accumulator with
//
//   numangle = (int)(pi / theta)        numrho = (int)((2*(w + h) + 1) / rho)
//
// and BOTH casts truncate, so pi/accH and (2*(w+h)+1)/accW are NOT the inverses of those
// formulas. At the cv.jit default theta = 5*0.01745329252 the accumulator is 35 rows high,
// and pi/35 = 0.0897598 rad against a true step of 0.0872665 -- 2.86% too large, which by
// the top theta bin is a 4.9 degree error in the reported line angle. Likewise the rho
// offset: HoughTransform uses the INTEGER (numrho - 1) / 2, so for any even numrho (560 at
// the default rho = 4 on a 640x480 frame) a float (accW - 1) / 2.0 is half a bin -- about
// 2 px -- of systematic bias. Wire cv::HoughTransform's `Theta step` / `Rho step` outlets
// straight in (cv::Extrema forwards them) and both errors vanish exactly.
//
// Leaving either at 0 means "unset" and falls back to the cv.jit-style inversion
// (dTheta = pi/accH, rho_scale = (2*(w+h)+1)/accW), which is what the standalone /
// shader-fed path has to use -- there is no producer to ask.
//
// DELIBERATE DEVIATION -- sin(theta) == 0: the Max abstraction divides straight through
// and emits +/-inf for theta_index 0 (a perfectly vertical line), which is useless to a
// renderer and poisons any downstream arithmetic. When |sin(theta)| is below a small
// epsilon we instead emit the actual vertical segment x = rho / cos(theta), i.e.
// (rho/cos, 0) -> (rho/cos, h). The result is always finite.
//
// DELIBERATE DEVIATION -- `threshold`: in the Max patch, `threshold` and `maxlines` were
// simply forwarded to the embedded cv.jit.extrema. Here cv::Extrema is a separate object
// that already owns its own threshold, so ours acts as an additional post-filter on the
// peak value and DEFAULTS TO 0 (pass-through) to avoid silently filtering twice.
// `Max lines` caps the emitted segments, keeping the first ones in list order.
struct HoughLines
{
  halp_meta(name, "Hough lines");
  halp_meta(c_name, "cv_hough_lines");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Convert Hough accumulator peaks into line segments in image pixel coordinates.");
  halp_meta(uuid, "c1a70000-0006-4a00-9000-000000000006");

  struct
  {
    struct
    {
      halp_meta(name, "Peaks")
      std::vector<hough_peak> value;
    } peaks;

    // Source-image dimensions (the `dim <w> <h>` argument of the Max abstraction).
    // cv::HoughTransform publishes these on `Src width` / `Src height` (and cv::Extrema
    // forwards them), so in the CPU chain they never have to be typed in by hand.
    halp::hslider_i32<"Width", halp::range{1, 8192, 640}> width;
    halp::hslider_i32<"Height", halp::range{1, 8192, 480}> height;

    // Accumulator dimensions. The defaults are the size of the accumulator produced by
    // CV/Shaders/Analysis/Hough.cs -- RHO_BINS(256) columns x THETA_BINS(180) rows -- but
    // SIZE IS NOT ENOUGH to decode that accumulator: also set `Rho origin` to Image centre
    // (see below). In the CPU chain these come from cv::Extrema's `Acc width`/`Acc height`.
    halp::hslider_i32<"Acc width", halp::range{2, 8192, 256}> acc_width;
    halp::hslider_i32<"Acc height", halp::range{1, 8192, 180}> acc_height;

    // Bin steps. 0 == unset == derive from the accumulator size (see the header note).
    // Feed cv::HoughTransform's `Theta step` / `Rho step` here whenever it is the producer.
    halp::hslider_t<double, "Theta step", halp::range{0., 3.15, 0.}> theta_step;
    halp::hslider_t<double, "Rho step", halp::range{0., 65536., 0.}> rho_step;

    // WHERE RHO IS MEASURED FROM. cv::HoughTransform (and cv.jit.hough) accumulate
    // rho = x*cos(theta) + y*sin(theta) with x, y in [0,W) x [0,H) -- distance from the
    // image CORNER, in pixels, bin (accW-1)/2 (integer) meaning rho = 0.
    //
    // CV/Shaders/Analysis/Hough.cs does something entirely different (Hough.cs:60-77): it
    // works in normalised coordinates centred on the image, c = uv - 0.5, and quantises
    //     rbin = int((rho / 0.70710678 * 0.5 + 0.5) * 255 + 0.5)
    // over +/- sqrt(1/2). Inverting that for a square of side S gives
    //     rho_px = (bin - 127.5) * S * sqrt(2) / 255
    // measured from the image CENTRE -- against this object's corner-origin
    // (bin - 127) * (4S + 1) / 256, i.e. a factor of ~2*sqrt(2) in scale PLUS an origin
    // shift (and a different offset convention on top). `Image centre` selects the shader
    // convention: the fractional (accW-1)/2.0
    // offset, an auto rho scale of sqrt(2)*Width/(accW-1), and the centre-to-corner
    // conversion rho += ((w-1)/2)*cos(theta) + ((h-1)/2)*sin(theta).
    //
    // The shader's rho axis is normalised per-axis, so it is a true pixel distance ONLY on
    // a square frame; on a non-square one no single scale can recover pixels and the auto
    // value uses `Width`. That anisotropy is exactly why cv::HoughTransform exists.
    halp::enum_t<HoughRhoOrigin, "Rho origin"> rho_origin;

    // Peak values are decoded from an 8-bit accumulator texture, so they never exceed 255;
    // cv.jit's [0, 4096] was dead travel above that.
    halp::hslider_i32<"Threshold", halp::range{0, 255, 0}> threshold;
    halp::hslider_i32<"Max lines", halp::range{1, 4096, 64}> maxlines;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Lines")
      std::vector<line_segment> value;
    } lines;
  } outputs;

  void operator()() noexcept;
};
}
