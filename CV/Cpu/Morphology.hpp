#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Which morphological operation to run. The numbering matches the `operation` input of
// CV/Shaders/Filters/Morphology.fs (0 = Erode, 1 = Dilate, 2 = Open, 3 = Close) so the two
// implementations stay interchangeable from a patch's point of view.
//
// Open  = erode then dilate  (removes specks smaller than the structuring element)
// Close = dilate then erode  (fills holes smaller than the structuring element)
// cv.jit ships these as the `cv.jit.open` / `cv.jit.close` abstractions, which are literally
// a cv.jit.erode -> cv.jit.dilate pair (and the reverse), so they are folded in here.
enum class MorphOperation
{
  Erode,  // 0
  Dilate, // 1
  Open,   // 2
  Close   // 3
};

// The structuring element. This is cv.jit's `mode` attribute (clipped to [0,1], default 0):
//   mode 0 -> square: the full (2r+1)^2 window, i.e. the 8-neighbourhood + centre at r = 1
//   mode 1 -> cross : only the taps with dx == 0 or dy == 0, i.e. the 4-neighbourhood + centre
// (Verified against cv.jit.dilate.cpp: `case 0` tests line1/ip/line3 at -1/0/+1, `case 1`
// tests only *line1, *(ip-1), *ip, *(ip+1), *line3.)
enum class MorphShape
{
  Square, // mode 0
  Cross   // mode 1
};

// Greyscale / binary morphology -- the CPU port of cv.jit.dilate and cv.jit.erode.
// (The two cv.jit externals are exact mirrors of one another: `||` / std::max for dilate,
// `&&` / std::min for erode. One object with an Operation enum covers both.)
//
// WHY THIS EXISTS ALONGSIDE THE SHADER
// CV/Shaders/Filters/Morphology.fs already does min/max morphology on the GPU, but it is
// *greyscale only*. cv.jit's `greyscale 0` (its DEFAULT) is a different operator: the
// neighbourhood is first thresholded to a binary mask, the OR / AND is taken over that mask,
// and the result is written back as a pure 0-or-255 image. The two disagree on any
// non-saturated input: over a neighbourhood of {3, 200}, binary dilate emits 255 while
// greyscale dilate emits 200. This object provides that missing binary path (and, being CPU,
// is unit-testable, which the shader is not).
//
// ATTRIBUTES, AND HOW THEY MAP
//  - `mode`      -> Shape (Square / Cross), default Square = cv.jit's default 0.
//  - `greyscale` -> the Binary toggle, INVERTED: cv.jit's greyscale defaults to 0 (binary),
//                   so Binary defaults to ON. NB cv.jit registers the attribute TWICE, as
//                   "greyscale" and as "grayscale", both at the same struct offset -- they
//                   are aliases for one and the same value, so a single control is faithful.
//  - there is no cv.jit attribute for the radius: its kernel is hard-wired to 3x3 (radius 1),
//    with no iteration count, no anchor and no custom structuring element. `Radius` is exposed
//    here because the shader already exposes it (a strict superset), and it DEFAULTS TO 1 so
//    the out-of-the-box behaviour is cv.jit-exact.
//
// BINARISATION (binary mode only)
// cv.jit works on 1-plane matrices and treats "foreground" as `!= 0`. score textures are
// RGBA8, so foreground here is `Rec.601 luma > Threshold * 255`. Threshold DEFAULTS TO 0,
// which reproduces `!= 0` exactly; raise it to ignore dark noise. The output is re-normalised
// to 0 / 255 on R, G and B (cv.jit writes 255 for char matrices).
//
// GREYSCALE MODE
// Plain max (dilate) / min (erode) of the neighbourhood, applied independently to R, G and B.
// Alpha is never touched by either mode: it is copied straight from the input pixel, since
// eroding alpha would eat holes in the image's opacity.
//
// BORDER
// Off-image neighbours are simply not tested, exactly as cv.jit's dedicated first-line /
// last-line / first-pixel / last-pixel cases do. That is equivalent to clamp-to-edge (a
// clamped tap always lands on a pixel that is already inside the window), so a blob touching
// the frame is not eroded from the outside.
//
// DELIBERATE DEVIATION (bug fix): cv.jit.dilate / cv.jit.erode size the loop with
//     dim[i] = MAX(in_minfo.dim[i], out_minfo.dim[i]);
// while every other cv.jit object uses MIN. With an output matrix larger than the input that
// MAX walks off the end of the input buffer -- a latent out-of-bounds read (and write, since
// the output is indexed with the input's stride). We use the input dimensions and allocate the
// output to match, which is the MIN behaviour.
struct Morphology
{
  halp_meta(name, "Morphology");
  halp_meta(c_name, "cv_morphology");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Erode / dilate / open / close over a square or cross structuring element, in "
      "cv.jit's binary mode or in greyscale. cv.jit.erode / cv.jit.dilate.");
  halp_meta(uuid, "c1a70000-0043-4a00-9000-000000000001");

  struct
  {
    halp::texture_input<"In"> image;
    halp::enum_t<MorphOperation, "Operation"> operation;
    halp::enum_t<MorphShape, "Shape"> shape;
    // ON = cv.jit `greyscale 0`, its default. See BINARISATION above.
    halp::toggle<"Binary", halp::toggle_setup{.init = true}> binary;
    // Foreground test in binary mode: luma > Threshold * 255. 0 == cv.jit's `!= 0`.
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.f}> threshold;
    // cv.jit is fixed at radius 1 (a 3x3 kernel); 1 is therefore the default.
    halp::hslider_i32<"Radius", halp::range{1, 16, 1}> radius;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image; // RGBA8, alpha passed through
  } outputs;

  void operator()() noexcept;

private:
  // Ping-pong planar scratch: 1 plane (0/255 mask) in binary mode, 3 planes (R,G,B) in
  // greyscale mode. Open / Close need two passes, hence two buffers.
  std::vector<std::uint8_t> m_a, m_b;
};
}
