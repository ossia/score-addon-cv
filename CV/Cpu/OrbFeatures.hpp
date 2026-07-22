#pragma once

// cv::keypoint (the list element type shared with FeatureMatch) lives in Brief.hpp so both
// objects can chain without either including the other.
#include <CV/Support/Brief.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace cv
{
// Output coordinate system for the emitted keypoints. Maps onto cv.jit.keypoints'
// `normalize` attribute, which selects the SCALE applied to the pixel positions:
//   Normalized -> sx = 1/W, sy = 1/H     (cv.jit @normalize 1)
//   Pixels     -> sx = sy = 1            (cv.jit @normalize 0 @glcoords 0)
//   GL         -> sx = 2/W, sy = 2/H     (cv.jit @normalize 0 @glcoords 1)
// The first enumerator is halp's default, so Normalized keeps the historical
// behaviour of this object.
enum class Coordinates
{
  Normalized,
  Pixels,
  GL
};

// Oriented FAST + rotated BRIEF (ORB) keypoint detector/descriptor. Port of
// cv.jit.keypoints, OpenCV-free. Detects FAST-9 corners over an `Octaves`-level halving
// image pyramid, computes an intensity-centroid orientation per corner, then a 256-bit
// steered-BRIEF descriptor. Stateless per frame.
//
// Parity notes vs cv.jit.keypoints:
//  - `Octaves` (default 4, as cv.jit) is what buys scale invariance: the same physical
//    feature seen at two sizes is detected at different pyramid levels and — because the
//    descriptor is computed on that level's own image — yields the SAME descriptor.
//    Keypoints are always reported in base-image coordinates, with `octave` recording the
//    level and `size` scaling as 31 * 2^octave.
//  - The keypoint payload now matches cv.jit's 6-plane row: position, size, angle,
//    response, octave (previously only position + angle were emitted).
//  - `angle` is in RADIANS here; cv.jit/OpenCV emit degrees. See CV/Support/Brief.hpp.
//  - The descriptor is a 256-bit steered BRIEF; cv.jit uses OpenCV BRISK's 512-bit
//    descriptor. The two are binary-incompatible either way, so BRISK is deliberately NOT
//    attempted (it would need its own sampling pattern and a scale-space AGAST detector).
//  - cv.jit exposes `method` (AKAZE/BRISK/KAZE/ORB) but only ever implements BRISK; there is
//    nothing to port there.
//  - No cross-octave duplicate suppression: a strong corner visible at several levels can be
//    reported once per level (each with its own octave). This is also what BRISK does before
//    its own NMS, and matching is unaffected (the ratio test tolerates it).
//
// COORDINATE MODES — an exact port of cv.jit.keypoints' `normalize` / `glcoords` pair.
// The two cv.jit attributes are NOT alternatives; they are orthogonal, and the help patch
// depends on the combination:
//
//     scale_x = normalize ? 1/W : (glcoords ? 2/W : 1)     // likewise for y
//     if(glcoords) { X = (px - W/2)*scale_x;  Y = (H/2 - py)*scale_y; }  // note the Y FLIP
//     else         { X = px*scale_x;          Y = py*scale_y; }
//     SIZE = size * max(scale_x, scale_y)
//
// i.e. `normalize` wins for the SCALE while `glcoords` still applies the CENTRING and the
// Y FLIP. `@normalize 1 @glcoords 1` therefore yields [-0.5, 0.5] with Y increasing upwards,
// which a textbook "glcoords means [-1,1]" implementation would get wrong.
//
// Here `Coordinates` picks the scale (and implies centring for GL), and the separate
// `GL centering` toggle is cv.jit's `glcoords`: it adds the centring + Y flip on top of any
// scale. The four cv.jit combinations are all reachable:
//   Normalized + off -> [0,1]                (default, historical behaviour)
//   Pixels     + off -> raw pixels
//   GL         + off -> [-1,1], Y up
//   Normalized + on  -> [-0.5,0.5], Y up     (the combined case above)
//   Pixels     + on  -> identical to GL, exactly as in cv.jit (normalize=0, glcoords=1)
struct OrbFeatures
{
  halp_meta(name, "ORB features");
  halp_meta(c_name, "cv_orb_features");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Oriented FAST + rotated BRIEF keypoint detector and 256-bit descriptors, over a "
      "multi-octave scale space.");
  halp_meta(uuid, "e963f6b1-808f-469d-8ffe-ad1c3c035e4c");

  struct
  {
    halp::texture_input<"In"> image;
    halp::hslider_f32<"Threshold", halp::range{0.f, 0.5f, 0.08f}> threshold;
    halp::hslider_i32<"Max features", halp::range{16, 4096, 512}> max_features;
    // Number of pyramid levels, cv.jit's `octaves` attribute (its default is 4 too).
    // 1 = single scale, i.e. no scale invariance.
    halp::hslider_i32<"Octaves", halp::range{1, 8, 4}> octaves;
    // cv.jit's `normalize` (as a 3-way scale choice); default Normalized = @normalize 1.
    halp::enum_t<Coordinates, "Coordinates"> coordinates;
    // cv.jit's `glcoords`: centring + Y flip, orthogonal to the scale above.
    halp::toggle<"GL centering"> gl_centering;
  } inputs;

  struct
  {
    halp::val_port<"Count", int> count;
    struct
    {
      halp_meta(name, "Keypoints");
      std::vector<keypoint> value;
    } keypoints;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_gray;
};
}
