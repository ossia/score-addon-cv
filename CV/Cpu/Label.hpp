#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
// Neighbourhood used to decide whether two foreground pixels belong to the same blob.
//
// IMPORTANT -- this port defaults to Eight, cv.jit is Four.
// cv.jit.label uses a 4-connected scanline seed-fill (see fillBlobLong(): it only ever
// walks left/right along a row and steps one row up/down, never diagonally), so two
// pixels that touch only at a corner are TWO blobs in cv.jit and ONE blob here at the
// default. The port shipped 8-connected first and existing patches/tests depend on it, so
// Eight stays the first enumerator (== the default); select Four for cv.jit parity.
enum class Connectivity
{
  Eight,
  Four
};

// cv.jit.label's `mode` attribute. Its meaning DEPENDS ON THE OUTPUT TYPE:
//   Sequential (mode 0): pixels carry the blob's index, renumbered 1..count over the
//                        blobs that survive the size filter.
//   Mass       (mode 1): with long output  -> pixels carry the blob's MASS (pixel count).
//                        with char output  -> pixels carry the blob's SIZE RANK, where
//                                             the LARGEST blob is rank 1 and the smallest
//                                             is rank ndx (cv.jit qsorts ascending by size
//                                             and emits ndx-i+1).
enum class LabelMode
{
  Sequential,
  Mass
};

// How `Min size` is compared against a blob's pixel count.
//   AtLeast     (default): keep when size >= min_size. This is what this port has always
//                          done and what the existing tests assert.
//   GreaterThan          : keep when size >  min_size. This is cv.jit's rule
//                          (`if (blobs[i].size > thresh)`), so a blob of exactly
//                          `min_size` pixels is DROPPED. Select this for cv.jit parity.
// Both rules are identical at min_size == 0 (a blob always has >= 1 pixel), which is why
// the default is harmless.
enum class SizeFilter
{
  AtLeast,
  GreaterThan
};

// Connected-components labeling (cv.jit.label equivalent), OpenCV-free.
// Thresholds the input to a binary mask, then runs a classic two-pass union-find
// (4- or 8-connectivity) to assign a distinct label to each blob. Outputs a visualization
// (each blob a distinct grey/colour) and the blob count. The label field itself can be
// consumed by downstream per-blob analysis.
//
// Output split:
//   `Out`    is always an IDENTITY visualization (one grey/colour per surviving blob); it
//            is unaffected by Mode / Char output, so a patch that only wants to *look* at
//            the segmentation keeps working whatever the numeric mode is.
//   `Labels` is the r32f data field. It carries exactly cv.jit's output matrix contents for
//            the selected (mode, charmode) pair — index, mass or size rank — but NORMALISED
//            (see below).
//
// LABEL FIELD SCALE — the addon-wide r32f contract
// -----------------------------------------------
// An r32f texture output carries [0,1]: score converts it to the RGBA8 that every texture
// input expects by interpreting the float as if it already were in that range (the contract,
// with the score source reference, is documented once at the top of CV/Cpu/CartoPol.hpp).
// Writing raw label ids there meant every id >= 1 arrived at the next object as 255 — the
// whole field collapsed into a single flat white blob, and a two-blob image was
// indistinguishable from a two-hundred-blob one.
//
// `Labels` therefore carries `value / Max label`, and `Max label` (a value port) is the
// largest value present in the field this frame:
//
//     value = Labels * Max label          (exactly 0 for background)
//
// `Max label` is 0 when the field is entirely background, in which case `Labels` is all 0.
// Normalising by the observed maximum rather than by a fixed constant is what keeps the
// visualisation useful (the field always spans the full range) and the numbers recoverable
// (no clipping is ever possible, whatever the mode: index, mass or rank).
//
// Deliberate deviations from cv.jit (bugs NOT reproduced):
//   * cv.jit declares `Blob Blobs[2048]` but clamps with `ndx = min(ndx, 2048)` and then
//     writes `blobs[ndx]`, i.e. a one-past-the-end write at the cap. Here the cap is a
//     real cap: blob number `Max blobs` + 1 and beyond are never started, their pixels
//     stay background, and `Overflow` is raised so the loss is observable instead of
//     silent memory corruption.
//   * In the (charmode == 1, mode == 1, threshold == 0, ndx >= 256) branch cv.jit loops
//     `for(i = 1; i < ndx; i++)`, leaving the LAST (largest) blob with a stale rank. Here
//     every blob is always ranked.
//   * cv.jit special-cases `ndx == 1` in the rank branch as `equiv[1] = 1` unconditionally,
//     so a lone blob is ranked 1 even when it fails the size filter. Here the size filter
//     always applies.
//   * cv.jit ranks with `qsort`, whose order for equal sizes is unspecified. Here ties are
//     broken by ascending blob index (stable sort), so the result is deterministic.
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
    // --- added after the initial port; every one of these defaults to the old behaviour
    halp::enum_t<Connectivity, "Connectivity"> connectivity;
    halp::enum_t<LabelMode, "Mode"> mode;
    // cv.jit's `charmode`: 0 -> long output, 1 -> char output. In char output every value
    // that does not fit a byte is clamped to 0 (cv.jit tests `> 255` then casts to char).
    halp::toggle<"Char output"> charmode;
    halp::enum_t<SizeFilter, "Size filter"> size_filter;
    // cv.jit hard-caps at 2048 blobs; same default here, but observable (see Overflow).
    halp::hslider_i32<"Max blobs", halp::range{1, 4096, 2048}> max_blobs;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image; // RGBA8 visualization (grey/colour per blob)
    // Single-channel label field, NORMALISED to [0,1] per the r32f contract (see the block
    // above): `Labels * Max label` is the blob id (0 = background, 1..count) in the default
    // Sequential/long mode, or the blob's mass / size rank in the other modes.
    halp::texture_output<"Labels", halp::r32f_texture> labels;
    // The largest value present in `Labels` before normalisation; 0 if the field is empty.
    // Multiply `Labels` by it to recover cv.jit's output-matrix values.
    halp::val_port<"Max label", float> max_label;
    halp::val_port<"Count", int> count;
    // True when at least one blob was discarded because `Max blobs` was reached.
    halp::val_port<"Overflow", bool> overflow;
  } outputs;

  void operator()() noexcept;
};
}
