#pragma once

// For cv::polar_codec — the flow field is signed and must use the addon's ONE bipolar
// encoding, not a second scheme of its own. The r32f [0,1] contract is documented at the
// top of that header.
#include <CV/Cpu/CartoPol.hpp>

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/texture.hpp>

#include <cstdint>
#include <vector>

namespace cv
{
/* Dense Horn-Schunck optical flow — the port of cv.jit.HSflow.
 *
 * cv.jit.HSflow is a thin wrapper around OpenCV 1.x's `cvCalcOpticalFlowHS`:
 *
 *   cvCalcOpticalFlowHS(previous, current, /usePrevious/ 0, flowX, flowY, lambda,
 *                       cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,
 *                                      maxIter, threshold));
 *
 * with attribute defaults `lambda = 0.001`, `maxiter = 3`, `threshold = 0.f`, a 1-plane
 * char 2D input and a 2-plane float32 output. This class reimplements the algorithm
 * itself (no OpenCV anywhere in this addon).
 *
 * ------------------------------------------------------------------ the iteration
 * With Ex, Ey, Et the spatio-temporal derivatives of the image pair and (ubar, vbar) the
 * local averages of the current flow estimate, each iteration is the classic 1981
 * Horn & Schunck update:
 *
 *   t = (Ex*ubar + Ey*vbar + Et) / (lambda^2 + Ex^2 + Ey^2)
 *   u = ubar - Ex * t
 *   v = vbar - Ey * t
 *
 * The averages use the standard Horn-Schunck kernel — 1/6 on the 4-neighbours, 1/12 on
 * the 4 diagonals — with border replication.
 *
 * LAMBDA CONVENTION (read this before tuning):
 *   There are two incompatible conventions in the literature for the regularisation
 *   weight in that denominator: `alpha^2 + Ex^2 + Ey^2` (a smoothness *weight*: bigger =
 *   smoother) and the Lagrange-multiplier form `1/lambda + Ex^2 + Ey^2` used by OpenCV
 *   (bigger = *less* smooth). **This object uses the first one**: `lambda` is the
 *   smoothness weight alpha, entering as `lambda^2`, so
 *
 *       a LARGER `Lambda` produces a SMOOTHER, more filled-in flow field,
 *
 *   which is the opposite direction from OpenCV's / cv.jit's `lambda`. The numeric value
 *   therefore does not transfer from a Max patch; only the default is kept (0.001) for
 *   documentation parity. Note that at that default the regulariser is effectively off
 *   (lambda^2 = 1e-6 against gradients of order 1e-1), i.e. cv.jit's default is a
 *   near-unregularised HS; raise `Lambda` into the 1e-2 .. 1e0 range to see the
 *   smoothness term do anything.
 *
 * IMAGE SCALE: the luminance is Rec.601 from the RGBA8 input, normalised to [0,1] (as in
 * CV/Cpu/OpticalFlowLK.cpp), so Ex/Ey are "grey levels per pixel" in [0,1] units. This is
 * the scale `Lambda` is measured against.
 *
 * OUTPUT: a dense per-pixel flow field split over two r32f textures (`Flow X`, `Flow Y`)
 * rather than cv.jit's single 2-plane float32 matrix, because a texture port here carries
 * one component set. Sign convention: `Flow X` is positive when the image content moved
 * towards increasing x (to the right) between the previous and the current frame; `Flow Y`
 * positive towards increasing y (downwards, the texture origin being top-left).
 *
 * ------------------------------------------------------------------- FLOW ENCODING
 * The flow is SIGNED and in pixels per frame, so it cannot be written to the texture raw:
 * score converts every r32f output to RGBA8 by interpreting the float as [0,1] (see the
 * contract block at the top of CV/Cpu/CartoPol.hpp), which would clamp every negative
 * component to black and saturate everything past 1 px/frame. The components are therefore
 * written with the addon's ONE bipolar encoding, `cv::polar_codec::encode_signed01`:
 *
 *     Flow X = 0.5 * (u / S) + 0.5        u = (2*Flow X - 1) * S
 *
 * where `S` is reported on the `Flow scale` value port, in pixels per frame. Zero flow is
 * exactly 0.5 and decodes back to exactly 0.
 *
 * `Flow scale` input control:
 *   * 0 (the DEFAULT) = AUTO: S is this frame's peak |component| (1 if the field is all
 *     zero). Nothing is ever clipped, and the pair (texture, value port) carries the field
 *     exactly. This is the mode a measurement chain wants.
 *   * > 0 = a FIXED scale in pixels/frame, for a stable visual mapping across frames or to
 *     match a fixed downstream `Range`. Components outside +/-S are clamped, and `Clipped`
 *     goes true so that the loss is observable instead of silent.
 *
 * CHAINING INTO CartoPol (the documented cv.jit use case, cv.jit.HSflow -> cv.jit.cartopol):
 *   `Flow X`/`Flow Y`  ---> CartoPol `X`/`Y`, `Signed input` ON
 *   `Flow scale` (value) ---> CartoPol `Range`
 * Both ends then agree on the same bipolar codec and the same scale. Before this was fixed
 * the two disagreed completely: HSflow wrote raw signed px/frame and CartoPol read a
 * bipolar 8-bit field, so every leftward/upward motion arrived as -Range.
 *
 * TERMINATION: stops after `Max iterations` sweeps, or earlier if `Threshold > 0` and the
 * largest per-pixel change of either component during a sweep fell below `Threshold`
 * (cv.jit passes `threshold` as the CV_TERMCRIT_EPS epsilon). The number of sweeps
 * actually performed is reported on the `Iterations` outlet (an addition — cv.jit has no
 * such outlet — so that early termination is observable).
 *
 * WARM START: cv.jit hardcodes `usePrevious = 0`, so every frame restarts the iteration
 * from a zero flow field; that is the default here too. The `Use previous` toggle enables
 * the warm start OpenCV would have done with `usePrevious = 1`; it is off by default for
 * cv.jit parity.
 *
 * ---------------------------------------------- deliberate deviations from cv.jit
 *  * FIRST FRAME. cv.jit keeps the previous frame in a Jitter matrix that is created,
 *    `_jit_sym_clear`ed (all zeroes) and only *then* filled at the end of the first
 *    matrix_calc. The first frame is therefore matched against an all-black image and
 *    emits a large spurious flow field. That is a bug; here the first frame (and the
 *    first frame after any dimension change) emits **exactly zero flow** and
 *    `Iterations = 0`.
 *  * INPUT TYPE. cv.jit errors out on anything but a 1-plane char 2D matrix. Score
 *    textures are RGBA8, so the input is reduced with Rec.601 luminance instead.
 *  * `lambda` direction and scale, see LAMBDA CONVENTION above.
 *  * AVERAGING KERNEL. cv.jit calls OpenCV 1.x's `cvCalcOpticalFlowHS`, whose local average
 *    is a plain 4-NEIGHBOUR MEAN (each of the up/down/left/right neighbours weighted 1/4,
 *    the diagonals ignored). This object uses the ORIGINAL Horn & Schunck 1981 kernel
 *    instead: 1/6 on the 4-neighbours and 1/12 on the 4 diagonals. The two are both
 *    unit-sum smoothing operators, but the 1981 one is slightly more isotropic, so the
 *    converged fields differ (the 1/12 diagonal term propagates flow across corners) and a
 *    per-pixel comparison against OpenCV would not match exactly. This is the classic
 *    textbook kernel and the one every published HS derivation uses; it is kept
 *    deliberately.
 *  * A non-finite result (which the lambda = 0 / flat-image degenerate case could
 *    otherwise produce) is clamped to 0 rather than propagated.
 *  * OUTPUT ENCODING, see FLOW ENCODING above (cv.jit emits raw float32 planes; a score
 *    texture cannot).
 */
struct HornSchunck
{
  halp_meta(name, "Optical flow (Horn-Schunck)");
  halp_meta(c_name, "cv_optical_flow_hs");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Dense Horn-Schunck optical flow (cv.jit.HSflow), float32 output.");
  halp_meta(uuid, "c1a70000-0044-4a00-9000-000000000001");

  struct
  {
    halp::texture_input<"In"> image;
    // cv.jit `maxiter`, default 3.
    halp::hslider_i32<"Max iterations", halp::range{1, 200, 3}> maxiter;
    // cv.jit `lambda`, default 0.001 — but see LAMBDA CONVENTION above: here it is the
    // smoothness weight (larger = smoother), entering the denominator squared.
    halp::hslider_f32<"Lambda", halp::range{0.f, 4.f, 0.001f}> lambda;
    // cv.jit `threshold`, default 0 (== no epsilon test, always run maxiter sweeps).
    halp::hslider_f32<"Threshold", halp::range{0.f, 1.f, 0.f}> threshold;
    // OpenCV's `usePrevious`, which cv.jit hardcodes to 0. Off = cv.jit behaviour.
    halp::toggle<"Use previous"> use_previous;
    // Pixels/frame that map to the ends of the bipolar encoding; 0 = auto (per-frame peak,
    // never clips). See FLOW ENCODING above.
    halp::hslider_f32<"Flow scale", halp::range{0.f, 64.f, 0.f}> flow_scale;
  } inputs;

  struct
  {
    // Dense flow, bipolar-encoded into [0,1] per the addon r32f contract:
    //     u_px_per_frame = (2 * Flow X - 1) * `Flow scale`
    halp::texture_output<"Flow X", halp::r32f_texture> dx;
    halp::texture_output<"Flow Y", halp::r32f_texture> dy;
    // Sweeps actually performed this frame (0 on the first frame / after a resize).
    halp::val_port<"Iterations", int> iterations;
    // Pixels/frame represented by the full swing of the two textures above. Wire into a
    // downstream CartoPol's `Range`.
    halp::val_port<"Flow scale", float> flow_scale;
    // True when a FIXED `Flow scale` was too small and at least one component was clamped.
    halp::val_port<"Clipped", bool> clipped;
  } outputs;

  void operator()() noexcept;

private:
  std::vector<float> m_prev; // previous frame luminance in [0,1], W*H
  std::vector<float> m_u;    // last emitted flow, kept for the `Use previous` warm start
  std::vector<float> m_v;
  int m_w = 0, m_h = 0;
  // Scratch, kept as members so a steady-state frame allocates nothing.
  std::vector<float> m_ex, m_ey, m_et, m_un, m_vn, m_cur;
};
}
