#include "FeatureMatch.hpp"

#include <CV/Support/EigenImage.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace cv
{
namespace
{
// The remaining keypoint planes (cv.jit's size / angle / response / octave) for the "prev"
// side of a match.
// `size` must be reported in the same units as the position, and as OrbFeatures reports it:
// cv.jit scales SIZE by max(scale_x, scale_y) = max(invW, invH). Without this the texture
// path would emit raw base-image pixels while the two-set path (fed by OrbFeatures) emits
// scaled values -- the same field in two different units.
inline void fill_prev(feature_match& fm, const cv_support::kp& k, float invW, float invH)
{
  fm.prev = {k.x * invW, k.y * invH};
  fm.prev_size = k.size * std::max(invW, invH);
  fm.prev_angle = k.angle;
  fm.prev_response = k.response;
  fm.prev_octave = k.octave;
}
inline void fill_cur(feature_match& fm, const cv_support::kp& k, float invW, float invH)
{
  fm.cur = {k.x * invW, k.y * invH};
  fm.cur_size = k.size * std::max(invW, invH);
  fm.cur_angle = k.angle;
  fm.cur_response = k.response;
  fm.cur_octave = k.octave;
}
inline void fill_prev(feature_match& fm, const keypoint& k)
{
  fm.prev = k.position;
  fm.prev_size = k.size;
  fm.prev_angle = k.angle;
  fm.prev_response = k.response;
  fm.prev_octave = k.octave;
}
inline void fill_cur(feature_match& fm, const keypoint& k)
{
  fm.cur = k.position;
  fm.cur_size = k.size;
  fm.cur_angle = k.angle;
  fm.cur_response = k.response;
  fm.cur_octave = k.octave;
}
}

void FeatureMatch::operator()() noexcept
{
  auto& out = outputs.matches.value;

  const auto& A = inputs.set_a.value;
  const auto& B = inputs.set_b.value;

  // ------------------------------------------------------------------ two-set mode
  // As soon as either list input carries anything, this object is a pure matcher over two
  // independently produced sets (cv.jit.keypoints.match's real signature). NO detection is
  // performed and the texture input is not read at all.
  if(!A.empty() || !B.empty())
  {
    out.clear();

    // cv.jit's silent tolerance: an empty set on either side emits an empty result rather
    // than an error, because this state occurs constantly while patching.
    if(!A.empty() && !B.empty())
    {
      constexpr auto get
          = [](const keypoint& k) -> const cv_support::descriptor& { return k.desc; };

      // A is the query set, B the train set (cv.jit: queryIdx indexes keypoints1,
      // trainIdx indexes keypoints2). matchRatioBy guards train.size() < 2, which is the
      // out-of-bounds read cv.jit has.
      std::vector<cv_support::match_result> matches;
      cv_support::matchRatioBy(A, B, get, get, inputs.ratio.value, matches);

      out.reserve(matches.size());
      for(const auto& m : matches)
      {
        feature_match fm;
        fill_prev(fm, A[static_cast<std::size_t>(m.query)]);
        fill_cur(fm, B[static_cast<std::size_t>(m.train)]);
        fm.distance = static_cast<float>(m.dist);
        out.push_back(fm);
      }
    }

    outputs.count = static_cast<int>(out.size());
    return;
  }

  // ------------------------------------------------------------------ texture mode
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || !in.width || !in.height)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  // Detect + describe the current frame. Single-scale: multi-octave detection lives in
  // OrbFeatures, which feeds the list inputs above.
  std::vector<cv_support::kp> cur;
  cv_support::detectAndDescribe(
      src.data, W, H, inputs.threshold.value, inputs.max_features.value, m_gray, cur);

  out.clear();

  // Rising edge of "Set reference": snapshot the current frame's keypoints as the fixed
  // reference/template set. Subsequent frames will match against this instead of the
  // previous frame, enabling object/template recognition.
  const bool setRef = inputs.set_reference.value;
  if(setRef && !m_prevSetRef)
  {
    m_ref = cur; // copy (cur is still needed below for prev bookkeeping)
    m_refW = W;
    m_refH = H;
    m_hasRef = true;
  }
  m_prevSetRef = setRef;

  // Choose the set to match against: the stored reference (if any and same resolution),
  // otherwise the previous frame (temporal mode, the default).
  const bool useRef = m_hasRef && m_refW == W && m_refH == H;
  const std::vector<cv_support::kp>& target = useRef ? m_ref : m_prev;
  const bool targetValid
      = useRef ? !m_ref.empty() : (!m_prev.empty() && m_prevW == W && m_prevH == H);

  // Positions are normalised so a resolution change just means no match this frame.
  if(targetValid)
  {
    const float invW = 1.f / W;
    const float invH = 1.f / H;

    // QUERY = the target (previous frame / stored reference), TRAIN = the current frame.
    //
    // Lowe's ratio test is ASYMMETRIC: every *query* keypoint yields at most one accepted
    // match, while a *train* keypoint can be claimed by many queries. So the direction decides
    // which side of `feature_match` carries the uniqueness guarantee, and the two modes of this
    // object must agree or the same output field means two different things. The two-set path
    // above queries with set A (-> fm.prev), which is what cv.jit does
    // (cv.jit.keypoints.match.cpp: knnMatch(desc1, desc2, ...), queryIdx indexes set 1, the
    // template side), so the target/"prev" side is the query here too. It is also the right
    // direction for the reference-matching use case: template -> scene, i.e. each template
    // feature is placed at most once in the incoming frame.
    std::vector<cv_support::match_result> matches;
    cv_support::matchRatio(target, cur, inputs.ratio.value, matches);

    out.reserve(matches.size());
    for(const auto& m : matches)
    {
      const auto& p = target[static_cast<std::size_t>(m.query)];
      const auto& c = cur[static_cast<std::size_t>(m.train)];
      feature_match fm;
      // `prev` reports the matched keypoint's position in the target set (previous frame, or
      // the stored reference when in reference mode).
      fill_prev(fm, p, invW, invH);
      fill_cur(fm, c, invW, invH);
      fm.distance = static_cast<float>(m.dist);
      out.push_back(fm);
    }
  }

  // Current becomes previous for the next frame (temporal mode bookkeeping is always kept,
  // so the matcher can fall back to it if the reference is cleared by a resolution change).
  m_prev = std::move(cur);
  m_prevW = W;
  m_prevH = H;

  outputs.count = static_cast<int>(out.size());
}
}
