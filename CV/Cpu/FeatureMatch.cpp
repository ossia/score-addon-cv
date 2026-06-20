#include "FeatureMatch.hpp"

#include <CV/Support/EigenImage.hpp>

#include <utility>
#include <vector>

namespace cv
{
void FeatureMatch::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || !in.width || !in.height)
    return;

  const int W = in.width;
  const int H = in.height;
  const auto src = cv_support::as_rgba(in);

  // Detect + describe the current frame.
  std::vector<cv_support::kp> cur;
  cv_support::detectAndDescribe(
      src.data, W, H, inputs.threshold.value, inputs.max_features.value, m_gray, cur);

  auto& out = outputs.matches.value;
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

    std::vector<cv_support::match_result> matches;
    cv_support::matchRatio(cur, target, inputs.ratio.value, matches);

    out.reserve(matches.size());
    for(const auto& m : matches)
    {
      const auto& c = cur[static_cast<std::size_t>(m.cur)];
      const auto& p = target[static_cast<std::size_t>(m.prev)];
      feature_match fm;
      // `prev` reports the matched keypoint's position in the target set (previous frame, or
      // the stored reference when in reference mode).
      fm.prev = {p.x * invW, p.y * invH};
      fm.cur = {c.x * invW, c.y * invH};
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
