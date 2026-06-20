#include "ChessboardCorners.hpp"

#include <CV/Support/Chessboard.hpp>
#include <CV/Support/EigenImage.hpp>

namespace cv
{
void ChessboardCorners::operator()() noexcept
{
  auto& in = inputs.image.texture;
  if(!in.changed || !in.bytes || !in.width || !in.height)
    return;

  const auto src = cv_support::as_rgba(in);

  cv_support::ChessboardParams p;
  p.cols = inputs.cols.value;
  p.rows = inputs.rows.value;
  p.threshold = inputs.threshold.value;

  const auto R = cv_support::find_chessboard_corners(src, p);

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
}
}
