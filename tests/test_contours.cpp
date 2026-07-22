// Tests for cv::Contours: raw boundary point list, Douglas-Peucker simplification,
// hole (inner border) detection with hierarchy, Jacob's stopping criterion and truncation
// reporting.
//
// ---------------------------------------------------------------------------------------
// DERIVATIONS used below (all hand-computed; x = column, y = row, y grows downwards).
//
// [A] Moore boundary of a solid k x k square = its 4k-4 ring pixels, traversal starting at
//     the top-left corner and running clockwise on screen (east along the top row first).
//     E.g. k=7 -> 24 points, k=9 -> 32 points, k=16 -> 60 points, k=4 -> 12 points.
//
// [B] Boundary of a 1-pixel-tall bar of length k: the tracer runs east along the row
//     (k points) then walks back west over the same pixels, excluding both ends
//     (k-2 points) -> 2k-2 points. k=10 -> 18 points.
//
// [C] Inner (hole) border of a square hole [hx0..hx1] x [hy0..hy1] of side s: the trace
//     hugs the hole with 8-connectivity, which cuts the corners. It is the octagon
//       (hx0-1, hy0 .. hy1)      left column,   s points
//       (hx0 .. hx1, hy1+1)      bottom row,    s points
//       (hx1+1, hy1 .. hy0)      right column,  s points
//       (hx1 .. hx0, hy0-1)      top row,       s points
//     -> 4*s points, traversed counter-clockwise on screen (negative shoelace), i.e.
//     opposite to an outer border (positive shoelace). s=2 -> 8 points, s=8 -> 32 points.
//
// [D] approxPolyDP on a closed curve splits the ring at its two most distant points and
//     runs RDP on the two chains. For a square the split points are two opposite corners,
//     the max deviation inside each chain is the remaining corner at (k-1)/sqrt(2) pixels,
//     and every sub-chain is then perfectly straight -> exactly the 4 corners survive for
//     any 1 < epsilon < (k-1)/sqrt(2).
//
// [E] The "bowtie": five pixels pinched at their topmost-leftmost pixel P,
//         . X .        P     = (3,2)
//         X . X        left  = (2,3),(2,4)
//         X . X        right = (4,3),(4,4)
//     The boundary walk legitimately passes through P twice - once between the two lobes,
//     once to close the loop:
//        (3,2) (4,3) (4,4) (4,3) (3,2) (2,3) (2,4) (2,3)  -> 8 points.
//     The "stop as soon as the start pixel is seen again" approximation stops at the 4th
//     point and reports only the right lobe. This is the regression test for the proper
//     Jacob criterion (stop when the start pixel is *left* in the original direction).
// ---------------------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "TestImage.hpp"

#include <CV/Cpu/Contours.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using Catch::Approx;
using namespace cvtest;

namespace
{
struct PxPt
{
  int x{}, y{}, c{};
};

// De-normalise the emitted point list back to integer pixel coordinates.
std::vector<PxPt> pixels(const cv::Contours& obj, int W, int H)
{
  std::vector<PxPt> v;
  v.reserve(obj.outputs.points.value.size());
  for(const auto& p : obj.outputs.points.value)
    v.push_back(PxPt{
        static_cast<int>(std::lround(p.x * W)), static_cast<int>(std::lround(p.y * H)),
        p.contour});
  return v;
}

std::vector<PxPt> ofContour(const std::vector<PxPt>& all, int c)
{
  std::vector<PxPt> v;
  for(const auto& p : all)
    if(p.c == c)
      v.push_back(p);
  return v;
}

// Signed area x2 of the closed polygon (positive = clockwise on screen, y downwards).
double shoelace2(const std::vector<PxPt>& p)
{
  double s = 0.0;
  const std::size_t n = p.size();
  for(std::size_t i = 0; i < n; ++i)
  {
    const auto& a = p[i];
    const auto& b = p[(i + 1) % n];
    s += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
  }
  return s;
}

bool closedLoop8Adjacent(const std::vector<PxPt>& p)
{
  const std::size_t n = p.size();
  if(n < 2)
    return false;
  for(std::size_t i = 0; i < n; ++i)
  {
    const auto& a = p[i];
    const auto& b = p[(i + 1) % n];
    const int dx = std::abs(a.x - b.x), dy = std::abs(a.y - b.y);
    if(std::max(dx, dy) != 1)
      return false;
  }
  return true;
}
} // namespace

// ------------------------------------------------------------------------- point list
TEST_CASE("Contours point list traces a k x k square as 4k-4 ordered points", "[contours]")
{
  // [A] with k = 7: the ring of the square spanning 10..16 in both axes.
  cv::Contours obj;
  Image img(32, 32, 0);
  img.fillRect(10, 10, 7, 7, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto pts = pixels(obj, 32, 32);
  REQUIRE(pts.size() == 24); // 4*7 - 4

  CHECK(obj.outputs.contours.value[0].point_count == 24);
  for(const auto& p : pts)
    CHECK(p.c == 0);

  // Traversal order: starts at the top-left corner and goes east.
  CHECK(pts[0].x == 10);
  CHECK(pts[0].y == 10);
  CHECK(pts[1].x == 11);
  CHECK(pts[1].y == 10);

  // Closed loop, every consecutive pair (including the wrap) 8-adjacent.
  CHECK(closedLoop8Adjacent(pts));

  // Every point is on the ring, and the whole ring is covered exactly once.
  for(const auto& p : pts)
  {
    const bool onRing
        = (p.x == 10 || p.x == 16 || p.y == 10 || p.y == 16) && p.x >= 10 && p.x <= 16
          && p.y >= 10 && p.y <= 16;
    CHECK(onRing);
  }
  auto sorted = pts;
  std::sort(sorted.begin(), sorted.end(), [](const PxPt& a, const PxPt& b) {
    return a.y != b.y ? a.y < b.y : a.x < b.x;
  });
  const auto dup = std::adjacent_find(
      sorted.begin(), sorted.end(),
      [](const PxPt& a, const PxPt& b) { return a.x == b.x && a.y == b.y; });
  CHECK(dup == sorted.end());

  // Outer borders are clockwise on screen -> positive shoelace.
  CHECK(shoelace2(pts) > 0.0);
  CHECK(obj.outputs.truncated.value == false);
}

TEST_CASE("Contours tags two disjoint blobs with contiguous ordinals", "[contours]")
{
  // [A] k=4 -> 12 points (blob A, scanned first), k=6 -> 20 points (blob B).
  cv::Contours obj;
  Image img(32, 32, 0);
  img.fillRect(2, 2, 4, 4, 255);
  img.fillRect(20, 20, 6, 6, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 2);
  const auto pts = pixels(obj, 32, 32);
  REQUIRE(pts.size() == 12 + 20);

  const auto c0 = ofContour(pts, 0);
  const auto c1 = ofContour(pts, 1);
  CHECK(c0.size() == 12);
  CHECK(c1.size() == 20);
  CHECK(obj.outputs.contours.value[0].point_count == 12);
  CHECK(obj.outputs.contours.value[1].point_count == 20);

  // Ordinals are contiguous from 0 and index into the summary list; points of one contour
  // are consecutive in the flat list.
  int expected = 0;
  for(std::size_t i = 0; i < pts.size(); ++i)
  {
    if(i > 0 && pts[i].c != pts[i - 1].c)
      ++expected;
    CHECK(pts[i].c == expected);
  }
  CHECK(expected == obj.outputs.count.value - 1);

  // Blob A is the one scanned first.
  CHECK(c0[0].x == 2);
  CHECK(c0[0].y == 2);
  CHECK(c1[0].x == 20);
  CHECK(c1[0].y == 20);
}

// -------------------------------------------------------------------- Douglas-Peucker
TEST_CASE("Contours epsilon collapses a straight run to its endpoints", "[contours]")
{
  // [B]: a 1px-tall, 10px-wide bar at row 10 spanning columns 4..13 -> 18 raw points.
  cv::Contours obj;
  Image img(32, 32, 0);
  img.fillRect(4, 10, 10, 1, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;

  obj.inputs.epsilon.value = 0.f;
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(pixels(obj, 32, 32).size() == 18); // 2*10 - 2

  obj.inputs.epsilon.value = 1.f;
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  const auto pts = pixels(obj, 32, 32);
  REQUIRE(pts.size() == 2);
  CHECK(pts[0].x == 4);
  CHECK(pts[0].y == 10);
  CHECK(pts[1].x == 13);
  CHECK(pts[1].y == 10);
  CHECK(obj.outputs.contours.value[0].point_count == 2);
}

TEST_CASE("Contours epsilon reduces a square to its four corners", "[contours]")
{
  // [D] with k=9 (columns/rows 8..16): 32 raw points, 4 corners at epsilon = 1.5
  // (1 < 1.5 < 8/sqrt(2) = 5.66).
  cv::Contours obj;
  Image img(32, 32, 0);
  img.fillRect(8, 8, 9, 9, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;

  SECTION("epsilon = 0 leaves the point count untouched")
  {
    obj.inputs.epsilon.value = 0.f;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    CHECK(pixels(obj, 32, 32).size() == 32); // 4*9 - 4
    CHECK(obj.outputs.contours.value[0].point_count == 32);
  }

  SECTION("epsilon = 1.5 keeps exactly the corners, in traversal order")
  {
    obj.inputs.epsilon.value = 1.5f;
    obj();
    REQUIRE(obj.outputs.count.value == 1);
    const auto pts = pixels(obj, 32, 32);
    REQUIRE(pts.size() == 4);
    CHECK(pts[0].x == 8);
    CHECK(pts[0].y == 8); // top-left
    CHECK(pts[1].x == 16);
    CHECK(pts[1].y == 8); // top-right
    CHECK(pts[2].x == 16);
    CHECK(pts[2].y == 16); // bottom-right
    CHECK(pts[3].x == 8);
    CHECK(pts[3].y == 16); // bottom-left
    CHECK(obj.outputs.contours.value[0].point_count == 4);
    // Still clockwise, and the summary is that of the simplified polygon: an 8x8 quad.
    CHECK(shoelace2(pts) == Approx(2.0 * 64.0));
    CHECK(obj.outputs.contours.value[0].area == Approx(64.f / (32.f * 32.f)));
  }
}

// ------------------------------------------------------------------------------ holes
TEST_CASE("Contours reports the inner border of an annulus", "[contours]")
{
  // Filled 8x8 square (4..11) with a 2x2 hole (7..8).
  // [A] outer ring: 4*8-4 = 28 points. [C] hole with s=2: 8 points.
  cv::Contours obj;
  Image img(16, 16, 0);
  img.fillRect(4, 4, 8, 8, 255);
  img.fillRect(7, 7, 2, 2, 0);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;

  SECTION("Find holes on (default) yields the outer contour and the hole")
  {
    REQUIRE(obj.inputs.find_holes.value == true); // default matches cv.jit's RETR_TREE
    obj();

    REQUIRE(obj.outputs.count.value == 2);
    const auto& outer = obj.outputs.contours.value[0];
    const auto& hole = obj.outputs.contours.value[1];

    CHECK(outer.is_hole == 0);
    CHECK(outer.parent == -1);
    CHECK(outer.point_count == 28);

    CHECK(hole.is_hole == 1);
    CHECK(hole.parent == 0);
    CHECK(hole.point_count == 8);

    const auto pts = pixels(obj, 16, 16);
    REQUIRE(pts.size() == 28 + 8);
    const auto po = ofContour(pts, 0);
    const auto ph = ofContour(pts, 1);
    REQUIRE(po.size() == 28);
    REQUIRE(ph.size() == 8);

    // The hole is traversed the other way round.
    CHECK(shoelace2(po) > 0.0);
    CHECK(shoelace2(ph) < 0.0);

    // [C]: the exact octagon around the 2x2 hole, starting on its left column.
    const std::vector<PxPt> expected{{6, 7, 1}, {6, 8, 1}, {7, 9, 1}, {8, 9, 1},
                                     {9, 8, 1}, {9, 7, 1}, {8, 6, 1}, {7, 6, 1}};
    for(std::size_t i = 0; i < expected.size(); ++i)
    {
      CHECK(ph[i].x == expected[i].x);
      CHECK(ph[i].y == expected[i].y);
    }
    CHECK(closedLoop8Adjacent(ph));
  }

  SECTION("Find holes off yields only the outer contour")
  {
    obj.inputs.find_holes.value = false;
    obj();

    REQUIRE(obj.outputs.count.value == 1);
    CHECK(obj.outputs.contours.value[0].is_hole == 0);
    CHECK(obj.outputs.contours.value[0].point_count == 28);
    const auto pts = pixels(obj, 16, 16);
    CHECK(pts.size() == 28);
    for(const auto& p : pts)
      CHECK(p.c == 0);
  }
}

TEST_CASE("Contours builds the parent chain of a blob in a hole in a blob", "[contours]")
{
  // outer blob 2..17 (16x16), hole 6..13 (8x8), inner blob 9..10 (2x2).
  // [A] outer: 4*16-4 = 60 ; [C] hole s=8: 32 ; [A] inner: 4*2-4 = 4.
  cv::Contours obj;
  Image img(24, 24, 0);
  img.fillRect(2, 2, 16, 16, 255);
  img.fillRect(6, 6, 8, 8, 0);
  img.fillRect(9, 9, 2, 2, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;

  SECTION("full hierarchy")
  {
    obj();
    REQUIRE(obj.outputs.count.value == 3);
    const auto& v = obj.outputs.contours.value;

    CHECK(v[0].is_hole == 0);
    CHECK(v[0].parent == -1);
    CHECK(v[0].point_count == 60);

    CHECK(v[1].is_hole == 1);
    CHECK(v[1].parent == 0);
    CHECK(v[1].point_count == 32);

    CHECK(v[2].is_hole == 0);
    CHECK(v[2].parent == 1);
    CHECK(v[2].point_count == 4);

    // A parent always precedes its child.
    for(std::size_t i = 0; i < v.size(); ++i)
      CHECK(v[i].parent < static_cast<int>(i));

    const auto pts = pixels(obj, 24, 24);
    CHECK(pts.size() == 60 + 32 + 4);
    CHECK(shoelace2(ofContour(pts, 0)) > 0.0); // outer
    CHECK(shoelace2(ofContour(pts, 1)) < 0.0); // hole
    CHECK(shoelace2(ofContour(pts, 2)) > 0.0); // inner blob, outer border again
  }

  SECTION("with holes off the child re-parents to the nearest kept ancestor")
  {
    obj.inputs.find_holes.value = false;
    obj();
    REQUIRE(obj.outputs.count.value == 2);
    const auto& v = obj.outputs.contours.value;
    CHECK(v[0].is_hole == 0);
    CHECK(v[0].parent == -1);
    CHECK(v[0].point_count == 60);
    // The hole is traced but not emitted: the inner blob must not point at a dropped
    // contour, it points at the outer blob instead.
    CHECK(v[1].is_hole == 0);
    CHECK(v[1].parent == 0);
    CHECK(v[1].point_count == 4);
  }
}

TEST_CASE("Contours does not duplicate the inner border as an outer one", "[contours]")
{
  // A 1px-wide ring: outer and inner border run over the very same pixels, so a naive
  // "already visited" test loses one of them and a naive rescan reports the inner one
  // twice. Ring 4..9 (6x6 outline), hole 5..8 (4x4).
  cv::Contours obj;
  Image img(16, 16, 0);
  img.fillRect(4, 4, 6, 6, 255);
  img.fillRect(5, 5, 4, 4, 0);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 2);
  CHECK(obj.outputs.contours.value[0].is_hole == 0);
  CHECK(obj.outputs.contours.value[1].is_hole == 1);
  CHECK(obj.outputs.contours.value[1].parent == 0);
  CHECK(obj.outputs.contours.value[0].point_count == 20); // [A] 4*6-4
  CHECK(obj.outputs.contours.value[1].point_count == 16); // [C] s=4

  const auto pts = pixels(obj, 16, 16);
  CHECK(shoelace2(ofContour(pts, 0)) > 0.0);
  CHECK(shoelace2(ofContour(pts, 1)) < 0.0);
}

// ------------------------------------------------------------------- Jacob's criterion
TEST_CASE("Contours traces both lobes of a pinched shape", "[contours]")
{
  // [E]: the bowtie. The approximate stopping criterion reports 4 points (right lobe
  // only); the proper Jacob criterion reports all 8.
  cv::Contours obj;
  Image img(8, 8, 0);
  img.setGray(3, 2, 255);
  img.setGray(2, 3, 255);
  img.setGray(4, 3, 255);
  img.setGray(2, 4, 255);
  img.setGray(4, 4, 255);

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto pts = pixels(obj, 8, 8);
  REQUIRE(pts.size() == 8);

  const std::vector<PxPt> expected{{3, 2, 0}, {4, 3, 0}, {4, 4, 0}, {4, 3, 0},
                                   {3, 2, 0}, {2, 3, 0}, {2, 4, 0}, {2, 3, 0}};
  for(std::size_t i = 0; i < expected.size(); ++i)
  {
    CHECK(pts[i].x == expected[i].x);
    CHECK(pts[i].y == expected[i].y);
  }

  // Both lobes are covered: all five foreground pixels appear.
  for(const PxPt& fgp : std::vector<PxPt>{{3, 2, 0}, {2, 3, 0}, {2, 4, 0}, {4, 3, 0},
                                          {4, 4, 0}})
  {
    const bool found = std::any_of(pts.begin(), pts.end(), [&](const PxPt& p) {
      return p.x == fgp.x && p.y == fgp.y;
    });
    CHECK(found);
  }
  CHECK(obj.outputs.truncated.value == false);
}

// ------------------------------------------------------------------------- truncation
TEST_CASE("Contours reports truncation instead of silently cutting a contour", "[contours]")
{
  cv::Contours obj;
  Image img(32, 32, 0);
  img.fillRect(10, 10, 7, 7, 255); // 24 boundary points

  feed(obj.inputs.image, img);
  obj.inputs.threshold.value = 0.5f;
  obj.inputs.min_perimeter.value = 0;

  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.truncated.value == false);
  CHECK(obj.outputs.points.value.size() == 24);

  // Force the step guard below the real boundary length. The first contour is cut short
  // (and, the boundary labelling being incomplete, the rest of the square gets picked up
  // as further fragments) - which is exactly why this must not stay silent.
  obj.set_max_trace_steps(5);
  obj();
  REQUIRE(obj.outputs.count.value >= 1);
  CHECK(obj.outputs.truncated.value == true);
  CHECK(obj.outputs.contours.value[0].point_count < 24);

  // ... and it clears again on the next clean frame.
  obj.set_max_trace_steps(0);
  obj();
  CHECK(obj.outputs.truncated.value == false);
  CHECK(obj.outputs.points.value.size() == 24);
}

// ------------------------------------------------------------------------ edge cases
TEST_CASE("Contours on an empty image outputs nothing", "[contours]")
{
  cv::Contours obj;
  Image img(16, 16, 0);
  feed(obj.inputs.image, img);
  obj.inputs.min_perimeter.value = 0;
  obj();

  CHECK(obj.outputs.count.value == 0);
  CHECK(obj.outputs.contours.value.empty());
  CHECK(obj.outputs.points.value.empty());
  CHECK(obj.outputs.truncated.value == false);
}

TEST_CASE("Contours skips a single isolated pixel", "[contours]")
{
  // A 1-pixel boundary is degenerate (no polygon): it has always been dropped, and stays
  // dropped, whatever the min-perimeter setting.
  cv::Contours obj;
  Image img(16, 16, 0);
  img.setGray(5, 5, 255);
  feed(obj.inputs.image, img);
  obj.inputs.min_perimeter.value = 0;
  obj();

  CHECK(obj.outputs.count.value == 0);
  CHECK(obj.outputs.points.value.empty());
  CHECK(obj.outputs.truncated.value == false);
}

TEST_CASE("Contours handles a blob flush against the image border", "[contours]")
{
  // [A] k=4 anchored at the top-left corner: the padding keeps the tracer in bounds.
  cv::Contours obj;
  Image img(8, 8, 0);
  img.fillRect(0, 0, 4, 4, 255);
  feed(obj.inputs.image, img);
  obj.inputs.min_perimeter.value = 0;
  obj();

  REQUIRE(obj.outputs.count.value == 1);
  const auto pts = pixels(obj, 8, 8);
  REQUIRE(pts.size() == 12);
  CHECK(pts[0].x == 0);
  CHECK(pts[0].y == 0);
  CHECK(closedLoop8Adjacent(pts));
  CHECK(obj.outputs.contours.value[0].bbox.x == Approx(0.f));
  CHECK(obj.outputs.contours.value[0].bbox.y == Approx(0.f));
  CHECK(obj.outputs.contours.value[0].bbox.w == Approx(4.f / 8.f));
}

TEST_CASE("Contours handles a 1x1 image", "[contours]")
{
  cv::Contours obj;

  SECTION("black")
  {
    Image img(1, 1, 0);
    feed(obj.inputs.image, img);
    obj.inputs.min_perimeter.value = 0;
    obj();
    CHECK(obj.outputs.count.value == 0);
    CHECK(obj.outputs.points.value.empty());
  }

  SECTION("white")
  {
    Image img(1, 1, 255);
    feed(obj.inputs.image, img);
    obj.inputs.min_perimeter.value = 0;
    obj();
    // Single pixel -> degenerate boundary, dropped.
    CHECK(obj.outputs.count.value == 0);
    CHECK(obj.outputs.points.value.empty());
  }
}

TEST_CASE("Contours re-traces correctly when the input dimensions change", "[contours]")
{
  cv::Contours obj;
  Image big(32, 32, 0);
  big.fillRect(10, 10, 7, 7, 255);
  feed(obj.inputs.image, big);
  obj.inputs.min_perimeter.value = 0;
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  CHECK(obj.outputs.points.value.size() == 24);

  Image small(8, 8, 0);
  small.fillRect(2, 2, 4, 4, 255);
  feed(obj.inputs.image, small);
  obj();
  REQUIRE(obj.outputs.count.value == 1);
  const auto pts = pixels(obj, 8, 8);
  REQUIRE(pts.size() == 12);
  CHECK(pts[0].x == 2);
  CHECK(pts[0].y == 2);
}
