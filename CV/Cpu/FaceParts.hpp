#pragma once

#include <CV/Cpu/PerspectivePoints.hpp> // cv::point2
#include <CV/Cpu/SolvePnP.hpp>          // cv::pnp_correspondence

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

namespace cv
{
// ---------------------------------------------------------------------------------------
// The 68-point landmark index table
// ---------------------------------------------------------------------------------------
// cv.jit ships this as a `coll` file, `misc/lbfmodel.yaml.map`, one `name, start count;`
// line per group, which both cv.jit.face.parts and cv.jit.face.rigidpoints load at
// loadbang time with `read lbfmodel.yaml.map`. It is a constant of the iBUG-68 landmark
// convention, not user data, so it is baked in here as a constexpr table instead of being
// read from disk at runtime. The names are kept byte-identical to the .map file so the two
// can be cross-checked.
struct landmark_group
{
  std::string_view name;
  int start;
  int count;
};

inline constexpr std::array<landmark_group, 9> face_landmark_map{
    {{"jaw", 0, 17},
     {"brow-l", 17, 5},
     {"brow-r", 22, 5},
     {"nose-ridge", 27, 4},
     {"nose-base", 31, 5},
     {"eye-l", 36, 6},
     {"eye-r", 42, 6},
     {"lips", 48, 12},
     {"mouth", 60, 8}}};

// 17 + 5 + 5 + 4 + 5 + 6 + 6 + 12 + 8 == 68.
inline constexpr int face_landmark_count = 68;

// Face landmark splitter (cv.jit.face.parts).
//
// The abstraction is a fan of nine `jit.submatrix` objects driven by
// `offset <start> 0, dim <count> <n>` messages whose start/count come from
// lbfmodel.yaml.map, distributed by a `cycle 9`; the nine outlets follow the order of the
// map file. This object does the same slicing with plain index arithmetic and emits the
// nine groups on nine separate list outlets, in that same order.
//
// Out-of-range handling: cv.jit relies on jit.submatrix clamping. Here a group is emitted
// only when the input actually contains all of its landmarks (start + count <= size);
// otherwise that one group comes out empty. A 68-point (or longer) list therefore fills
// all nine outlets, a 30-point list fills only jaw / brow-l / brow-r, and an empty list
// empties all nine. Nothing is ever read out of bounds.
struct FaceParts
{
  halp_meta(name, "Face parts");
  halp_meta(c_name, "cv_face_parts");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Splits a 68-point facial landmark list (iBUG-68 / lbfmodel) into the nine "
      "cv.jit.face.parts groups: jaw, brow-l, brow-r, nose-ridge, nose-base, eye-l, "
      "eye-r, lips, mouth. Groups whose landmarks are not all present come out empty.");
  halp_meta(uuid, "c1a70000-0040-4a00-9000-000000000002");

  struct
  {
    struct
    {
      halp_meta(name, "Landmarks");
      std::vector<point2> value;
    } landmarks;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Jaw");
      std::vector<point2> value;
    } jaw; // map: jaw, 0 17
    struct
    {
      halp_meta(name, "Brow L");
      std::vector<point2> value;
    } brow_l; // map: brow-l, 17 5
    struct
    {
      halp_meta(name, "Brow R");
      std::vector<point2> value;
    } brow_r; // map: brow-r, 22 5
    struct
    {
      halp_meta(name, "Nose ridge");
      std::vector<point2> value;
    } nose_ridge; // map: nose-ridge, 27 4
    struct
    {
      halp_meta(name, "Nose base");
      std::vector<point2> value;
    } nose_base; // map: nose-base, 31 5
    struct
    {
      halp_meta(name, "Eye L");
      std::vector<point2> value;
    } eye_l; // map: eye-l, 36 6
    struct
    {
      halp_meta(name, "Eye R");
      std::vector<point2> value;
    } eye_r; // map: eye-r, 42 6
    struct
    {
      halp_meta(name, "Lips");
      std::vector<point2> value;
    } lips; // map: lips, 48 12
    struct
    {
      halp_meta(name, "Mouth");
      std::vector<point2> value;
    } mouth; // map: mouth, 60 8
  } outputs;

  void operator()() noexcept
  {
    const auto& lm = inputs.landmarks.value;

    slice(lm, face_landmark_map[0], outputs.jaw.value);
    slice(lm, face_landmark_map[1], outputs.brow_l.value);
    slice(lm, face_landmark_map[2], outputs.brow_r.value);
    slice(lm, face_landmark_map[3], outputs.nose_ridge.value);
    slice(lm, face_landmark_map[4], outputs.nose_base.value);
    slice(lm, face_landmark_map[5], outputs.eye_l.value);
    slice(lm, face_landmark_map[6], outputs.eye_r.value);
    slice(lm, face_landmark_map[7], outputs.lips.value);
    slice(lm, face_landmark_map[8], outputs.mouth.value);
  }

private:
  static void slice(
      const std::vector<point2>& lm, const landmark_group& g,
      std::vector<point2>& out) noexcept
  {
    const std::size_t start = static_cast<std::size_t>(g.start);
    const std::size_t count = static_cast<std::size_t>(g.count);
    if(start + count > lm.size())
    {
      out.clear();
      return;
    }
    out.assign(lm.begin() + static_cast<std::ptrdiff_t>(start),
               lm.begin() + static_cast<std::ptrdiff_t>(start + count));
  }
};

// ---------------------------------------------------------------------------------------
// cv.jit.face.rigidpoints
// ---------------------------------------------------------------------------------------
// The abstraction's comment says "extracts 4 face landmarks" and then lists six of them;
// the implementation (a `p gen_offsets` subpatcher feeding six `jit.submatrix` into a
// `jit.glue @columns 6`) emits six. gen_offsets derives each index from the map:
//
//   outlet 1  nose-ridge.start                     = 27  -> nose top (bridge)
//   outlet 2  nose-base.start  + nose-base.count/2 = 31 + 5/2 = 31 + 2 = 33  (integer /2)
//   outlet 3  eye-l.start                          = 36  -> left-eye outer corner
//   outlet 4  eye-r.start      + eye-r.count/2     = 42 + 6/2 = 42 + 3 = 45
//   outlet 5  jaw.start        + 2                 = 0 + 2  = 2   -> left jaw
//   outlet 6  jaw.start        + (jaw.count - 3)   = 0 + 14 = 14  -> right jaw
//
// and jit.glue concatenates them in that outlet order, so the emitted 6-point set is
// (27, 33, 36, 45, 2, 14) -- which is the order kept here. The indices are hard-coded
// below rather than recomputed, but the derivation above is what they come from, so the
// `/2` truncations (5/2 = 2, not 2.5) are reproduced, not accidents.
inline constexpr std::array<int, 6> face_rigid_indices{27, 33, 36, 45, 2, 14};

// Shortest landmark list that contains every rigid index (max index + 1 == 46).
inline constexpr std::size_t face_rigid_min_landmarks = [] {
  int m = 0;
  for(int i : face_rigid_indices)
    m = (i > m) ? i : m;
  return static_cast<std::size_t>(m) + 1u;
}();
static_assert(face_rigid_min_landmarks == 46u);

// One 3D model point of the rigid head model.
struct model_point3
{
  float x, y, z;

  halp_field_names(x, y, z);
};

// Approximate anthropometric 3D positions, in millimetres, of the six rigid landmarks,
// in the SAME order as face_rigid_indices. Axis convention matches image space so that it
// composes with SolvePnP's pinhole model directly: X right, Y down, Z away from the camera
// (i.e. into the head). The origin is the base of the nose (landmark 33), the most forward
// point of the set; the model is left/right symmetric, which is what makes a zero-yaw pose
// come out as zero.
//
// These are averages, not a measurement of any particular head: a head-pose chain wants
// *some* rigid model, and hard-coding a sane one is what makes
// `FaceRigidPoints -> SolvePnP` usable out of the box. Override it with the "Model" list
// inlet when a calibrated model is available; "Model scale" rescales the built-in one
// (rotation is scale-invariant, translation is not).
inline constexpr std::array<model_point3, 6> face_rigid_model{
    {{0.f, -55.f, 25.f},    // 27 nose top / bridge, between the eyes
     {0.f, 0.f, 0.f},       // 33 nose base centre  (origin)
     {-65.f, -60.f, 50.f},  // 36 eye-l outer corner (image left)
     {65.f, -60.f, 50.f},   // 45 eye-r outer corner (image right)
     {-78.f, 30.f, 95.f},   // 2  jaw, image left
     {78.f, 30.f, 95.f}}};  // 14 jaw, image right

// Rigid head-pose landmarks (cv.jit.face.rigidpoints).
//
// Extracts the six landmarks that move rigidly with the skull -- the ones that are not
// deformed by expression -- from a 68-point landmark list, in cv.jit's order.
//
// Two outlets:
//  - "Points" is the plain 6-point list, the equivalent of the abstraction's single outlet.
//  - "Correspondences" pairs those six image points with the 3D rigid model above and is
//    typed `std::vector<cv::pnp_correspondence>`, i.e. exactly SolvePnP's "Points" list
//    input, so `FaceRigidPoints -> SolvePnP` is a direct cable and yields head pose.
//    SolvePnP has no image-point-only inlet (its list element interleaves the 3D and the
//    2D side into one struct), which is why the 3D side has to be supplied here.
//
// Short input: all six indices are needed for a pose, so if the list does not reach
// landmark 45 both outlets come out empty rather than partially filled.
struct FaceRigidPoints
{
  halp_meta(name, "Face rigid points");
  halp_meta(c_name, "cv_face_rigid_points");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "Extracts the 6 rigid head-pose landmarks (27, 33, 36, 45, 2, 14) from a 68-point "
      "facial landmark list (cv.jit.face.rigidpoints). Also emits them paired with a 3D "
      "rigid head model, ready to plug into Solve PnP's Points input for head pose.");
  halp_meta(uuid, "c1a70000-0040-4a00-9000-000000000003");

  struct
  {
    struct
    {
      halp_meta(name, "Landmarks");
      std::vector<point2> value;
    } landmarks;

    // Optional 3D model override. Used only when it holds exactly 6 points, in the same
    // order as face_rigid_indices; otherwise the built-in model is used. (Same
    // "non-empty list replaces the default" convention as Homography and SolvePnP.)
    struct
    {
      halp_meta(name, "Model");
      std::vector<model_point3> value;
    } model;

    // Multiplies the model coordinates. The built-in model is in millimetres, so this is
    // how the translation reported downstream by SolvePnP gets its unit.
    halp::hslider_f32<"Model scale", halp::range{0.001f, 1000.f, 1.f}> model_scale;
  } inputs;

  struct
  {
    struct
    {
      halp_meta(name, "Points");
      std::vector<point2> value;
    } points;

    // Directly pluggable into SolvePnP's "Points" inlet.
    struct
    {
      halp_meta(name, "Correspondences");
      std::vector<pnp_correspondence> value;
    } correspondences;
  } outputs;

  void operator()() noexcept
  {
    const auto& lm = inputs.landmarks.value;
    auto& pts = outputs.points.value;
    auto& corr = outputs.correspondences.value;

    if(lm.size() < face_rigid_min_landmarks)
    {
      pts.clear();
      corr.clear();
      return;
    }

    const bool custom = inputs.model.value.size() == face_rigid_indices.size();
    const float s = inputs.model_scale.value;

    pts.resize(face_rigid_indices.size());
    corr.resize(face_rigid_indices.size());
    for(std::size_t k = 0; k < face_rigid_indices.size(); ++k)
    {
      const point2 p = lm[static_cast<std::size_t>(face_rigid_indices[k])];
      pts[k] = p;

      const model_point3 m = custom ? inputs.model.value[k] : face_rigid_model[k];
      corr[k] = pnp_correspondence{m.x * s, m.y * s, m.z * s, p.x, p.y};
    }
  }
};
}
