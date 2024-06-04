#pragma once

#include <CompVis/CV.hpp>
#include <CompVis/Geometry.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>

#include <cmath>
namespace CompVis
{
struct detected_object_v8
{
  std::string name;
  rect geometry;
  float probability{};
  std::vector<halp::xy_type<float>> keypoints;

  halp_field_names(name, geometry, probability, keypoints);
};

struct YOLOv8Pose;
struct YoloV8PoseDetector
{
public:
  halp_meta(name, "YOLOv8 Pose");
  halp_meta(c_name, "yolov8_pose");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "YOLOv8 authors, OpenCV");
  halp_meta(description, "Recognizer using DNN. Requires Cuda.");
  halp_meta(uuid, "115e2f43-2ec5-477f-bbe8-bc1c592850c7");

  struct
  {
    halp::texture_input<"In"> image;
    halp::lineedit<"Model", ""> model;

  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<detected_object_v8> value;
    } detection;
  } outputs;

  YoloV8PoseDetector() noexcept;
  ~YoloV8PoseDetector();

  void operator()();

private:
  std::unique_ptr<YOLOv8Pose> detector;
};
} // namespace CompVis
