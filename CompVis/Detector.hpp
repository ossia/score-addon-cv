#pragma once

#include <CompVis/CV.hpp>
#include <CompVis/Geometry.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>
namespace CompVis
{
struct detected_object
{
  std::string name;
  rect geometry;
  float probability{};

  halp_field_names(name, geometry, probability);

  // static constexpr auto field_names()
  // {
  //   return std::array<std::string_view, 3>{"name", "geometry", "probability"};
  // }
  //halp_field_names
};

struct YOLO;
struct YoloV4Detector
{
public:
  halp_meta(name, "YOLOv4");
  halp_meta(c_name, "yolov4");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "YOLOv4 authors, OpenCV");
  halp_meta(description, "Recognizer using DNN. Requires Cuda.");
  halp_meta(uuid, "17d59b59-054d-4899-aaf9-fec0cff9f523");

  struct
  {
    halp::texture_input<"In"> image;

    halp::lineedit<"Configuration", ""> config;
    halp::lineedit<"Weights", ""> weights;
    halp::lineedit<"Classes", ""> classes;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

    struct
    {
      halp_meta(name, "Detection");
      std::vector<detected_object> value;
    } detection;
  } outputs;

  YoloV4Detector() noexcept;
  ~YoloV4Detector();

  void operator()();

private:
  std::vector<int> classIds;
  std::vector<float> scores;
  std::vector<cv::Rect> boxes;
  std::unique_ptr<YOLO> detector;
};
}
