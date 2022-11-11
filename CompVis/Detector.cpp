#include "Detector.hpp"

#undef READ
#undef WRITE
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#define FMT_HEADER_ONLY 1
#include <fmt/format.h>

namespace CompVis
{
using namespace cv;
using namespace dnn;
struct YOLO
{
  std::string class_file;
  std::string weights_file;
  std::string config_file;
  std::vector<std::string> m_classes;
  std::unique_ptr<cv::dnn::Net> m_net;
  std::unique_ptr<cv::dnn::DetectionModel> m_model;

  YOLO() { }

  void reload(
      const std::string& classf,
      const std::string& configf,
      const std::string& weightf)
  {
    if (classf != class_file)
    {
      class_file = classf;
      m_classes.clear();

      std::ifstream file{class_file};
      std::string line;
      while (std::getline(file, line))
        m_classes.push_back(line);
    }

    bool rebuild = false;
    if (weightf != weights_file)
    {
      weights_file = weightf;
      rebuild = true;
    }

    if (configf != config_file)
    {
      config_file = configf;
      rebuild = true;
    }

    if (rebuild)
    {
      m_net = std::make_unique<cv::dnn::Net>(
          readNetFromDarknet(config_file, weights_file));
      m_net->setPreferableBackend(DNN_BACKEND_CUDA);
      m_net->setPreferableTarget(DNN_TARGET_CUDA);

      m_model = std::make_unique<cv::dnn::DetectionModel>(*m_net);
      m_model->setInputParams(
          1 / 255.0, cv::Size(416, 416), cv::Scalar(), true);
    }
  }
};

YoloV4Detector::YoloV4Detector() noexcept
{
  outputs.image.create(1, 1);
}

YoloV4Detector::~YoloV4Detector() { }

void YoloV4Detector::operator()()
{
  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  if (!this->detector)
    this->detector = std::make_unique<YOLO>();

  this->detector->reload(
      this->inputs.classes, this->inputs.config, this->inputs.weights);

  // RGBA pixels to cv::Mat
  cv::Mat img_source(in_tex.height, in_tex.width, CV_8UC4, in_tex.bytes);

  // Convert to the right format
  cv::cvtColor(img_source, img_source, cv::COLOR_BGRA2RGB);
  cv::Mat img;
  cv::flip(img_source, img, 0);

  // Perform detection
  detector->m_model->detect(img, classIds, scores, boxes, 0.6, 0.4);

  // Save detected objects
  outputs.detection.value.clear();
  for (std::size_t i = 0; i < classIds.size(); i++)
  {
    // Store them in the output port
    auto& classname = detector->m_classes[classIds[i]];
    outputs.detection.value.push_back(
        {.name = classname,
         .geometry
         = {.x = (float)boxes[i].x,
            .y = (float)boxes[i].y,
            .w = (float)boxes[i].width,
            .h = (float)boxes[i].height},
         .probability = scores[i]});

    // Draw them on the visualization
    rectangle(img, boxes[i], Scalar(0, 255, 0), 2);
    const auto& text = fmt::format("{}: {:.2f}", classname, scores[i]);
    putText(
        img,
        text,
        Point(boxes[i].x, boxes[i].y - 5),
        cv::FONT_HERSHEY_SIMPLEX,
        1,
        cv::Scalar(0, 255, 0),
        2);
  }

  // Write the modified texture back
  outputs.image.create(in_tex.width, in_tex.height);
  cv::cvtColor(img, img, cv::COLOR_RGB2BGRA);

  outputs.image.texture
      = {.bytes = img.data,
         .width = in_tex.width,
         .height = in_tex.height,
         .changed = true};
}

}
