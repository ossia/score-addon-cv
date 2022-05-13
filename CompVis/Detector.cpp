#include "Detector.hpp"

#undef READ
#undef WRITE
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

namespace CompVis
{
using namespace cv;
using namespace dnn;
struct YOLO
{
  std::string class_file;// = "/home/jcelerier/yolo/coco.names";
  std::string weights_file;// = "/home/jcelerier/yolo/yolov4_new.weights";
  std::string config_file;// = "/home/jcelerier/yolo/yolov4.cfg";
  std::vector<std::string> m_classes;
  std::unique_ptr<cv::dnn::Net> m_net;
  std::unique_ptr<cv::dnn::DetectionModel> m_model;

  YOLO()
  {
  }

  void reload(const std::string& classf, const std::string& configf, const std::string& weightf)
  {
    if(classf != class_file)
    {
      class_file = classf;
      m_classes.clear();

      std::ifstream file{class_file};
      std::string line;
      while (std::getline(file, line))
       m_classes.push_back(line);
    }

    bool rebuild = false;
    if(weightf != weights_file)
    {
      weights_file = weightf;
      rebuild = true;
    }

    if(configf != config_file)
    {
      config_file = configf;
      rebuild = true;
    }

    if(rebuild)
    {
      m_net = std::make_unique<cv::dnn::Net>(readNetFromDarknet(config_file, weights_file));
      m_net->setPreferableBackend(DNN_BACKEND_CUDA);
      m_net->setPreferableTarget(DNN_TARGET_CUDA);

      m_model = std::make_unique<cv::dnn::DetectionModel>(*m_net);
      m_model->setInputParams(1 / 255.0, cv::Size(416, 416), cv::Scalar(), true);
    }
  }
};

YoloV4Detector::YoloV4Detector() noexcept
{
  outputs.image.create(1, 1);
}

YoloV4Detector::~YoloV4Detector()
{

}

void YoloV4Detector::operator()()
{
  auto& in = inputs.image.texture;
  if(!in.changed)
    return;

  if(!this->detector)
    this->detector = std::make_unique<YOLO>();

  this->detector->reload(this->inputs.classes, this->inputs.config, this->inputs.weights);

  // RGBA pixels to cv::Mat
  cv::Mat img_source(in.height, in.width, CV_8UC4, in.bytes);

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
    outputs.detection.value.push_back({
      .name = detector->m_classes[classIds[i]]
    , .geometry = { .x = (float)boxes[i].x, .y = (float)boxes[i].y,
                    .w = (float)boxes[i].width, .h = (float)boxes[i].height }
    , .probability = scores[i]
    });

    rectangle(img, boxes[i], Scalar(0, 255, 0), 2);

    char text[100];
    snprintf(text, sizeof(text), "%s: %.2f", detector->m_classes[classIds[i]].c_str(), scores[i]);
    putText(img, text, Point(boxes[i].x, boxes[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
  }

  // Write the modified texture back
  outputs.image.create(in.width, in.height);
  cv::cvtColor(img, img, cv::COLOR_RGB2BGRA);
  outputs.image.texture.width = inputs.image.texture.width;
  outputs.image.texture.height= inputs.image.texture.height;
  outputs.image.texture.bytes = img.data;
  outputs.image.texture.changed = true;
}

}
