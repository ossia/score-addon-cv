#include "YOLOv8-pose.hpp"

#include "yolov8.h"
#undef READ
#undef WRITE
#include <opencv2/opencv.hpp>

namespace CompVis
{
struct YOLOv8Pose
{
  YoloV8Config config;
  std::unique_ptr<YoloV8> yoloV8;

  void reload(const std::string& model)
  {
    std::string onnxModelPath = "/home/jcelerier/Downloads/yolov8x-pose.onnx";

    yoloV8 = std::make_unique<YoloV8>(onnxModelPath, config);
  }
};

YoloV8PoseDetector::YoloV8PoseDetector() noexcept { }

YoloV8PoseDetector::~YoloV8PoseDetector() { }

void YoloV8PoseDetector::operator()()
{
  auto& in_tex = inputs.image.texture;
  if(!in_tex.changed)
    return;

  if(!this->detector)
  {
    this->detector = std::make_unique<YOLOv8Pose>();
    this->detector->reload("");
  }

  // RGBA pixels to cv::Mat
  cv::Mat img(in_tex.height, in_tex.width, CV_8UC4, in_tex.bytes);

  // Convert to the right format
  cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
  cv::flip(img, img, 0);

  // Perform detection
  const auto objects = detector->yoloV8->detectObjects(img);

  // Draw the bounding boxes on the image
  detector->yoloV8->drawObjectLabels(img, objects);

  // Save detected objects
  outputs.detection.value.clear();
  for(const auto& object : objects)
  {
    // Store them in the output port
    auto& classname = detector->config.classNames[object.label];
    auto& box = object.rect;

    std::vector<halp::xy_type<float>> keypoints;
    if(!object.kps.empty())
    {
      auto NUM_KPS = detector->yoloV8->NUM_KPS;
      auto KPS_THRESHOLD = detector->yoloV8->KPS_THRESHOLD;
      auto& SKELETON = detector->yoloV8->SKELETON;
      auto& kps = object.kps;
      for(int k = 0; k < NUM_KPS + 2; k++)
      {
        if(k < NUM_KPS)
        {
          int kpsX = std::round(kps[k * 3]);
          int kpsY = std::round(kps[k * 3 + 1]);
          float kpsS = kps[k * 3 + 2];
          if(kpsS > KPS_THRESHOLD)
          {
            keypoints.emplace_back(kpsX, kpsY);
          }
        }
      }
    }

    outputs.detection.value.push_back(
        {.name = classname,
         .geometry
         = {.x = (float)box.x,
            .y = (float)box.y,
            .w = (float)box.width,
            .h = (float)box.height},
         .probability = object.probability,
         .keypoints = std::move(keypoints)});
  }

  // Write the modified texture back
  //   outputs.image.create(in_tex.width, in_tex.height);
  //   cv::cvtColor(img, img, cv::COLOR_RGB2BGRA);
  //   cv::flip(img, img, 0);
  //
  outputs.image.texture
      = {.bytes = in_tex.bytes,
         .width = in_tex.width,
         .height = in_tex.height,
         .changed = true};
}

} // namespace CompVis
