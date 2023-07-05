#include "Contrast.hpp"

#undef READ
#undef WRITE
#undef CONSTANT
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

namespace CompVis
{

Contrast::Contrast() noexcept { }

Contrast::~Contrast() { }

void Contrast::operator()()
{
  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;

  // qDebug() << "processing: " << in_tex.height << in_tex.width << in_tex.bytes;
  if (!in_tex.bytes || in_tex.width == 0 || in_tex.height == 0)
    return;

  // RGBA pixels to cv::Mat
  cv::Mat img_source(in_tex.height, in_tex.width, CV_8UC4, in_tex.bytes);

  switch (this->inputs.mode)
  {
    case decltype(this->inputs.mode.value)::Michelson:
      cv::cvtColor(img_source, img_source, cv::COLOR_BGRA2GRAY);
      double min, max;
      cv::minMaxLoc(img_source, &min, &max);
      outputs.contrast = (max - min) / (max + min);
      break;

    case decltype(this->inputs.mode.value)::RMS:
      cv::cvtColor(img_source, img_source, cv::COLOR_BGRA2GRAY);
      cv::Scalar mean, stddev;
      cv::meanStdDev(img_source, mean, stddev);
      outputs.contrast = stddev.val[0];
      break;
  }
}

}
