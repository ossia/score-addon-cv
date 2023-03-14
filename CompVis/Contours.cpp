#include "Contours.hpp"

#undef READ
#undef WRITE
#undef CONSTANT
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

namespace CompVis
{

Contours::Contours() noexcept
{
  outputs.image.create(1, 1);
}

Contours::~Contours() { }

// NOTE: we could use a worker thread for this with our triple-buffer implementation maybe?
// BUT: it would be better to import the texture in a cv::gl::Texture2D
// but will only work for GL...
// load the texture, copyTo GpuMat, do the processing
void Contours::operator()()
{
  auto& in_tex = inputs.image.texture;
  if (!in_tex.changed)
    return;
  //auto canny = cv::cuda::createCannyEdgeDetector(100, 150);

  // RGBA pixels to cv::Mat
  cv::Mat img_source(in_tex.height, in_tex.width, CV_8UC4, in_tex.bytes);

  // Convert to the right format
  //cv::cvtColor(img_source, img_source, cv::COLOR_BGRA2RGB);

  cv::Mat canny_output;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  float thresh = this->inputs.threshold;
  //canny->detect(img_source, canny_output);
  cv::Canny(img_source, canny_output, thresh, thresh * 2, 3);
  cv::findContours(
      canny_output,
      contours,
      hierarchy,
      cv::RETR_EXTERNAL,
      cv::CHAIN_APPROX_NONE,
      cv::Point(0, 0));

  outputs.contours = contours.size();
  outputs.density = 0;
  for (auto& c : contours)
    outputs.density += c.size();
  /*
  cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
  for (size_t i = 0; i < contours.size(); i++)
  {
    cv::Scalar color = cv::Scalar(255, 0, 0);
    drawContours(
        drawing, contours, (int)i, color, 2, 8, hierarchy, 0, cv::Point());
  }
*/
  // Write the modified texture back
  auto& drawing = canny_output;
  outputs.image.create(in_tex.width, in_tex.height);
  cv::cvtColor(drawing, drawing, cv::COLOR_RGB2BGRA);

  outputs.image.texture
      = {.bytes = drawing.data,
         .width = in_tex.width,
         .height = in_tex.height,
         .changed = true};
}

}
