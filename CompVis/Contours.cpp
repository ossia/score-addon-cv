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

  // qDebug() << "processing: " << in_tex.height << in_tex.width << in_tex.bytes;
  if (!in_tex.bytes || in_tex.width == 0 || in_tex.height == 0)
    return;

  // RGBA pixels to cv::Mat
  cv::Mat img_source(in_tex.height, in_tex.width, CV_8UC4, in_tex.bytes);

  // Convert to the right format
  cv::cvtColor(img_source, img_source, cv::COLOR_BGRA2GRAY);

  contours.clear();
  hierarchy.clear();

  float thresh = this->inputs.threshold;

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

  // Write the modified texture back
  auto& drawing = canny_output;
  outputs.image.create(drawing.cols, drawing.rows);
  //cv::cvtColor(drawing, drawing, cv::COLOR_GRAY2BGRA);

  outputs.image.texture
      = {.bytes = drawing.data,
         .width = drawing.cols,
         .height = drawing.rows,
         .changed = true};
}

}
