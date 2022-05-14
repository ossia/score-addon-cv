cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT TARGET score_lib_base)
  include("${SCORE_SOURCE_DIR}/cmake/ScoreExternalAddon.cmake")
endif()

project(score_addon_compvis LANGUAGES CXX)

find_package(OpenCV REQUIRED)
add_library(
  score_addon_compvis
  CompVis/Geometry.hpp
  CompVis/Detector.hpp
  CompVis/Detector.cpp
  score_addon_compvis.hpp
  score_addon_compvis.cpp
)

setup_score_plugin(score_addon_compvis)

target_link_libraries(score_addon_compvis
  PRIVATE
    score_plugin_engine score_plugin_avnd
    opencv_dnn opencv_imgproc opencv_core opencv_imgcodecs opencv_highgui
)
