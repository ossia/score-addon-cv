cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(CMAKE_CXX_FLAGS MATCHES ".*_GLIBCXX_DEBUG.*")
  return() # Must be disabled when _GLIBCXX_DEBUG
endif()

if(EMSCRIPTEN)
  return()
endif()

if(NOT TARGET score_lib_base)
  include(ScoreExternalAddon)
endif()

if(NOT TARGET score_plugin_avnd)
  return()
endif()

project(score_addon_compvis LANGUAGES CXX)

find_package(OpenCV)
if(NOT OpenCV_FOUND)
    return()
endif()
FetchContent_Declare(
  yolov8
  GIT_REPOSITORY "https://github.com/jcelerier/YOLOv8-TensorRT-CPP"
  GIT_TAG ossia-2024-07-05
  GIT_PROGRESS true
  SYSTEM
)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}/_deps/yolov8-src/cmake")
FetchContent_MakeAvailable(yolov8)

avnd_score_plugin_init(BASE_TARGET score_addon_compvis)

target_include_directories(score_addon_compvis
  PRIVATE
    "${CMAKE_BINARY_DIR}/_deps/yolov8-src/src"
)

target_link_libraries(
  score_addon_compvis
  PRIVATE
    score_plugin_engine
    score_plugin_avnd
    opencv_dnn
    opencv_imgproc
    opencv_core
    opencv_imgcodecs
    opencv_highgui
    fmt
    YoloV8_TRT
  )

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_compvis
  SOURCES
    CompVis/Detector.hpp
    CompVis/Detector.cpp
  TARGET
    yolov4
  MAIN_CLASS
    YoloV4Detector
  NAMESPACE
    CompVis
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_compvis
  SOURCES
    CompVis/YOLOv8-pose.hpp
    CompVis/YOLOv8-pose.cpp
  TARGET
    yolov8-pose
  MAIN_CLASS
    YoloV8PoseDetector
  NAMESPACE
    CompVis
)

if(TARGET opencv_cudaimgproc)
  target_link_libraries(score_addon_compvis PRIVATE opencv_cudaimgproc)
  avnd_score_plugin_add(
    BASE_TARGET
      score_addon_compvis
    SOURCES
      CompVis/Contours.hpp
      CompVis/Contours.cpp
    TARGET
      contours
    MAIN_CLASS
      Contours
    NAMESPACE
      CompVis
  )

  avnd_score_plugin_add(
    BASE_TARGET
      score_addon_compvis
    SOURCES
      CompVis/Contrast.hpp
      CompVis/Contrast.cpp
    TARGET
      contrast
    MAIN_CLASS
      Contrast
    NAMESPACE
      CompVis
  )
endif()

avnd_score_plugin_finalize(BASE_TARGET score_addon_compvis PLUGIN_VERSION 1
                           PLUGIN_UUID "b71f97c7-d781-4309-ab07-ece8cd6413bb")
