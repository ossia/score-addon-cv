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

project(score_addon_cv LANGUAGES CXX)

# Computer-vision addon — OpenCV-free.
# Image filters are done as ISF/CSF shaders (referenced from the shipped shaderlib);
# analyses producing data run as CSF compute shaders writing an SSBO, read back via
# uo::BufferToArray. CPU-side (sequential / library) algorithms are avendish objects built
# only on already-vendored libraries: Eigen, xtensor/xsimd, Boost.Geometry.
# No new third-party dependencies.

avnd_score_plugin_init(BASE_TARGET score_addon_cv)

target_link_libraries(
  score_addon_cv
  PRIVATE
    score_plugin_engine
    score_plugin_avnd
    score_plugin_gfx
    fmt::fmt
    Eigen3::Eigen
    xtensor
    xsimd
)

target_include_directories(score_addon_cv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# --- Phase 0 anchor: a minimal Path-A object exercising the texture<->Eigen view path ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/Luminance.hpp
    CV/Cpu/Luminance.cpp
  TARGET
    cv_luminance
  MAIN_CLASS
    Luminance
  NAMESPACE
    cv
)

# --- Phase 2: typed SSBO readback objects ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Readback/CentroidReadback.hpp
  TARGET
    cv_centroid_readback
  MAIN_CLASS
    CentroidReadback
  NAMESPACE
    cv
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Readback/MomentsReadback.hpp
  TARGET
    cv_moments_readback
  MAIN_CLASS
    MomentsReadback
  NAMESPACE
    cv
)

avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Readback/PointListReadback.hpp
  TARGET
    cv_pointlist_readback
  MAIN_CLASS
    PointListReadback
  NAMESPACE
    cv
)

# --- Phase 3: connected-components labeling (union-find, OpenCV-free) ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/Label.hpp
    CV/Cpu/Label.cpp
  TARGET
    cv_label
  MAIN_CLASS
    Label
  NAMESPACE
    cv
)

# --- Phase 3: contour tracing + per-contour geometry (OpenCV-free) ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/Contours.hpp
    CV/Cpu/Contours.cpp
  TARGET
    cv_contours
  MAIN_CLASS
    Contours
  NAMESPACE
    cv
)

# --- Phase 3: per-blob measurements (centroid/bbox/area/orientation/elongation) ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/BlobStats.hpp
    CV/Cpu/BlobStats.cpp
  TARGET
    cv_blob_stats
  MAIN_CLASS
    BlobStats
  NAMESPACE
    cv
)

# --- Phase 3: scanline flood fill ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/FloodFill.hpp
    CV/Cpu/FloodFill.cpp
  TARGET
    cv_floodfill
  MAIN_CLASS
    FloodFill
  NAMESPACE
    cv
)

# --- Phase 3: temporally-stable blob IDs ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/BlobSort.hpp
    CV/Cpu/BlobSort.cpp
  TARGET
    cv_blob_sort
  MAIN_CLASS
    BlobSort
  NAMESPACE
    cv
)

# --- Phase 4: 2D Kalman point smoother (Eigen) ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/Kalman.hpp
  TARGET
    cv_kalman2d
  MAIN_CLASS
    Kalman
  NAMESPACE
    cv
)

# --- Phase 4: 4-point homography (Eigen DLT) ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/Homography.hpp
  TARGET
    cv_homography
  MAIN_CLASS
    Homography
  NAMESPACE
    cv
)

# --- Phase 4: sparse Lucas-Kanade optical flow ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/OpticalFlowLK.hpp
    CV/Cpu/OpticalFlowLK.cpp
  TARGET
    cv_optical_flow_lk
  MAIN_CLASS
    OpticalFlowLK
  NAMESPACE
    cv
)

# --- Phase 5: FAST corner detector ---
avnd_score_plugin_add(
  BASE_TARGET
    score_addon_cv
  SOURCES
    CV/Cpu/FastCorners.hpp
    CV/Cpu/FastCorners.cpp
  TARGET
    cv_fast_corners
  MAIN_CLASS
    FastCorners
  NAMESPACE
    cv
)

# --- Larger objects: features & matching ---
avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/OrbFeatures.hpp CV/Cpu/OrbFeatures.cpp CV/Support/Brief.hpp
  TARGET cv_orb_features MAIN_CLASS OrbFeatures NAMESPACE cv)

avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/FeatureMatch.hpp CV/Cpu/FeatureMatch.cpp CV/Support/Brief.hpp
  TARGET cv_feature_match MAIN_CLASS FeatureMatch NAMESPACE cv)

# --- Larger objects: camera geometry ---
avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/ChessboardCorners.hpp CV/Cpu/ChessboardCorners.cpp
  TARGET cv_chessboard_corners MAIN_CLASS ChessboardCorners NAMESPACE cv)

avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/Calibration.hpp CV/Cpu/Calibration.cpp
  TARGET cv_calibration MAIN_CLASS Calibration NAMESPACE cv)

avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/SolvePnP.hpp
  TARGET cv_solve_pnp MAIN_CLASS SolvePnP NAMESPACE cv)

# --- Larger objects: classification ---
avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/Learn.hpp CV/Cpu/Learn.cpp
  TARGET cv_learn MAIN_CLASS Learn NAMESPACE cv)

avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/Recognize.hpp CV/Cpu/Recognize.cpp
  TARGET cv_recognize MAIN_CLASS Recognize NAMESPACE cv)

# --- Larger objects: colour-window tracker ---
avnd_score_plugin_add(
  BASE_TARGET score_addon_cv
  SOURCES CV/Cpu/CamShift.hpp CV/Cpu/CamShift.cpp
  TARGET cv_camshift MAIN_CLASS CamShift NAMESPACE cv)

# --- Image-processing / analysis shaders (.fs / .cs) live under CV/Shaders/ and are loaded
#     into the user library at runtime; they are not compiled here.

avnd_score_plugin_finalize(BASE_TARGET score_addon_cv PLUGIN_VERSION 1
                           PLUGIN_UUID "b71f97c7-d781-4309-ab07-ece8cd6413bb")

### Tests ###
# Unit tests for the CPU avendish objects. Each object is a plain struct with operator(),
# so tests construct it, set inputs, call it, and assert on outputs — no engine harness.
# We compile the real object .cpp files into the test executable so we test shipping code.
# Opt-in via -DSCORE_ADDON_CV_TESTS=ON. Catch2 is used if already provided by the parent
# build, otherwise fetched standalone so the tests are self-contained.
option(SCORE_ADDON_CV_TESTS "Build score-addon-cv unit tests" OFF)
if(SCORE_ADDON_CV_TESTS AND NOT SCORE_FAST_DEV_BUILD)
  if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(
      Catch2cv
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.4
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2cv)
  endif()
  add_executable(score_addon_cv_tests
    tests/test_main_objects.cpp
    tests/test_blobs.cpp
    tests/test_features.cpp
    tests/test_geometry.cpp
    tests/test_readbacks.cpp
    tests/test_tracking.cpp
    # real object sources under test:
    CV/Cpu/Luminance.cpp
    CV/Cpu/Label.cpp
    CV/Cpu/Contours.cpp
    CV/Cpu/BlobStats.cpp
    CV/Cpu/BlobSort.cpp
    CV/Cpu/FloodFill.cpp
    CV/Cpu/FastCorners.cpp
    CV/Cpu/OrbFeatures.cpp
    CV/Cpu/FeatureMatch.cpp
    CV/Cpu/OpticalFlowLK.cpp
    CV/Cpu/CamShift.cpp
    CV/Cpu/ChessboardCorners.cpp
    CV/Cpu/Calibration.cpp
    CV/Cpu/Learn.cpp
    CV/Cpu/Recognize.cpp
  )
  target_include_directories(score_addon_cv_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(score_addon_cv_tests PRIVATE
    Catch2::Catch2WithMain
    score_plugin_avnd
    score_plugin_gfx
    fmt::fmt
    Eigen3::Eigen
    xtensor
    xsimd
  )
  add_test(NAME score_addon_cv_tests COMMAND score_addon_cv_tests)
endif()
