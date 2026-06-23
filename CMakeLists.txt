cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# In a score build these two configs are unsupported for this addon and must early-out.
# Standalone they are valid: the portability/smoke CI compiles the dump targets with
# -D_GLIBCXX_DEBUG, and there is a WebAssembly lane -- so only bail when score is present.
if(TARGET score_lib_base OR COMMAND score_common_setup)
  if(CMAKE_CXX_FLAGS MATCHES ".*_GLIBCXX_DEBUG.*")
    return() # Must be disabled when _GLIBCXX_DEBUG
  endif()
  if(EMSCRIPTEN)
    return()
  endif()
endif()

project(score_addon_cv LANGUAGES CXX)

# Max/MSP externals link the static CRT (/MT) on Windows, as the official
# max-sdk-base requires. Set it for the whole addon -- before Avendish and the
# object libraries are created -- so every target shares one runtime; mixing /MT
# and /MD trips MSVC with LNK2038. (CMP0091 is NEW via cmake_minimum_required.)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# OpenCV-free computer-vision objects (CPU algorithms on Eigen + halp textures/buffers),
# built both against ossia score and standalone through the unified avnd_addon_* API.

# Score's external-addon bootstrap only exists in a score build tree; guard it.
if(NOT TARGET score_lib_base)
  include(ScoreExternalAddon OPTIONAL)
endif()

# Use the Avendish the host (ossia score) provides; otherwise fetch it.
# Pin: celtera/avendish (#116) -- carries the standalone "base target" fix so
# target_sources/target_link_libraries(<base> ...) work between init and the objects, and the
# max/jitter matrix_to_buffer fix (raw_data is unsigned char*: assigning a const char* there
# tripped MSVC C2440, breaking every _readback object's max external on Windows).
find_package(Avendish QUIET)
if(NOT Avendish_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Avendish
    GIT_REPOSITORY "https://github.com/celtera/avendish"
    GIT_TAG fa0b8836b46723a15d89e06426f225f2431574db)
  FetchContent_Populate(Avendish)
  list(APPEND CMAKE_PREFIX_PATH "${avendish_SOURCE_DIR}")
  find_package(Avendish REQUIRED)
endif()

# Common dependencies of every CV object: the addon root (objects include via <CV/...>)
# and Eigen. In a score build these come from the score deps tree; standalone we resolve
# Eigen ourselves (header-only). Passed to each object via DEPENDENCIES so they propagate
# to whatever compiles the object -- the score BASE target or the standalone back-ends.
add_library(score_addon_cv_deps INTERFACE)
target_include_directories(score_addon_cv_deps INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# Standalone, an object whose category produces no enabled back-end (e.g. a texture object
# in the SDK-free portability lane where every back-end is OFF) still compiles its bare
# static lib -- which must see halp/avnd headers. Score brings those via score_plugin_avnd,
# so add Avendish's include dir only outside a score build.
if(NOT (TARGET score_plugin_avnd OR COMMAND avnd_score_plugin_add))
  if(AVND_SOURCE_DIR)
    target_include_directories(score_addon_cv_deps INTERFACE "${AVND_SOURCE_DIR}/include")
  endif()
endif()

if(TARGET Eigen3::Eigen)
  target_link_libraries(score_addon_cv_deps INTERFACE Eigen3::Eigen)
else()
  find_package(Eigen3 QUIET)
  if(TARGET Eigen3::Eigen)
    target_link_libraries(score_addon_cv_deps INTERFACE Eigen3::Eigen)
  else()
    include(FetchContent)
    FetchContent_Declare(
      Eigen3cv
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0
      GIT_SHALLOW TRUE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_GetProperties(Eigen3cv)
    if(NOT eigen3cv_POPULATED)
      FetchContent_Populate(Eigen3cv)
    endif()
    target_include_directories(score_addon_cv_deps INTERFACE "${eigen3cv_SOURCE_DIR}")
  endif()
endif()

avnd_addon_init(NAME score_addon_cv)

# Explicit BACKENDS (not the CATEGORY presets) so `dump` -- the SDK-free target the
# portability smoke builds -- is always emitted. Ignored in a score build.
# (gstreamer can now be added too: avendish#116 fixed its xy_pad binding.)
set(CV_TEXTURE_BACKENDS dump ossia max touchdesigner:TOP godot:TEXTURE python)  # texture I/O
set(CV_DATA_BACKENDS    dump ossia max touchdesigner:CHOP_MESSAGE godot:NODE python)  # value/SSBO I/O

function(add_cv_object kind c_name main_class)
  if("${kind}" STREQUAL "texture")
    set(_backends ${CV_TEXTURE_BACKENDS})
  else()
    set(_backends ${CV_DATA_BACKENDS})
  endif()

  set(_sources ${ARGN})

  # Header-only objects (a single .hpp, no translation unit) make avnd_addon_object's
  # standalone base static library empty. An empty archive fails to link on macOS
  # ("ar: no archive members specified") and Windows ("LNK1181: cannot open .lib"),
  # which then cascades into other targets; only GNU ar tolerates it. The main header
  # is still passed as MAIN_FILE to the back-end glue, so generate a one-line .cpp that
  # includes it: this gives the base library a real TU on every platform with no effect
  # on the object's behaviour. (No-op in a score build: score introspects the header.)
  list(GET _sources 0 _main_file)
  set(_has_tu FALSE)
  foreach(_src ${_sources})
    if(_src MATCHES "\\.(cpp|cxx|cc)$")
      set(_has_tu TRUE)
    endif()
  endforeach()
  if(NOT _has_tu)
    set(_stub "${CMAKE_CURRENT_BINARY_DIR}/cv_stubs/${c_name}_stub.cpp")
    file(WRITE "${_stub}"
      "// Auto-generated: gives ${c_name}'s base static library a translation unit so\n"
      "// it is not an empty archive. See add_cv_object() in CMakeLists.txt.\n"
      "#include \"${CMAKE_CURRENT_SOURCE_DIR}/${_main_file}\"\n")
    list(APPEND _sources "${_stub}")
  endif()

  avnd_addon_object(
    BASE score_addon_cv
    C_NAME ${c_name}
    CLASS ${main_class}
    NAMESPACE cv
    BACKENDS ${_backends}
    DEPENDENCIES score_addon_cv_deps
    MAIN_FILE "${_main_file}"
    SOURCES ${_sources})

  # avnd_make_python builds the module as a plain shared library, which inherits the
  # platform-default name (lib-prefixed .dylib / .dll). CPython imports "<module>.so"
  # (Unix) / "<module>.pyd" (Windows) with no "lib" prefix, so force that here
  # (no-op in a score build: no _python target).
  if(TARGET ${c_name}_python)
    set_target_properties(${c_name}_python PROPERTIES PREFIX "")
    if(WIN32)
      set_target_properties(${c_name}_python PROPERTIES SUFFIX ".pyd")
    else()
      set_target_properties(${c_name}_python PROPERTIES SUFFIX ".so")
    endif()
  endif()
endfunction()

# --- Phase 0 anchor: a minimal Path-A object exercising the texture<->Eigen view path ---
add_cv_object(texture cv_luminance Luminance
  CV/Cpu/Luminance.hpp CV/Cpu/Luminance.cpp)

# --- Phase 2: typed SSBO readback objects (pure data: cpu_buffer_input -> value ports) ---
add_cv_object(object cv_centroid_readback CentroidReadback
  CV/Readback/CentroidReadback.hpp)
add_cv_object(object cv_moments_readback MomentsReadback
  CV/Readback/MomentsReadback.hpp)
add_cv_object(object cv_pointlist_readback PointListReadback
  CV/Readback/PointListReadback.hpp)

# --- Phase 3: connected-components labeling (union-find, OpenCV-free) ---
add_cv_object(texture cv_label Label
  CV/Cpu/Label.hpp CV/Cpu/Label.cpp)

# --- Phase 3: contour tracing + per-contour geometry (OpenCV-free) ---
add_cv_object(texture cv_contours Contours
  CV/Cpu/Contours.hpp CV/Cpu/Contours.cpp)

# --- Phase 3: per-blob measurements (centroid/bbox/area/orientation/elongation) ---
add_cv_object(texture cv_blob_stats BlobStats
  CV/Cpu/BlobStats.hpp CV/Cpu/BlobStats.cpp)

# --- Phase 3: scanline flood fill ---
add_cv_object(texture cv_floodfill FloodFill
  CV/Cpu/FloodFill.hpp CV/Cpu/FloodFill.cpp)

# --- Phase 3: temporally-stable blob IDs ---
add_cv_object(texture cv_blob_sort BlobSort
  CV/Cpu/BlobSort.hpp CV/Cpu/BlobSort.cpp)

# --- Phase 4: 2D Kalman point smoother (Eigen) ---
add_cv_object(object cv_kalman2d Kalman
  CV/Cpu/Kalman.hpp)

# --- Phase 4: 4-point homography (Eigen DLT) ---
add_cv_object(object cv_homography Homography
  CV/Cpu/Homography.hpp)

# --- Phase 4: sparse Lucas-Kanade optical flow ---
add_cv_object(texture cv_optical_flow_lk OpticalFlowLK
  CV/Cpu/OpticalFlowLK.hpp CV/Cpu/OpticalFlowLK.cpp)

# --- Phase 5: FAST corner detector ---
add_cv_object(texture cv_fast_corners FastCorners
  CV/Cpu/FastCorners.hpp CV/Cpu/FastCorners.cpp)

# --- Larger objects: features & matching ---
add_cv_object(texture cv_orb_features OrbFeatures
  CV/Cpu/OrbFeatures.hpp CV/Cpu/OrbFeatures.cpp CV/Support/Brief.hpp)
add_cv_object(texture cv_feature_match FeatureMatch
  CV/Cpu/FeatureMatch.hpp CV/Cpu/FeatureMatch.cpp CV/Support/Brief.hpp)

# --- Larger objects: camera geometry ---
add_cv_object(texture cv_chessboard_corners ChessboardCorners
  CV/Cpu/ChessboardCorners.hpp CV/Cpu/ChessboardCorners.cpp)
add_cv_object(texture cv_calibration Calibration
  CV/Cpu/Calibration.hpp CV/Cpu/Calibration.cpp)
add_cv_object(object cv_solve_pnp SolvePnP
  CV/Cpu/SolvePnP.hpp)

# --- Larger objects: classification ---
add_cv_object(texture cv_learn Learn
  CV/Cpu/Learn.hpp CV/Cpu/Learn.cpp)
add_cv_object(object cv_recognize Recognize
  CV/Cpu/Recognize.hpp CV/Cpu/Recognize.cpp)

# --- Larger objects: colour-window tracker ---
add_cv_object(texture cv_camshift CamShift
  CV/Cpu/CamShift.hpp CV/Cpu/CamShift.cpp)

# --- Image-processing / analysis shaders (.fs / .cs) live under CV/Shaders/ and are loaded
#     into the user library at runtime; they are not compiled here.

avnd_addon_finalize(NAME score_addon_cv UUID "b71f97c7-d781-4309-ab07-ece8cd6413bb" VERSION 1)

### Tests ###
# Unit tests for the CPU avendish objects. Each object is a plain struct with operator(),
# so tests construct it, set inputs, call it, and assert on outputs -- no engine harness.
# We compile the real object .cpp files into the test executable so we test shipping code.
# Opt-in via -DSCORE_ADDON_CV_TESTS=ON. Catch2 is used if already provided by the parent
# build, otherwise fetched standalone so the tests are self-contained.
# The tests pull halp/avendish headers in through score_plugin_avnd, so they are only
# wired up in a score build (standalone has its own back-end-level CI via portability).
option(SCORE_ADDON_CV_TESTS "Build score-addon-cv unit tests" OFF)
if(SCORE_ADDON_CV_TESTS AND NOT SCORE_FAST_DEV_BUILD AND TARGET score_plugin_avnd)
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
    score_addon_cv_deps
  )
  if(TARGET score_plugin_avnd)
    target_link_libraries(score_addon_cv_tests PRIVATE score_plugin_avnd)
  endif()
  if(TARGET score_plugin_gfx)
    target_link_libraries(score_addon_cv_tests PRIVATE score_plugin_gfx)
  endif()
  if(TARGET fmt::fmt)
    target_link_libraries(score_addon_cv_tests PRIVATE fmt::fmt)
  endif()
  add_test(NAME score_addon_cv_tests COMMAND score_addon_cv_tests)
endif()
