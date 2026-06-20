# score-addon-cv

Classic (non-DNN) computer-vision processes for ossia score — **OpenCV-free**.

Image filters and analyses run on the GPU as ISF/CSF shaders; sequential / stateful /
geometry algorithms run as avendish C++ objects built only on libraries already vendored by
score (**Eigen, xtensor/xsimd, Boost.Geometry**). No new third-party dependencies. DNN models
(YOLO, pose, depth, segmentation) live in `score-addon-onnx`, not here.

See [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for architecture and
[`PORTING_CANDIDATES.md`](./PORTING_CANDIDATES.md) for the full algorithm catalog.

## Three implementation paths

- **Path I — ISF fragment shader** (`CV/Shaders/Filters/*.fs`): pure image→image filters.
- **Path B — CSF compute shader** (`CV/Shaders/Analysis/*.cs`): GPU analyses that write a
  result into an SSBO, read back to the host as a value with `uo::BufferToArray` (generic) or
  a typed readback object in `CV/Readback/`.
- **Path A — avendish C++ object** (`CV/Cpu/*`): sequential / stateful / geometry algorithms.

## Objects

### Image filters (Path I shaders — `CV/Shaders/Filters/`)
| Shader | Does |
|---|---|
| `Threshold.fs` | global + adaptive (mean−C) binarization |
| `Morphology.fs` | erode / dilate / open / close, square or cross SE |
| `Binedge.fs` | binary blob boundary pixels (cv.jit.binedge) |
| `ChromaKey.fs` | HSV colour-range key → mask or alpha cut |
| `Resize.fs` | centre-anchored zoom/resample, nearest or bilinear (cv.jit.resize) |
| `Perspective.fs` | warp by a 3×3 homography, inverse-mapped bilinear (cv.jit.perspective) |
| `FrameDiff.fs` | motion vs. running-average background (feedback; RGB=model, A=motion) |
| `RunningAverage.fs` | leaky/IIR running average (feedback; cv.jit.ravg) |
| `TemporalMean.fs` | long-window temporal mean (feedback; cv.jit.mean) |
| `DenseFlow.fs` | dense optical-flow field (Horn-Schunck-style; flow in RG) |

Reuse the shaders already shipped in score's `shaderlib/image-processing/`: `EdgeDetect.fs`
(Sobel/Prewitt/Laplacian/Roberts), `GaussianBlur.fs`, `Displacement.fs`, `NoiseGenerator.fs`.

### Analyses (Path B compute → SSBO — `CV/Shaders/Analysis/` + `CV/Readback/`)
| Shader | Result | Readback |
|---|---|---|
| `Sum.cs` | total luminance | `BufferToArray` |
| `Centroid.cs` | centroid + mass | `CentroidReadback` → `(x,y)` + mass |
| `Histogram.cs` | 256-bin luma histogram | `BufferToArray` (UInt32) |
| `MinMaxMean.cs` | luma min / max / mean | `BufferToArray` |
| `Covariance.cs` | covariance of two images' luma | `BufferToArray`, host: Eab−Ea·Eb |
| `Moments.cs` | raw moments ≤3rd order | `MomentsReadback` → centroid, orientation, eccentricity, 7 Hu |
| `Corners.cs` | Harris corners (response+NMS+append) | `PointListReadback` → point list |
| `Hough.cs` | θ×ρ line accumulator | `BufferToArray`, peak-pick on host |

### CPU objects (Path A — `CV/Cpu/`)
| Object | Does |
|---|---|
| `Luminance` | RGBA8 → r8 luma |
| `Label` | connected components (union-find, 8-conn) + count + viz |
| `Contours` | Moore-neighbor border tracing + per-contour geometry list |
| `BlobStats` | per-blob centroid / bbox / area / orientation / direction / elongation |
| `FloodFill` | scanline flood fill from a seed by luminance similarity |
| `BlobSort` | temporally-stable blob IDs (nearest-neighbour) |
| `OpticalFlowLK` | sparse Lucas-Kanade flow on a grid |
| `Homography` | 4-point perspective transform (Eigen SVD DLT) |
| `Kalman` | 2D constant-velocity point smoother/predictor |
| `FastCorners` | FAST-9 corner detector + NMS |
| `OrbFeatures` | oriented-FAST + rotated-BRIEF keypoints (cv.jit.keypoints) |
| `FeatureMatch` | temporal ORB matching, Lowe ratio test (cv.jit.keypoints.match) |
| `ChessboardCorners` | chessboard inner-corner detection (cv.jit.findchessboardcorners) |
| `Calibration` | camera intrinsics + distortion, Zhang's method (cv.jit.calibration) |
| `SolvePnP` | object pose R,t from 3D↔2D points (cv.jit.unproject) |
| `Learn` | train mean+covariance model (cv.jit.learn) |
| `Recognize` | Mahalanobis-distance classifier (cv.jit.blobs.recon) |
| `CamShift` | hue-histogram colour-window tracker (cv.jit.shift) |

Plus `Undistort.fs` (Brown-Conrady lens undistortion shader).

## Tests

Catch2 unit tests for every CPU object live in `tests/`. They construct each object, set its
`inputs`, call `operator()`, and assert on `outputs` — no engine harness; the real object
`.cpp` files are compiled into the test binary. Build with `-DSCORE_ADDON_CV_TESTS=ON`
(Catch2 is reused from the parent build if present, otherwise fetched), then run
`score_addon_cv_tests` (or `ctest -R score_addon_cv_tests`).

Coverage: 45 cases / 280 assertions across Luminance, Label, Contours, BlobStats, BlobSort,
FloodFill, FastCorners, OrbFeatures, FeatureMatch, OpticalFlowLK, CamShift, ChessboardCorners,
Calibration, Kalman, Homography, SolvePnP, Learn, Recognize, and the three readback decoders.
The suite runs clean under ASAN/UBSAN — and during bring-up it caught a real heap-buffer
overflow in the rotated-BRIEF descriptor (`Brief.hpp`) at image borders, now fixed.

## Status

All objects are written and their core algorithms unit-tested in isolation (connected
components, Moore tracing, flood fill, Hu-moment invariance, Kalman, homography DLT, LK flow,
FAST). The shaders are validated against the CSF/ISF parser + renderer dispatch model.
A full on-device compile-link/run is the remaining validation step.

Deferred follow-ups (non-blocking): pyramidal LK, solvePnP/calibration, ORB descriptors +
matching, convex hull / Douglas-Peucker via Boost.Geometry. Fiducial markers (AprilTag) are
deferred to Phase 7 (the only feature that would add a dependency).
