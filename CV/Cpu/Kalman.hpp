#pragma once

#include <halp/controls.hpp>
#include <halp/messages.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace cv
{
// Which of the two estimates produced by one filter step is sent out.
//  * Corrected  : the a-posteriori estimate, i.e. after the measurement update.
//  * Prediction : the a-priori estimate, i.e. what the model predicted *before* seeing the
//                 measurement. This is what cv.jit.kalman outputs.
enum class KalmanOutput
{
  Prediction,
  Corrected
};

// cv.jit.kalman -- an N-dimensional Kalman filter over a list of floats.
//
// This is a plain Max box (class_new + ext_main in cv.jit.kalman.cpp, no Jitter MOP and no
// max.cv.jit.* companion file), so the whole interface lives in that one file:
//   * one list inlet (the measurement) and one list outlet (the predicted state),
//   * a `control` list message,
//   * `clear` / `reset` messages (same handler),
//   * two attributes: `velocity_weight` and `acceleration_weight`, both defaulting to 0.001.
//
// ALGORITHM (ported exactly unless listed as a deviation below)
//
// The incoming list length *defines* the filter: `measurement_dims` M = list length, and the
// state dimension is D = 3*M -- for every measured dimension the model tracks a position, a
// velocity and an acceleration, laid out in three contiguous blocks:
//
//     state = [ p_0..p_{M-1} | v_0..v_{M-1} | a_0..a_{M-1} ]
//
// A list whose length differs from the current one fully re-initialises the filter.
//
// The transition matrix is rebuilt from scratch on every list from
// `weights = { 1, velocity_weight, acceleration_weight }`:
//
//     for (i = 0; i < D; i++)
//       for (j = i, k = 0; j < D && k < 3; j += M, k++)
//         A(i, j) = weights[k];
//
// which is *not* the textbook constant-acceleration matrix: it is the asymmetric block form
//
//     A = [ I   wv*I  wa*I ]
//         [ 0    I    wv*I ]
//         [ 0    0     I   ]
//
// (the acceleration block feeds velocity and position with the *same* weights, and there is
// no 1/2 dt^2 term). Consequences, per the cv.jit help file ("amount of noise", range [0,20]):
// velocity_weight = acceleration_weight = 0 makes A the identity, i.e. a constant-signal
// model; acceleration_weight = 0 alone makes it a constant-velocity model.
//
// Noise: cv.jit hardcodes processNoiseCov = 1e-5*I, measurementNoiseCov = 1e-1*I and
// errorCovPost = 1*I in its `clear` handler. This port keeps the pre-existing "Process noise"
// and "Measurement noise" sliders but uses those two values as their defaults; errorCovPost
// is fixed at 1*I as in cv.jit.
//
// measurementMatrix is cv::setIdentity() on the M x D matrix, i.e. H = [I 0 0]: only the
// position block is observed.
//
// Per list: `prediction = predict(control)` then `correct(measurement)`, and the *prediction*
// (the a-priori estimate captured before the correction) is what is sent out. The "Output"
// enum selects between that and the corrected (a-posteriori) estimate; it defaults to
// `Prediction`, as in cv.jit, and applies to the fixed-2D fallback path as well.
//
// DELIBERATE DEVIATIONS
//
//  * CONTROL INPUT. cv.jit's control input has no effect whatsoever: `cv_jit_kalman_control`
//    allocates `cv::Mat(1, control_dims, CV_32F)` -- a *row* vector, where cv::KalmanFilter
//    wants control_dims x 1 -- and `controlMatrix` is never initialised (it is all zeros after
//    `init`), so `statePre += controlMatrix * control` adds nothing. This port keeps a working
//    control injection instead: the control vector is added straight onto the position block
//    (B = [I 0 0]^T). A control list whose length differs from the measurement length is
//    ignored rather than triggering a full re-init (cv.jit re-inits, but since its control is
//    inert that re-init is pure state loss).
//  * SEEDING. cv.jit sets `statePre[i] = measurement[i]` on the first list after init, but
//    `cv::KalmanFilter::predict()` immediately overwrites statePre with A*statePost (statePost
//    being all zeros), so the seed is dropped and the filter actually starts from the origin.
//    Here the seed is written to *both* statePre and statePost, so it survives the predict and
//    the first output is exactly the first measurement -- which is plainly what was intended,
//    and matches the pre-existing 2D behaviour of this object.
//  * The pre-existing fixed-2D constant-velocity path (state [x, y, vx, vy], "Measurement" /
//    "Control" xy ports, "Estimate" / "Velocity" xy outputs) is kept as a fallback and runs
//    whenever the measurement list input is empty. It is unrelated to cv.jit's algorithm.
struct Kalman
{
  halp_meta(name, "Kalman");
  halp_meta(c_name, "cv_kalman2d");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(
      description,
      "N-dimensional Kalman filter (cv.jit.kalman): the measurement list length sets the "
      "number of dimensions, each tracked with position/velocity/acceleration.");
  halp_meta(uuid, "5e9c2a14-8b07-4d63-9f1a-2c6b0e8d3a59");

  struct
  {
    // --- fixed-2D fallback ports (used when the measurement list is empty) ---
    halp::xy_spinboxes_f32<"Measurement"> measurement;
    // Optional control input. Applied as state += B*control in the predict step;
    // (0,0) leaves behaviour unchanged.
    halp::xy_spinboxes_f32<"Control", halp::range{-1000.f, 1000.f, 0.f}> control;

    // Noise: defaults are cv.jit's hardcoded values (1e-5 and 1e-1).
    halp::hslider_f32<"Process noise", halp::range{1e-7f, 1.f, 1e-5f}> q;
    halp::hslider_f32<"Measurement noise", halp::range{1e-7f, 1.f, 1e-1f}> r;
    halp::toggle<"Reset"> reset;

    // --- cv.jit.kalman proper ---
    // The list length defines the number of measured dimensions; a length change fully
    // re-initialises the filter.
    struct
    {
      halp_meta(name, "Measurement list") std::vector<float> value;
    } measurement_list;

    // N-dimensional control vector; used when it has exactly as many elements as the
    // measurement list, ignored otherwise.
    struct
    {
      halp_meta(name, "Control list") std::vector<float> value;
    } control_list;

    // cv.jit attributes, both default 0.001. Help text: "amount of noise", range [0, 20].
    halp::hslider_f32<"Velocity weight", halp::range{0.f, 20.f, 0.001f}> vel_weight;
    halp::hslider_f32<"Acceleration weight", halp::range{0.f, 20.f, 0.001f}> accel_weight;

    halp::enum_t<KalmanOutput, "Output"> output;
  } inputs;

  struct
  {
    halp::val_port<"Estimate", halp::xy_type<float>> estimate;
    halp::val_port<"Velocity", halp::xy_type<float>> velocity;

    // The M position components of the selected estimate (cv.jit's list outlet).
    struct
    {
      halp_meta(name, "State") std::vector<float> value;
    } state;
  } outputs;

  Kalman() { reset(); }

  // cv.jit.kalman's `clear` and `reset` messages both run the same handler.
  void reset()
  {
    // 2D fallback state
    m_x.setZero();
    m_P = Eigen::Matrix4f::Identity();
    m_initialized = false;

    // N-D state: forget the dimensionality entirely, the next list re-derives it.
    m_M = 0;
    m_D = 0;
    m_seed_pending = true;
    m_statePre.resize(0);
    m_statePost.resize(0);
    m_errorCovPre.resize(0, 0);
    m_errorCovPost.resize(0, 0);
    m_A.resize(0, 0);
    m_H.resize(0, 0);
  }
  void clear() { reset(); }

  halp_start_messages(Kalman)
    halp_mem_fun(reset)
    halp_mem_fun(clear)
  halp_end_messages

  void operator()() noexcept
  {
    // Rising edge on the Reset toggle, evaluated before anything else so it can never be
    // swallowed by a tick that carries no new measurement.
    if(inputs.reset.value && !m_was_reset)
    {
      reset();
      m_was_reset = true;
    }
    else if(!inputs.reset.value)
    {
      m_was_reset = false;
    }

    if(!inputs.measurement_list.value.empty())
      run_nd();
    else
      run_2d();
  }

  // --- introspection, for tests -------------------------------------------------------
  int measurement_dims() const noexcept { return m_M; }
  int state_dims() const noexcept { return m_D; }
  const Eigen::MatrixXf& transition_matrix() const noexcept { return m_A; }
  const Eigen::MatrixXf& measurement_matrix() const noexcept { return m_H; }
  const Eigen::VectorXf& predicted_state() const noexcept { return m_statePre; }
  const Eigen::VectorXf& corrected_state() const noexcept { return m_statePost; }

private:
  // ------------------------------------------------------------------ N-dimensional path
  void init_nd(int M)
  {
    m_M = M;
    m_D = 3 * M; // position, velocity and acceleration for every measured dimension

    m_statePre = Eigen::VectorXf::Zero(m_D);
    m_statePost = Eigen::VectorXf::Zero(m_D);
    m_errorCovPre = Eigen::MatrixXf::Zero(m_D, m_D);
    // cv.jit: setIdentity(errorCovPost, Scalar::all(1))
    m_errorCovPost = Eigen::MatrixXf::Identity(m_D, m_D);
    m_A = Eigen::MatrixXf::Zero(m_D, m_D);

    // cv.jit: setIdentity(measurementMatrix) on the M x D matrix -> observe positions only.
    m_H = Eigen::MatrixXf::Zero(m_M, m_D);
    for(int i = 0; i < m_M; i++)
      m_H(i, i) = 1.f;

    m_seed_pending = true;
  }

  void build_transition() noexcept
  {
    const float weights[3]
        = {1.f, inputs.vel_weight.value, inputs.accel_weight.value};

    m_A.setZero();
    for(int i = 0; i < m_D; i++)
      for(int j = i, k = 0; j < m_D && k < 3; j += m_M, k++)
        m_A(i, j) = weights[k];
  }

  void run_nd() noexcept
  {
    const auto& in = inputs.measurement_list.value;
    const int M = static_cast<int>(in.size());

    // A change of list length re-initialises everything (cv.jit_kalman_list -> _clear).
    if(M != m_M)
      init_nd(M);

    Eigen::VectorXf z(M);
    for(int i = 0; i < M; i++)
      z(i) = in[i];

    // First list after an init seeds the position block from the measurement.
    if(m_seed_pending)
    {
      m_statePre.head(M) = z;
      m_statePost.head(M) = z;
      m_seed_pending = false;
    }

    build_transition();

    // --- predict ---
    m_statePre = m_A * m_statePost;

    // Control: B = [I 0 0]^T, i.e. it acts directly on the position block.
    const auto& u = inputs.control_list.value;
    if(static_cast<int>(u.size()) == M)
      for(int i = 0; i < M; i++)
        m_statePre(i) += u[i];

    m_errorCovPre = m_A * m_errorCovPost * m_A.transpose()
                    + Eigen::MatrixXf::Identity(m_D, m_D) * inputs.q.value;

    const Eigen::VectorXf prediction = m_statePre;

    // --- correct ---
    const Eigen::MatrixXf S = m_H * m_errorCovPre * m_H.transpose()
                              + Eigen::MatrixXf::Identity(M, M) * inputs.r.value;
    const Eigen::MatrixXf K = m_errorCovPre * m_H.transpose() * S.inverse();
    m_statePost = m_statePre + K * (z - m_H * m_statePre);
    m_errorCovPost
        = (Eigen::MatrixXf::Identity(m_D, m_D) - K * m_H) * m_errorCovPre;

    const Eigen::VectorXf& out
        = (inputs.output.value == KalmanOutput::Prediction) ? prediction : m_statePost;

    auto& dst = outputs.state.value;
    dst.resize(M);
    for(int i = 0; i < M; i++)
      dst[i] = out(i);

    // Mirror the first (up to) two dimensions on the xy outputs, for convenience.
    outputs.estimate.value = {out(0), M > 1 ? out(1) : 0.f};
    outputs.velocity.value = {out(M), M > 1 ? out(M + 1) : 0.f};
  }

  // ---------------------------------------------------- fixed-2D constant-velocity path
  void run_2d() noexcept
  {
    const float dt = 1.0f; // per-tick; control rate is handled by the host scheduling.

    const Eigen::Vector2f z(inputs.measurement.value.x, inputs.measurement.value.y);

    // On the first update after a reset, seed the position state directly from the current
    // measurement (velocity 0) so there is no slow pull from the origin.
    if(!m_initialized)
    {
      m_x << z(0), z(1), 0.f, 0.f;
      m_P = Eigen::Matrix4f::Identity();
      m_initialized = true;
      publish_2d(m_x);
      return;
    }

    // State transition (constant velocity).
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;

    // Control matrix B: control acts directly on the position state.
    Eigen::Matrix<float, 4, 2> B = Eigen::Matrix<float, 4, 2>::Zero();
    B(0, 0) = 1.f;
    B(1, 1) = 1.f;
    const Eigen::Vector2f u(inputs.control.value.x, inputs.control.value.y);

    // Process + measurement noise from the controls.
    const Eigen::Matrix4f Q = Eigen::Matrix4f::Identity() * inputs.q.value;
    const Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * inputs.r.value;

    // Measurement matrix (observe position only).
    Eigen::Matrix<float, 2, 4> H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.f;
    H(1, 1) = 1.f;

    // Predict (incorporating the optional control input).
    m_x = F * m_x + B * u;
    m_P = F * m_P * F.transpose() + Q;
    const Eigen::Vector4f prediction = m_x;

    // Update.
    const Eigen::Vector2f y = z - H * m_x;
    const Eigen::Matrix2f S = H * m_P * H.transpose() + R;
    const Eigen::Matrix<float, 4, 2> K = m_P * H.transpose() * S.inverse();
    m_x = m_x + K * y;
    m_P = (Eigen::Matrix4f::Identity() - K * H) * m_P;

    publish_2d(inputs.output.value == KalmanOutput::Prediction ? prediction : m_x);
  }

  void publish_2d(const Eigen::Vector4f& s) noexcept
  {
    outputs.estimate.value = {s(0), s(1)};
    outputs.velocity.value = {s(2), s(3)};
    outputs.state.value.assign({s(0), s(1)});
  }

  // 2D fallback
  Eigen::Vector4f m_x;  // state
  Eigen::Matrix4f m_P;  // covariance
  bool m_was_reset = false;
  bool m_initialized = false;

  // N-D (cv.jit.kalman)
  int m_M = 0; // measurement_dims
  int m_D = 0; // dynamic_dims = 3 * M
  bool m_seed_pending = true;
  Eigen::VectorXf m_statePre, m_statePost;
  Eigen::MatrixXf m_errorCovPre, m_errorCovPost;
  Eigen::MatrixXf m_A; // transitionMatrix, D x D
  Eigen::MatrixXf m_H; // measurementMatrix, M x D
};
}
