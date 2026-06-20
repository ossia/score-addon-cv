#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <Eigen/Dense>

namespace cv
{
// 2D constant-velocity Kalman filter (cv.jit.kalman). Smooths / predicts a noisy 2D point
// such as a blob centroid or tracked feature. State = [x, y, vx, vy]. Pure Eigen.
struct Kalman
{
  halp_meta(name, "Kalman 2D");
  halp_meta(c_name, "cv_kalman2d");
  halp_meta(category, "Visuals/Computer Vision");
  halp_meta(author, "ossia score");
  halp_meta(description, "Smooth/predict a noisy 2D point (constant-velocity Kalman).");
  halp_meta(uuid, "5e9c2a14-8b07-4d63-9f1a-2c6b0e8d3a59");

  struct
  {
    halp::xy_spinboxes_f32<"Measurement"> measurement;
    // Optional control input (cv.jit.kalman had a control vector). Applied as
    // state += B*control in the predict step; (0,0) leaves behaviour unchanged.
    // Init defaults to 0 so existing behaviour is unchanged unless driven.
    halp::xy_spinboxes_f32<"Control", halp::range{-1000.f, 1000.f, 0.f}> control;
    halp::hslider_f32<"Process noise", halp::range{0.0001f, 1.f, 0.01f}> q;
    halp::hslider_f32<"Measurement noise", halp::range{0.0001f, 1.f, 0.1f}> r;
    halp::toggle<"Reset"> reset;
  } inputs;

  struct
  {
    halp::val_port<"Estimate", halp::xy_type<float>> estimate;
    halp::val_port<"Velocity", halp::xy_type<float>> velocity;
  } outputs;

  Kalman() { init(); }

  void operator()() noexcept
  {
    if(inputs.reset.value && !m_was_reset)
    {
      init();
      m_was_reset = true;
    }
    else if(!inputs.reset.value)
    {
      m_was_reset = false;
    }

    const float dt = 1.0f; // per-tick; control rate is handled by the host scheduling.

    Eigen::Vector2f z(inputs.measurement.value.x, inputs.measurement.value.y);

    // On the first update after a reset, seed the position state directly from
    // the current measurement (velocity 0) so there is no slow pull from the
    // origin. This matches cv.jit.kalman initialising to the first measurement.
    if(!m_initialized)
    {
      m_x << z(0), z(1), 0.f, 0.f;
      m_P = Eigen::Matrix4f::Identity();
      m_initialized = true;
      outputs.estimate.value = {m_x(0), m_x(1)};
      outputs.velocity.value = {m_x(2), m_x(3)};
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
    Eigen::Vector2f u(inputs.control.value.x, inputs.control.value.y);

    // Process + measurement noise from the controls.
    Eigen::Matrix4f Q = Eigen::Matrix4f::Identity() * inputs.q.value;
    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * inputs.r.value;

    // Measurement matrix (observe position only).
    Eigen::Matrix<float, 2, 4> H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.f;
    H(1, 1) = 1.f;

    // Predict (incorporating the optional control input).
    m_x = F * m_x + B * u;
    m_P = F * m_P * F.transpose() + Q;

    // Update.
    Eigen::Vector2f y = z - H * m_x;
    Eigen::Matrix2f S = H * m_P * H.transpose() + R;
    Eigen::Matrix<float, 4, 2> K = m_P * H.transpose() * S.inverse();
    m_x = m_x + K * y;
    m_P = (Eigen::Matrix4f::Identity() - K * H) * m_P;

    outputs.estimate.value = {m_x(0), m_x(1)};
    outputs.velocity.value = {m_x(2), m_x(3)};
  }

private:
  void init()
  {
    m_x.setZero();
    m_P = Eigen::Matrix4f::Identity();
    // Defer seeding to the next update: the first measurement initialises state.
    m_initialized = false;
  }

  Eigen::Vector4f m_x;  // state
  Eigen::Matrix4f m_P;  // covariance
  bool m_was_reset = false;
  bool m_initialized = false;
};
}
