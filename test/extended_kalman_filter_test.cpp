#include <gtest/gtest.h>

#define private public
#include "ekf_localizer/extended_kalman_filter.hpp"
#undef private

#include "ekf_localizer/state.hpp"
#include "ekf_localizer/system_model.hpp"
#include "ekf_localizer/gps_measurement_model.hpp"
#include "ekf_localizer/imu_measurement_model.hpp"
#include "ekf_localizer/vel_measurement_model.hpp"


using namespace ekf_localizer;

class EKFTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    ekf = std::make_unique<EKF>();
    init_x.setOnes();
    init_P.setIdentity();

    sys = std::make_unique<SystemModel>(dt);

    gps = std::make_unique<GpsMeasurementModel>();
    zg << 1.1, 1.1;

    imu = std::make_unique<ImuMeasurementModel>();
    zi << 1.5, 1.5, 1.5;

    vel = std::make_unique<VelMeasurementModel>();
    zv << 1.5;
  }

  void TearDown() override
  {
  }

  std::unique_ptr<EKF> ekf;
  State init_x;
  kalman::Covariance<State> init_P;

  std::unique_ptr<SystemModel> sys;
  double dt = 0.1;

  std::unique_ptr<GpsMeasurementModel> gps;
  GpsMeasurement zg;

  std::unique_ptr<ImuMeasurementModel> imu;
  ImuMeasurement zi;

  std::unique_ptr<VelMeasurementModel> vel;
  VelMeasurement zv;
};

TEST_F(EKFTest, Contructor_TC1)
{
  EXPECT_TRUE(ekf->getState().isZero());
  EXPECT_TRUE(ekf->getCovariance().isIdentity());
  EXPECT_TRUE(ekf->I_.isIdentity());
  EXPECT_EQ(ekf->getState().rows(), 6);
  EXPECT_EQ(ekf->getCovariance().rows(), 6);
  EXPECT_EQ(ekf->getCovariance().cols(), 6);
}

TEST_F(EKFTest, Init_TC1)
{
  ekf->init(init_x);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::X), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::Y), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::THETA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::NU), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::ALPHA), 1.0);

  ekf->setCovariance(init_P * 2);
  EXPECT_TRUE(ekf->getCovariance().isDiagonal());
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::X), 2.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::Y), 2.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::THETA), 2.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::NU), 2.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::OMEGA), 2.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::ALPHA), 2.0);
}

TEST_F(EKFTest, Predict_TC1)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);
  ekf->predict(*sys);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::X), 1.0522449600286314);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::Y), 1.0910794386873719);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::THETA), 1.1);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::NU), 1.1);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::ALPHA), 1.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::X), 1.0108250897967503423);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::Y), -0.0004446067744095863);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::THETA), -0.09199023307424552);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::NU), 0.0502546758370644);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::OMEGA), -0.009107943868737180);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::ALPHA), 0.004975710478917271);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::X), -0.0004446067744095863);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::Y), 1.0103050352032492132);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::THETA), 0.05276740962891765);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::NU), 0.08760974578499572);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::OMEGA), 0.005224496002863135);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::ALPHA), 0.008674232255940171);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::X), -0.0919902330742455171);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::Y), 0.0527674096289176484);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::THETA), 1.02);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::OMEGA), 0.2);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::X), 0.0502546758370644164);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::Y), 0.0876097457849957240);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::NU), 1.02);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::ALPHA), 0.2);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::X), -0.0091079438687371797);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::Y), 0.0052244960028631346);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::THETA), 0.2);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::OMEGA), 2.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::X), 0.0049757104789172708);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::Y), 0.0086742322559401706);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::NU), 0.2);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::ALPHA), 2.0);
}

TEST_F(EKFTest, GpsUpdate_TC1)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);
  EXPECT_TRUE(ekf->update(*gps, zg));

  EXPECT_DOUBLE_EQ(ekf->getState()(State::X), 1.05);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::Y), 1.05);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::THETA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::NU), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::ALPHA), 1.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::X), 0.5);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::Y), 0.5);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::THETA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::NU), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::ALPHA), 1.0);
}

TEST_F(EKFTest, GpsUpdate_TC2)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);

  GpsMeasurementModel gps;
  GpsMeasurement zg;
  zg << 100.0, 100.0;

  EXPECT_FALSE(ekf->update(gps, zg));
  EXPECT_TRUE(ekf->getState().isOnes());
  EXPECT_TRUE(ekf->getCovariance().isIdentity());
}

TEST_F(EKFTest, ImuUpdate_TC1)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);
  EXPECT_TRUE(ekf->update(*imu, zi));

  EXPECT_DOUBLE_EQ(ekf->getState()(State::X), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::Y), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::THETA), 1.25);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::NU), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::OMEGA), 1.25);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::ALPHA), 1.25);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::X), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::Y), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::THETA), 0.5);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::NU), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::OMEGA), 0.5);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::ALPHA), 0.5);
}

TEST_F(EKFTest, ImuUpdate_TC2)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);

  ImuMeasurementModel imu;
  ImuMeasurement zi;
  zi << 100.0, 100.0, 100.0;

  EXPECT_FALSE(ekf->update(imu, zi));
  EXPECT_TRUE(ekf->getState().isOnes());
  EXPECT_TRUE(ekf->getCovariance().isIdentity());
}

TEST_F(EKFTest, VelUpdate_TC1)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);
  EXPECT_TRUE(ekf->update(*vel, zv));

  EXPECT_DOUBLE_EQ(ekf->getState()(State::X), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::Y), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::THETA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::NU), 1.25);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getState()(State::ALPHA), 1.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::X), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::X, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::Y), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::Y, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::THETA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::THETA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::NU), 0.5);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::NU, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::OMEGA, State::ALPHA), 0.0);

  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::X), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::Y), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::THETA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::NU), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::OMEGA), 0.0);
  EXPECT_DOUBLE_EQ(ekf->getCovariance()(State::ALPHA, State::ALPHA), 1.0);
}

TEST_F(EKFTest, VelUpdate_TC2)
{
  ekf->init(init_x);
  ekf->setCovariance(init_P);

  VelMeasurementModel vel;
  VelMeasurement zv;
  zv << 100.0;

  EXPECT_FALSE(ekf->update(vel, zv));
  EXPECT_TRUE(ekf->getState().isOnes());
  EXPECT_TRUE(ekf->getCovariance().isIdentity());
}
