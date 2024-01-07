#include <gtest/gtest.h>

#define private public
#define protected public
#include "ekf_localizer/vel_measurement_model.hpp"
#undef protected
#undef private


using namespace ekf_localizer;

class VelMeasurementTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    vel = std::make_unique<VelMeasurement>();
  }
  void TearDown() override
  {
  }

  std::unique_ptr<VelMeasurement> vel;
};

TEST_F(VelMeasurementTest, Contructor_TC1)
{
  EXPECT_EQ(vel->size(), 1);
}

TEST_F(VelMeasurementTest, SetGet_TC1)
{
  vel->nu() = 1.0;

  EXPECT_DOUBLE_EQ(vel->nu(), 1.0);
}

class VelMeasurementModelTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    meas = std::make_unique<VelMeasurementModel>();
    x.setOnes();
  }
  void TearDown() override
  {
  }

  std::unique_ptr<VelMeasurementModel> meas;
  State x;
};

TEST_F(VelMeasurementModelTest, Contructor_TC1)
{
  EXPECT_DOUBLE_EQ(meas->chisq_threshold_, 3.841458821);
  EXPECT_TRUE(meas->H_.isZero());
  EXPECT_TRUE(meas->V_.isIdentity());
  EXPECT_TRUE(meas->getCovariance().isIdentity());
}

TEST_F(VelMeasurementModelTest, setCovariance_TC1)
{
  kalman::Covariance<VelMeasurement> R;
  R.setZero();
  R.diagonal() << 1.0;
  meas->setCovariance(R);

  EXPECT_EQ(meas->getCovariance().rows(), 1);
  EXPECT_EQ(meas->getCovariance().cols(), 1);
  EXPECT_TRUE(meas->getCovariance().isDiagonal());
  EXPECT_DOUBLE_EQ(meas->getCovariance()(VelMeasurement::NU, VelMeasurement::NU), 1.0);
}

TEST_F(VelMeasurementModelTest, h_TC1)
{
  VelMeasurement measurement;
  measurement = meas->h(x);

  EXPECT_DOUBLE_EQ(measurement.nu(), 1.0);
}

TEST_F(VelMeasurementModelTest, updateJacobians_TC1)
{
  meas->updateJacobians(x);
  EXPECT_DOUBLE_EQ(meas->H_(VelMeasurement::NU, State::NU), 1.0);

  EXPECT_TRUE(meas->V_.isIdentity());
}
