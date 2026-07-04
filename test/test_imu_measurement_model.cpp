#include <gtest/gtest.h>

#define private public
#define protected public
#include "ekf_localizer/imu_measurement_model.hpp"
#undef protected
#undef private


using namespace ekf_localizer;

class ImuMeasurementTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    imu = std::make_unique<ImuMeasurement>();
  }
  void TearDown() override
  {
  }

  std::unique_ptr<ImuMeasurement> imu;
};

TEST_F(ImuMeasurementTest, Contructor_TC1)
{
  EXPECT_EQ(imu->size(), 3);
}

TEST_F(ImuMeasurementTest, SetGet_TC1)
{
  imu->theta() = 1.0;
  imu->omega() = 2.0;
  imu->alpha() = 3.0;

  EXPECT_DOUBLE_EQ(imu->theta(), 1.0);
  EXPECT_DOUBLE_EQ(imu->omega(), 2.0);
  EXPECT_DOUBLE_EQ(imu->alpha(), 3.0);
}

class ImuMeasurementModelTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    meas = std::make_unique<ImuMeasurementModel>();
    x.setOnes();
  }
  void TearDown() override
  {
  }

  std::unique_ptr<ImuMeasurementModel> meas;
  State x;
};

TEST_F(ImuMeasurementModelTest, Contructor_TC1)
{
  EXPECT_DOUBLE_EQ(meas->chisq_threshold_, 7.814727903);
  EXPECT_TRUE(meas->H_.isZero());
  EXPECT_TRUE(meas->V_.isIdentity());
  EXPECT_TRUE(meas->getCovariance().isIdentity());
}

TEST_F(ImuMeasurementModelTest, setCovariance_TC1)
{
  kalman::Covariance<ImuMeasurement> R;
  R.setZero();
  R.diagonal() << 1.0, 2.0, 3.0;
  meas->setCovariance(R);

  EXPECT_EQ(meas->getCovariance().rows(), 3);
  EXPECT_EQ(meas->getCovariance().cols(), 3);
  EXPECT_TRUE(meas->getCovariance().isDiagonal());
  EXPECT_DOUBLE_EQ(meas->getCovariance()(ImuMeasurement::THETA, ImuMeasurement::THETA), 1.0);
  EXPECT_DOUBLE_EQ(meas->getCovariance()(ImuMeasurement::OMEGA, ImuMeasurement::OMEGA), 2.0);
  EXPECT_DOUBLE_EQ(meas->getCovariance()(ImuMeasurement::ALPHA, ImuMeasurement::ALPHA), 3.0);
}

TEST_F(ImuMeasurementModelTest, h_TC1)
{
  ImuMeasurement measurement;
  measurement = meas->h(x);

  EXPECT_DOUBLE_EQ(measurement.theta(), 1.0);
  EXPECT_DOUBLE_EQ(measurement.omega(), 1.0);
  EXPECT_DOUBLE_EQ(measurement.alpha(), 1.0);
}

TEST_F(ImuMeasurementModelTest, updateJacobians_TC1)
{
  meas->updateJacobians(x);
  EXPECT_DOUBLE_EQ(meas->H_(ImuMeasurement::THETA, State::THETA), 1.0);
  EXPECT_DOUBLE_EQ(meas->H_(ImuMeasurement::OMEGA, State::OMEGA), 1.0);
  EXPECT_DOUBLE_EQ(meas->H_(ImuMeasurement::ALPHA, State::ALPHA), 1.0);

  EXPECT_TRUE(meas->V_.isIdentity());
}
