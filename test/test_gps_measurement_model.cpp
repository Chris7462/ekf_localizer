#include <gtest/gtest.h>

#define private public
#define protected public
#include "ekf_localizer/gps_measurement_model.hpp"
#undef protected
#undef private


using namespace ekf_localizer;

class GpsMeasurementTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    gps = std::make_unique<GpsMeasurement>();
  }
  void TearDown() override
  {
  }

  std::unique_ptr<GpsMeasurement> gps;
};

TEST_F(GpsMeasurementTest, Contructor_TC1)
{
  EXPECT_EQ(gps->size(), 2);
}

TEST_F(GpsMeasurementTest, SetGet_TC1)
{
  gps->x() = 1.0;
  gps->y() = 2.0;

  EXPECT_DOUBLE_EQ(gps->x(), 1.0);
  EXPECT_DOUBLE_EQ(gps->y(), 2.0);
}

class GpsMeasurementModelTest : public ::testing::Test
{
public:
  void SetUp() override
  {
    meas = std::make_unique<GpsMeasurementModel>();
    x.setOnes();
  }
  void TearDown() override
  {
  }

  std::unique_ptr<GpsMeasurementModel> meas;
  State x;
};

TEST_F(GpsMeasurementModelTest, Contructor_TC1)
{
  EXPECT_DOUBLE_EQ(meas->chisq_threshold_, 5.991464547);
  EXPECT_TRUE(meas->H_.isZero());
  EXPECT_TRUE(meas->V_.isIdentity());
  EXPECT_TRUE(meas->getCovariance().isIdentity());
}

TEST_F(GpsMeasurementModelTest, setCovariance_TC1)
{
  kalman::Covariance<GpsMeasurement> R;
  R.setZero();
  R.diagonal() << 1.0, 2.0;
  meas->setCovariance(R);

  EXPECT_EQ(meas->getCovariance().rows(), 2);
  EXPECT_EQ(meas->getCovariance().cols(), 2);
  EXPECT_TRUE(meas->getCovariance().isDiagonal());
  EXPECT_DOUBLE_EQ(meas->getCovariance()(GpsMeasurement::X, GpsMeasurement::X), 1.0);
  EXPECT_DOUBLE_EQ(meas->getCovariance()(GpsMeasurement::Y, GpsMeasurement::Y), 2.0);
}

TEST_F(GpsMeasurementModelTest, h_TC1)
{
  GpsMeasurement measurement;
  measurement = meas->h(x);

  EXPECT_DOUBLE_EQ(measurement.x(), 1.0);
  EXPECT_DOUBLE_EQ(measurement.y(), 1.0);
}

TEST_F(GpsMeasurementModelTest, updateJacobians_TC1)
{
  meas->updateJacobians(x);
  EXPECT_DOUBLE_EQ(meas->H_(GpsMeasurement::X, State::X), 1.0);
  EXPECT_DOUBLE_EQ(meas->H_(GpsMeasurement::Y, State::Y), 1.0);

  EXPECT_TRUE(meas->V_.isIdentity());
}
