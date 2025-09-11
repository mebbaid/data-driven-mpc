#include "data-driven-mpc/data-driven-mpc.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>

namespace DataDrivenMPC {

// Test fixture
class DDMPCTest : public ::testing::Test {
protected:
    // You can define common setup code here (called before each test)
    void SetUp() override {
        Q = Eigen::MatrixXd::Identity(2, 2);
        R = Eigen::MatrixXd::Identity(1, 1);
    }

    // You can define common teardown code here (called after each test)
    void TearDown() override {}

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
};

// Test case for initialization
TEST_F(DDMPCTest, Initialization) {
    auto solver = std::make_unique<QPSolver>();
    DDMPC mpc(5, 10, 5, Q, R, std::move(solver));

    ASSERT_EQ(mpc.getLambdaU(), 0.0);
    ASSERT_EQ(mpc.getLambdaY(), 0.0);
    ASSERT_EQ(mpc.getLambdaG(), 0.0);

}

// Test case for setting input constraints
TEST_F(DDMPCTest, SetInputConstraints) {
    auto solver = std::make_unique<QPSolver>();
    DDMPC mpc(5, 10, 5, Q, R, std::move(solver));

    Eigen::VectorXd u_min(1);
    u_min << -1.0;
    Eigen::VectorXd u_max(1);
    u_max << 1.0;

    mpc.setInputConstraints(u_min, u_max);
}

// Test case for solve method (more involved)
TEST_F(DDMPCTest, Solve) {
    auto solver = std::make_unique<QPSolver>();
    DDMPC mpc(2, 3, 2, Q, R, std::move(solver));

    std::vector<Eigen::VectorXd> u_data = {
        Eigen::VectorXd::Constant(1, 0.5),
        Eigen::VectorXd::Constant(1, -0.5),
        Eigen::VectorXd::Constant(1, 0.2),
        Eigen::VectorXd::Constant(1, -0.2),
        Eigen::VectorXd::Constant(1, 0.0),
        Eigen::VectorXd::Constant(1, 0.0),
        Eigen::VectorXd::Constant(1, 0.0)
    };
    std::vector<Eigen::VectorXd> y_data = {
        Eigen::VectorXd::Constant(2, 0.0),
        Eigen::VectorXd::Constant(2, 1.0),
        Eigen::VectorXd::Constant(2, 0.5),
        Eigen::VectorXd::Constant(2, 0.5),
        Eigen::VectorXd::Constant(2, 0.5),
        Eigen::VectorXd::Constant(2, 0.5),
        Eigen::VectorXd::Constant(2, 0.5)
    };
     std::vector<Eigen::VectorXd> u_data_ini = {
        Eigen::VectorXd::Constant(1, 0.5),
        Eigen::VectorXd::Constant(1, -0.5),
    };
    std::vector<Eigen::VectorXd> y_data_ini = {
        Eigen::VectorXd::Constant(2, 0.0),
        Eigen::VectorXd::Constant(2, 1.0),
    };
    Eigen::VectorXd reference = Eigen::VectorXd::Constant(2, 1.0);
    Eigen::VectorXd u_prev = Eigen::VectorXd::Constant(1, 0.0);

    // add constraints
    Eigen::VectorXd u_min(1);
    u_min << -5.0;
    Eigen::VectorXd u_max(1);
    u_max << 5.0;
    mpc.setInputConstraints(u_min, u_max);
    Eigen::VectorXd u_opt = mpc.solve(u_data, y_data, reference, u_prev, u_data_ini, y_data_ini);

    // Check that the result has the correct size
    ASSERT_EQ(u_opt.size(), 1);

    // Add more assertions here to check the *correctness* of the solution.
    // This is where you'd put in expected values based on your understanding
    // of the MPC problem.  This might involve checking for constraint
    // satisfaction, optimality (if possible), etc.  For a simple test,
    // you could at least check that the result is within reasonable bounds.
    // For example:
    ASSERT_TRUE(u_opt(0) >= -5.0 && u_opt(0) <= 5.0);
}

} // namespace DataDrivenMPC

