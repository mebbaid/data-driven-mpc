set(GTEST_LINKED_AS_SHARED_LIBRARY 1)
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(ddmpc-test test-ddmpc.cpp)

target_include_directories(ddmpc-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(ddmpc-test
    DataDrivenMPC::data-driven-mpc
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(ddmpc-test)