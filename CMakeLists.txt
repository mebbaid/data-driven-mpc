cmake_minimum_required(VERSION 3.10)
project(data-driven-mpc)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(QpSolversEigen REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OSQP_INCLUDE_DIRS}
)

add_library(data-driven-mpc STATIC
    src/ddmpc.cpp
    src/hankel-matrix.cpp
)

target_link_libraries(data-driven-mpc
        Eigen3::Eigen
        QpSolversEigen::QpSolversEigen
)

add_executable(simpleExample examples/simpleExample.cpp)

target_link_libraries(simpleExample
    data-driven-mpc
)

# Add include directories for the EXECUTABLE
target_include_directories(simpleExample
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

# Optional: Create an install target
install(TARGETS data-driven-mpc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)