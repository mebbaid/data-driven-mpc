cmake_minimum_required(VERSION 3.15)
project(DataDrivenMPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(QpSolversEigen REQUIRED)

option(ENABLE_EXAMPLES "Enable examples" OFF)
option(ENABLE_TESTING "Enable testing" OFF)
option(ENABLE_PYTHON "Enable Python bindings" OFF)

# if python or examples, find python
if(ENABLE_PYTHON OR ENABLE_EXAMPLES)
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
endif()

add_subdirectory(src)
if(ENABLE_TESTING)
    add_subdirectory(test)

endif()
if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()
if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
