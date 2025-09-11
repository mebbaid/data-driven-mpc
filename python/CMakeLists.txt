cmake_minimum_required(VERSION 3.15)
project(py_ddmpc)

find_package(pybind11 REQUIRED)

if(Python3_VERSION VERSION_GREATER_EQUAL 3.12)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import os;import sysconfig;relative_site_packages = sysconfig.get_path('purelib').replace(sysconfig.get_path('data'), '').lstrip(os.path.sep);print(relative_site_packages)"
                    OUTPUT_VARIABLE _PYTHON_INSTDIR)
else()
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,0,prefix=''))"
                    OUTPUT_VARIABLE _PYTHON_INSTDIR)
endif()

string(STRIP ${_PYTHON_INSTDIR} PYTHON_INSTALL_DIR)

pybind11_add_module(py_ddmpc
    src/py-ddmpc.cpp
)

target_link_libraries(py_ddmpc
    PUBLIC
        DataDrivenMPC::data-driven-mpc
        ${PYTHON_LIBRARIES}
)

set_target_properties(py_ddmpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bindings OUTPUT_NAME "py_ddmpc")

install(TARGETS py_ddmpc DESTINATION ${PYTHON_INSTALL_DIR}/py_ddmpc COMPONENT python)
