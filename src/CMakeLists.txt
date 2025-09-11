add_library(data-driven-mpc SHARED
    ddmpc.cpp
    hankel-matrix.cpp
)

# add alias for the library
add_library(DataDrivenMPC::data-driven-mpc ALIAS data-driven-mpc)

target_include_directories(data-driven-mpc PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(data-driven-mpc PUBLIC
    Eigen3::Eigen
    QpSolversEigen::QpSolversEigen

)

install(TARGETS data-driven-mpc
    EXPORT data-driven-mpc-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

install(EXPORT data-driven-mpc-targets
FILE DataDrivenMPCTargets.cmake
     NAMESPACE DataDrivenMPC::
     DESTINATION lib/cmake/DataDrivenMPC
)