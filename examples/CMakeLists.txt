
include_directories(${Python3_NumPy_INCLUDE_DIRS})
add_executable(simpleExample simpleExample.cpp)
target_include_directories(simpleExample PRIVATE
   ${Python3_INCLUDE_DIRS}
   ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(simpleExample
        DataDrivenMPC::data-driven-mpc
        Python3::Python
)

add_executable(lipmExample lipmExample.cpp)
target_include_directories(lipmExample PRIVATE
   ${Python3_INCLUDE_DIRS}
   ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(lipmExample
        DataDrivenMPC::data-driven-mpc
        Python3::Python
)