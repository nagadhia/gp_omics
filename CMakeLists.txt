cmake_minimum_required(VERSION 3.10)

project(GP_omics)
project(GP_omics_N)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the installed libraries via vcpkg
find_package(fmt CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)

add_executable(GP_omics GPC_for_omics.cpp)
add_executable(GP_omics_N GPC_for_omics_nan.cpp)

target_link_libraries(GP_omics PRIVATE fmt::fmt)
target_link_libraries(GP_omics PRIVATE armadillo)

target_link_libraries(GP_omics_N PRIVATE fmt::fmt)
target_link_libraries(GP_omics_N PRIVATE armadillo)