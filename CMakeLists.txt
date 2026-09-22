cmake_minimum_required(VERSION 3.14)
project(MHDSimulator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_executable(mhd_sim ${SOURCES})

target_include_directories(mhd_sim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
)

if(MSVC)
    target_compile_options(mhd_sim PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(mhd_sim PRIVATE -O3 -ffast-math -march=native)
endif()

target_link_libraries(mhd_sim PRIVATE Eigen3::Eigen)
