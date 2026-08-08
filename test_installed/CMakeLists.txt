cmake_minimum_required(VERSION 3.14)

project(CorrSolverInstalledTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies first - CorrSolverConfig.cmake references EllAlgo::EllAlgo (installed alongside
# the library by `cmake --install`).
find_package(EllAlgo REQUIRED)
find_package(CorrSolver REQUIRED)

add_executable(corrsolver_installed_test main.cpp)
target_link_libraries(corrsolver_installed_test PRIVATE CorrSolver::CorrSolver)

enable_testing()
add_test(NAME corrsolver_installed_test COMMAND corrsolver_installed_test)
