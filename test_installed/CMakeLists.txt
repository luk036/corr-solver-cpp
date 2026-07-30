cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

project(CorrSolver LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies (all installed by the parent build) ----

include(../cmake/CPM.cmake)

# Find dependencies first — CorrSolverConfig.cmake references fmt::fmt and EllAlgo::EllAlgo
find_package(fmt CONFIG REQUIRED)
# CPMAddPackage for EllAlgo instead of find_package because its installed config
# references missing targets. This matches the pattern used by successful projects.
CPMAddPackage(
  NAME EllAlgo
  GIT_TAG v1.6.8
  GITHUB_REPOSITORY luk036/ellalgo-cpp
  OPTIONS "INSTALL_ONLY YES"
)
find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
else()
  find_package(LAPACK REQUIRED)
  find_package(BLAS REQUIRED)
endif()
find_package(${PROJECT_NAME} REQUIRED)

set(SPECIFIC_LIBS EllAlgo::EllAlgo fmt::fmt
    ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../standalone ${CMAKE_BINARY_DIR}/test_installed)
