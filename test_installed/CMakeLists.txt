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

# Disable find_package for fmt to avoid target conflict with CPM-built fmt
# (the build+install step installs fmt-config.cmake which creates fmt::fmt,
# but transitive deps like EllAlgo use CPMAddPackage for fmt → duplicate ALIAS)
set(CMAKE_DISABLE_FIND_PACKAGE_fmt ON)
include(../specific.cmake)

find_package(${PROJECT_NAME} REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../standalone ${CMAKE_BINARY_DIR}/test_installed)
