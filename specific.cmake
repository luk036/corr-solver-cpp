CPMAddPackage(
  NAME fmt
  GIT_TAG 10.2.1
  GITHUB_REPOSITORY fmtlib/fmt
  OPTIONS "FMT_INSTALL YES" # create an installable target
)

CPMAddPackage("gh:xtensor-stack/xtl#0.7.4")
if(xtl_ADDED)
  message(STATUS "Found xtl: ${xtl_SOURCE_DIR}")
  include_directories(${xtl_SOURCE_DIR}/include)
endif(xtl_ADDED)

CPMAddPackage("gh:xtensor-stack/xtensor#0.24.3")
if(xtensor_ADDED)
  message(STATUS "Found xtensor: ${xtensor_SOURCE_DIR}")
  include_directories(${xtensor_SOURCE_DIR}/include)
endif(xtensor_ADDED)

CPMAddPackage(
  NAME EllAlgo
  GIT_TAG 1.5.1
  GITHUB_REPOSITORY luk036/ellalgo-cpp
  OPTIONS "INSTALL_ONLY YES" # create an installable target
)

# CPMAddPackage( NAME LmiSolver GIT_TAG 1.3.8 GITHUB_REPOSITORY luk036/lmi-solver-cpp OPTIONS
# "INSTALL_ONLY YES" # create an installable target )

find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
  # target_include_directories(OpenBLAS::OpenBLAS SYSTEM INTERFACE ${OpenBLAS_INCLUDE_DIRS})
  include_directories(${OpenBLAS_INCLUDE_DIRS})
endif(OpenBLAS_FOUND)

find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
  message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
  include_directories(${LAPACK_INCLUDE_DIRS})
endif(LAPACK_FOUND)

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
  message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
  include_directories(${BLAS_INCLUDE_DIRS})
endif(BLAS_FOUND)

CPMAddPackage("gh:xtensor-stack/xtensor-blas#0.20.0")
if(xtensor-blas_ADDED)
  message(STATUS "Found xtensor-blas: ${xtensor-blas_SOURCE_DIR}")
  include_directories(${xtensor-blas_SOURCE_DIR}/include)
endif(xtensor-blas_ADDED)
# remember to turn off the warnings

if(WIN32)
  add_definitions(-DXTENSOR_USE_FLENS_BLAS)
endif()

set(SPECIFIC_LIBS EllAlgo::EllAlgo ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
                  fmt::fmt
)
