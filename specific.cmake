CPMAddPackage(
  NAME fmt
  GIT_TAG 12.1.0
  GITHUB_REPOSITORY fmtlib/fmt
  OPTIONS "FMT_INSTALL YES" # create an installable target
)

CPMAddPackage(
  NAME EllAlgo
  GIT_TAG 1.6.6
  GITHUB_REPOSITORY luk036/ellalgo-cpp
  OPTIONS "INSTALL_ONLY YES" # create an installable target
)

# CPMAddPackage( NAME LmiSolver GIT_TAG 1.3.8 GITHUB_REPOSITORY luk036/lmi-solver-cpp OPTIONS
# "INSTALL_ONLY YES" # create an installable target )

find_package(OpenBLAS)
if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
  include_directories(${OpenBLAS_INCLUDE_DIRS})
  # OpenBLAS provides both BLAS and LAPACK -- no separate search needed
  set(_have_blas_lapack TRUE)
else()
  find_package(LAPACK REQUIRED)
  if(LAPACK_FOUND)
    message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
    include_directories(${LAPACK_INCLUDE_DIRS})
  endif()

  find_package(BLAS REQUIRED)
  if(BLAS_FOUND)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
    include_directories(${BLAS_INCLUDE_DIRS})
  endif()
  set(_have_blas_lapack TRUE)
endif()

if(WIN32)
  add_definitions(-DXTENSOR_USE_FLENS_BLAS)
endif()

set(SPECIFIC_LIBS EllAlgo::EllAlgo ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
                  fmt::fmt
)
