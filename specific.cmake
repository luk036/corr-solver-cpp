CPMAddPackage("gh:microsoft/GSL@3.1.0")
CPMAddPackage("gh:luk036/ellalgo-cpp#1.0.1")

find_package(OpenBLAS REQUIRED)
if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
  # target_include_directories(OpenBLAS::OpenBLAS SYSTEM INTERFACE ${OpenBLAS_INCLUDE_DIRS})
endif(OpenBLAS_FOUND)
if (WIN32)
  add_definitions(-DHAVE_CBLAS=0)
else()
  add_definitions(-DHAVE_CBLAS=1)
endif()

CPMAddPackage("gh:xtensor-stack/xtl#0.6.23")
message(STATUS "Found xtl: ${xtl_SOURCE_DIR}")
include_directories(${xtl_SOURCE_DIR}/include)

CPMAddPackage("gh:xtensor-stack/xtensor#0.22.0")
message(STATUS "Found xtensor: ${xtensor_SOURCE_DIR}")
include_directories(${xtensor_SOURCE_DIR}/include)

CPMAddPackage("gh:xtensor-stack/xtensor-blas#0.18.0")
# if(xtensor-blas_ADDED)
message(STATUS "Found xtensor-blas: ${xtensor-blas_SOURCE_DIR}")
include_directories(${xtensor-blas_SOURCE_DIR}/include)
# endif(xtensor-blas_ADDED) remember to turn off the warnings

set(SPECIFIC_LIBS EllAlgo::EllAlgo ${OpenBLAS_LIBRARIES} GSL)
