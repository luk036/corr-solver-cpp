# [=======================================================================
# FindOpenBLAS.cmake
# -----------------
#
# Find the OpenBLAS library (BLAS + LAPACK).
#
# This module searches in the following order: 1. OpenBLAS_HOME environment variable 2. CONDA_PREFIX
# environment variable 3. CMAKE_PREFIX_PATH / default system paths
#
# Exported variables: OpenBLAS_FOUND         - TRUE if OpenBLAS was found OpenBLAS_LIBRARIES     -
# list of library paths OpenBLAS_INCLUDE_DIRS  - list of include directories OpenBLAS_VERSION -
# version string
# =======================================================================]

if(OpenBLAS_FOUND)
  return()
endif()

# ---- Collect search roots ----
set(_openblas_roots)

if(DEFINED ENV{OpenBLAS_HOME})
  list(APPEND _openblas_roots "$ENV{OpenBLAS_HOME}")
endif()

if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _openblas_roots "$ENV{CONDA_PREFIX}")
endif()

# macOS Homebrew paths (brew install openblas)
if(DEFINED ENV{HOMEBREW_PREFIX})
  list(APPEND _openblas_roots "$ENV{HOMEBREW_PREFIX}")
  list(APPEND _openblas_roots "$ENV{HOMEBREW_PREFIX}/opt/openblas")
else()
  # Typical Apple Silicon Homebrew path
  list(APPEND _openblas_roots "/opt/homebrew/opt/openblas")
  # Typical Intel Homebrew path
  list(APPEND _openblas_roots "/usr/local/opt/openblas")
endif()

# ---- Find include directory ----
find_path(
  OpenBLAS_INCLUDE_DIRS
  NAMES openblas/cblas.h openblas/openblas_config.h
  PATHS ${_openblas_roots}
  PATH_SUFFIXES
    Library/include # Windows conda layout
    include # Unix conda / generic layout
    include/openblas
  NO_DEFAULT_PATH
)

# If not found via env roots, fall back to default paths
if(NOT OpenBLAS_INCLUDE_DIRS)
  find_path(
    OpenBLAS_INCLUDE_DIRS
    NAMES openblas/cblas.h openblas/openblas_config.h
    PATH_SUFFIXES include include/openblas openblas
  )
endif()

# ---- Find library ----
find_library(
  OpenBLAS_LIBRARIES
  NAMES openblas libopenblas
  PATHS ${_openblas_roots}
  PATH_SUFFIXES Library/lib # Windows conda layout
                lib # Unix conda / generic layout
  NO_DEFAULT_PATH
)

if(NOT OpenBLAS_LIBRARIES)
  find_library(
    OpenBLAS_LIBRARIES
    NAMES openblas libopenblas
    PATH_SUFFIXES lib
  )
endif()

# ---- Config-mode fallback for system-installed OpenBLAS ----
# Homebrew (macOS) and apt (Ubuntu) install their own OpenBLASConfig.cmake. If our module-mode
# search didn't find it, try config mode.
if(NOT OpenBLAS_FOUND)
  # Prevent recursion: this will skip Module mode and go directly to Config mode
  find_package(OpenBLAS CONFIG QUIET)

  # Config mode may set IMPORTED targets but not the legacy variables. Extract variables from the
  # IMPORTED target if needed.
  if(OpenBLAS_FOUND AND NOT OpenBLAS_LIBRARIES)
    if(TARGET OpenBLAS::OpenBLAS)
      get_target_property(_loc OpenBLAS::OpenBLAS IMPORTED_LOCATION)
      if(_loc)
        set(OpenBLAS_LIBRARIES "${_loc}")
      endif()
    elseif(TARGET OpenBLAS::openblas)
      get_target_property(_loc OpenBLAS::openblas IMPORTED_LOCATION)
      if(_loc)
        set(OpenBLAS_LIBRARIES "${_loc}")
      endif()
    endif()
  endif()

  if(OpenBLAS_FOUND AND NOT OpenBLAS_INCLUDE_DIRS)
    if(TARGET OpenBLAS::OpenBLAS)
      get_target_property(_inc OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
      if(_inc)
        set(OpenBLAS_INCLUDE_DIRS "${_inc}")
      endif()
    elseif(TARGET OpenBLAS::openblas)
      get_target_property(_inc OpenBLAS::openblas INTERFACE_INCLUDE_DIRECTORIES)
      if(_inc)
        set(OpenBLAS_INCLUDE_DIRS "${_inc}")
      endif()
    endif()
  endif()

  if(OpenBLAS_FOUND AND NOT OpenBLAS_VERSION)
    if(TARGET OpenBLAS::OpenBLAS)
      get_target_property(_ver OpenBLAS::OpenBLAS IMPORTED_VERSION)
      if(_ver)
        set(OpenBLAS_VERSION "${_ver}")
      endif()
    endif()
  endif()
endif()

# ---- Set version (default if not provided by config mode) ----
if(NOT OpenBLAS_VERSION)
  set(OpenBLAS_VERSION "0.3.31")
endif()

# ---- Handle standard arguments ----
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenBLAS
  REQUIRED_VARS OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIRS
  VERSION_VAR OpenBLAS_VERSION
)

mark_as_advanced(OpenBLAS_INCLUDE_DIRS OpenBLAS_LIBRARIES)
