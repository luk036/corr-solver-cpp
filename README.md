[![Actions Status](https://github.com/luk036/corr-solver-cpp/workflows/MacOS/badge.svg)](https://github.com/luk036/corr-solver-cpp/actions)
[![Actions Status](https://github.com/luk036/corr-solver-cpp/workflows/Windows/badge.svg)](https://github.com/luk036/corr-solver-cpp/actions)
[![Actions Status](https://github.com/luk036/corr-solver-cpp/workflows/Ubuntu/badge.svg)](https://github.com/luk036/corr-solver-cpp/actions)
[![Actions Status](https://github.com/luk036/corr-solver-cpp/workflows/Style/badge.svg)](https://github.com/luk036/corr-solver-cpp/actions)
[![Actions Status](https://github.com/luk036/corr-solver-cpp/workflows/Install/badge.svg)](https://github.com/luk036/corr-solver-cpp/actions)
[![codecov](https://codecov.io/gh/luk036/corr-solver-cpp/branch/master/graph/badge.svg)](https://codecov.io/gh/luk036/corr-solver-cpp)

<p align="center">
  <img src="https://repository-images.githubusercontent.com/254842585/4dfa7580-7ffb-11ea-99d0-46b8fe2f4170" height="175" width="auto" />
</p>

# 🖇️ CorrSolverCpp

Correlation solver for modern C++

- This library requires C++20 or above.
- This library depends on EllAlgo.

## ✨ Features

- Lazy evaluation of Cholesky/LDLT decomposition
- Suport semi- and strictlu positive definite.
- [Modern CMake practices](https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/)
- Integrated test suite
- Continuous integration via [GitHub Actions](https://help.github.com/en/actions/)
- Code coverage via [codecov](https://codecov.io)
- Code formatting enforced by [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and [cmake-format](https://github.com/cheshirekow/cmake_format) via [Format.cmake](https://github.com/TheLartians/Format.cmake)
- Reproducible dependency management via [CPM.cmake](https://github.com/TheLartians/CPM.cmake)
- Installable target with automatic versioning information and header generation via [PackageProject.cmake](https://github.com/TheLartians/PackageProject.cmake)
- Automatic [documentation](https://thelartians.github.io/ModernCppStarter) and deployment with [Doxygen](https://www.doxygen.nl) and [GitHub Pages](https://pages.github.com)
- Optional [clang-tidy](#clang-tidy) analysis and [code coverage](#cmake-options) via gcovr

## Usage

### Adjust the template to your needs

- Use this repo [as a template](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template).
- Replace all occurrences of "CorrSolver" in the relevant CMakeLists.txt with the name of your project
  - Capitalization matters here: `CorrSolver` means the name of the project, while `corrsolver` is used in file names.
  - Remember to rename the `include/corrsolver` directory to use your project's lowercase name and update all relevant `#include`s accordingly.
- Replace the source files with your own
- For header-only libraries: see the comments in [CMakeLists.txt](CMakeLists.txt)
- Add [your project's codecov token](https://docs.codecov.io/docs/quick-start) to your project's github secrets under `CODECOV_TOKEN`
- Happy coding!

Eventually, you can remove any unused files, such as the standalone directory or irrelevant github workflows for your project.
Feel free to replace the License with one suited for your project.

All targets (library, standalone, benchmark, tests) are defined in a single root `CMakeLists.txt`, so a single configure step builds everything.

### Build and run the standalone target

Use the following command to build and run the executable target.

```bash
cmake -S. -B build
cmake --build build
./build/CorrSolver --help
```

### Build and run test suite

Use the following commands from the project's root directory to run the test suite.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

# or maybe simply call the executable:
./build/CorrSolverTests
```

To collect code coverage information, run CMake with the `-DCORRSOLVER_ENABLE_COVERAGE=ON` option.

### Run clang-format

Use the following commands from the project's root directory to check and fix C++ and CMake source style.
This requires _clang-format_, _cmake-format_ and _pyyaml_ to be installed on the current system.

```bash
cmake -S . -B build

# view changes
cmake --build build --target format

# apply changes
cmake --build build --target fix-format
```

See [Format.cmake](https://github.com/TheLartians/Format.cmake) for details.

### Build the documentation

The documentation is automatically built and [published](https://luk036.github.io/corr-solver-cpp) whenever a [GitHub Release](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) is created.
To manually build documentation, call the following command.

```bash
cmake -S . -B build -DCORRSOLVER_BUILD_DOCS=ON
cmake --build build --target GenerateDocs
# view the docs
open build/doxygen/html/index.html
```

To build the documentation locally, you will need Doxygen, jinja2 and Pygments on installed your system.

### Additional tools

#### Static analysis

clang-tidy can be enabled by configuring CMake with `-DCORRSOLVER_ENABLE_CLANG_TIDY=ON`; this provides a `clang-tidy` target that analyzes the public headers.

#### Code coverage

Code coverage (GCC/Clang, via gcovr) can be enabled by configuring CMake with `-DCORRSOLVER_ENABLE_COVERAGE=ON`; this provides a `coverage` target that runs the tests and writes an HTML report to `build/coverage/index.html`.

## Related projects and alternatives

- [**ModernCppStarter & PVS-Studio Static Code Analyzer**](https://github.com/viva64/pvs-studio-cmake-examples/tree/master/modern-cpp-starter): Official instructions on how to use the ModernCppStarter with the PVS-Studio Static Code Analyzer.
- [**lefticus/cpp_starter_project**](https://github.com/lefticus/cpp_starter_project/): A popular C++ starter project, created in 2017.
- [**filipdutescu/modern-cpp-template**](https://github.com/filipdutescu/modern-cpp-template): A recent starter using a more traditional approach for CMake structure and dependency management.
- [**vector-of-bool/pitchfork**](https://github.com/vector-of-bool/pitchfork/): Pitchfork is a Set of C++ Project Conventions.
