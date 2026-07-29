file(GLOB BENCHMARK_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

add_executable(
  ${PROJECT_NAME}Benchmark ${BENCHMARK_SOURCES} ${PROJECT_SOURCE_DIR}/source/qmi_oracle.cpp
)
target_link_libraries(${PROJECT_NAME}Benchmark ${PROJECT_NAME}::${PROJECT_NAME} ${SPECIFIC_LIBS})
set_target_properties(${PROJECT_NAME}Benchmark PROPERTIES CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(${PROJECT_NAME}Benchmark PRIVATE -Wall -Wextra -Wconversion)
elseif(MSVC)
  target_compile_options(${PROJECT_NAME}Benchmark PRIVATE /utf-8 /W4 /WX)
endif()
