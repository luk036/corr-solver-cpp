set_languages("c++20")

add_rules("mode.debug", "mode.release", "mode.coverage")
add_requires("fmt", "doctest", "spdlog", "nanobench")

if is_mode("release") then
    set_optimize("fastest")
end

if is_mode("coverage") then
    add_cxflags("-ftest-coverage", "-fprofile-arcs", {force = true})
end

if is_plat("linux") then
    set_warnings("all", "error")
    -- host-native tuning for auto-vectorization
    if is_mode("release") then
        add_cxflags("-march=native", "-mtune=native", {force = true})
    end
elseif is_plat("windows") then
    add_cxflags("/EHsc /utf-8 /W4 /WX /wd4702", {force = true})
    if is_mode("release") then
        add_cxflags("/arch:AVX2", {force = true})
    end
end

-- sibling projects: Arr (ellalgo) and the LDS generators (lds-gen)
local ellalgo_dir = path.join(os.projectdir(), "../ellalgo-cpp")
local ellalgo_inc = path.join(ellalgo_dir, "include")
local ldsgen_inc = path.join(os.projectdir(), "../lds-gen-cpp/include")

add_includedirs("include", ellalgo_inc, ldsgen_inc)

target("EllAlgo")
    set_kind("static")
    add_includedirs(ellalgo_inc, {public = true})
    add_files(path.join(ellalgo_dir, "source/*.cpp"))
    add_packages("fmt", "spdlog")
    set_group("Dependencies")

target("CorrSolver")
    set_kind("static")
    add_includedirs(ellalgo_inc, {public = true})
    add_includedirs(ldsgen_inc, {public = true})
    add_files("source/*.cpp")
    add_deps("EllAlgo")
    add_packages("fmt")

target("test_corr_solver")
    add_deps("CorrSolver", "EllAlgo")
    add_files("test/source/*.cpp")
    add_packages("fmt", "doctest")
    add_tests("default")

target("benchmark_corr_solver")
    add_deps("EllAlgo")
    add_files("benchmark/source/*.cpp")
    add_files("source/*.cpp")
    add_packages("fmt", "nanobench")

target("corr_experiments")
    add_deps("CorrSolver")
    add_files("experiments/source/*.cpp")
    add_packages("fmt")
