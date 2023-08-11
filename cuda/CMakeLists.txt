cmake_minimum_required(VERSION 3.15)
project(cuda-perf-opt-test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
enable_language(CXX CUDA)

# for each cpp add exe
file(GLOB_RECURSE CPPS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
foreach(CPP ${CPPS})
    get_filename_component(EXE ${CPP} NAME_WE)
    add_executable(${EXE} ${CPP})
    target_link_libraries(${EXE} CUDA::cudart)
endforeach()
