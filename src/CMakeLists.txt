cmake_minimum_required(VERSION 3.15)
project(hip-perf-opt-test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(hip REQUIRED)
enable_language(CXX HIP)

# for each cpp add exe
file(GLOB_RECURSE CPPS ${CMAKE_CURRENT_SOURCE_DIR}/*.hip)
foreach(CPP ${CPPS})
    get_filename_component(EXE ${CPP} NAME_WE)
    add_executable(${EXE} ${CPP})
    target_link_libraries(${EXE} hip::host)
endforeach()
