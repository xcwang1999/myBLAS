cmake_minimum_required(VERSION 3.25)
project(myblas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found.")
    #    execute_process(COMMAND which nvcc OUTPUT_VARIABLE NVCC_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    #    set(CMAKE_CUDA_COMPILER ${NVCC_PATH})
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CUDA_STANDARD_REQUIRED "${CMAKE_CXX_STANDARD_REQUIRED}")
else ()
    message(FATAL_ERROR "CUDAToolkit not found!")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS off)
endif ()

find_package(Git REQUIRED)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    message(STATUS "Populating Git submodule.")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if (NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR
                "git submodule updata --init --recursive failed with ${GIT_SUBMOD_RESULT}.")
    endif ()
endif ()

add_subdirectory(src)

set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
add_subdirectory(third-party/googletest)
add_subdirectory(third-party/benchmark)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})

add_subdirectory(test)
add_subdirectory(benchmark)