set(MYBLAS_LIBRARIES "")
macro(add_myblas_sublibrary target)
    set(options NO_PTX_OPTIONS)
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    )
    add_library(${target} STATIC
            ${srcs}
    )
    target_include_directories(${target} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src/include
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_options(${target} PRIVATE
            -w
    )
    if (NOT ARG_NO_PTX_OPTIONS)
        target_compile_options(${target} PRIVATE
                --ptxas-options=-v
        )
    endif ()
    list(APPEND MYBLAS_LIBRARIES ${target})
    set(MYBLAS_LIBRARIES "${MYBLAS_LIBRARIES}" PARENT_SCOPE)
    set_target_properties(${target} PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
    )
endmacro()

add_subdirectory(hgemm)
add_subdirectory(sgemm)
add_subdirectory(transpose)

add_library(myblas STATIC myblas.cpp)
message(STATUS "MYBLAS_LIBRARIES: ${MYBLAS_LIBRARIES}")
target_link_libraries(myblas PUBLIC ${MYBLAS_LIBRARIES})
target_include_directories(myblas PUBLIC ${PROJECT_SOURCE_DIR}/include)

