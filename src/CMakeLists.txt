set(MYBLAS_LIBRARIES "")

macro(add_sublibrary target)
    set(options NO_PTX_OPTIONS)
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    )
    add_library(${target} STATIC
        ${srcs}
    )
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
    )
    target_compile_options(${target} PRIVATE
        -w
    )
    if(NOT ARG_NO_PTX_OPTIONS)
        target_compile_options(${target} PRIVATE
            --ptxas-options=-v
        )
    endif()
    list(APPEND MYBLAS_LIBRARIES ${target})
    set(MYBLAS_LIBRARIES "${MYBLAS_LIBRARIES}" PARENT_SCOPE)
    set_target_properties(${target} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endmacro()

add_subdirectory(hgemm)
add_subdirectory(sgemm)
add_subdirectory(matrix)
add_subdirectory(others)

add_library(myblas INTERFACE)
message(STATUS "MYBLAS_LIBRARIES: ${MYBLAS_LIBRARIES}")
target_link_libraries(myblas INTERFACE
    ${MYBLAS_LIBRARIES}
)
