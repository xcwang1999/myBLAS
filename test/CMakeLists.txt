macro(add_myblas_unit_test target)
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
            ${ARGN}
    )
    enable_testing()
    add_executable(${target}
            ${srcs}
    )
    target_include_directories(${target} SYSTEM PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_include_directories(${target} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src/include
    )
    target_link_libraries(${target} PRIVATE
            myblas
            cublas
            GTest::gtest
            CUDA::cudart
    )
endmacro()

include(GoogleTest)

add_subdirectory(sgemm)
add_subdirectory(layout)
add_subdirectory(ptx)
