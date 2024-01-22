macro(add_myblas_benchmark target)
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
            ${ARGN}
    )
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
            CUDA::cudart
            benchmark::benchmark
    )
endmacro()

add_subdirectory(sgemm)