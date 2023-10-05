file(GLOB_RECURSE srcs CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)
add_executable(executable
    ${srcs}
)
target_include_directories(executable PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third-party/OpenBLAS
    ${PROJECT_BINARY_DIR}/third-party/OpenBLAS
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(executable PRIVATE
    myblas
    cublas
    openblas
)