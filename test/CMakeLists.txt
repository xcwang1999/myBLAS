file(GLOB_RECURSE srcs CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)
add_executable(executable
    ${srcs}
)
find_package(OpenBLAS REQUIRED)
if(OpenBLAS_FOUND)
    message("OpenBLAS found.")
    message("OpenBLAS include directories: ${OpenBLAS_INCLUDE_DIRS}")
    message("OpenBLAS libraries: ${OpenBLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenBLAS not found!")
endif()
target_include_directories(executable PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenBLAS_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(executable PRIVATE
    myblas
    cublas
    ${OpenBLAS_LIBRARIES}
)