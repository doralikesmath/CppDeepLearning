cmake_minimum_required(VERSION 3.13)

project(DeepLearning)
set (DeepLearning_VERSION_MAJOR 0)
set (DeepLearning_VERSION_MINOR 1)
set(CMAKE_CXX_STANDARD 14)

# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

FOREACH(subdir ${SUBDIRS})
    ADD_SUBDIRECTORY(${subdir})
    INCLUDE_DIRECTORIES(${subdir})
ENDFOREACH()

add_executable(kmean src/include/clustering/kmean.h examples/kmean_clustering.cpp)
add_executable(xornn examples/xornn.cpp)
add_executable(xornn_2 examples/xornn_take_2.cpp)