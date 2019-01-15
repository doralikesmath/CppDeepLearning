cmake_minimum_required(VERSION 3.13)
project(DeepLearning)
set (DeepLearning_VERSION_MAJOR 0)
set (DeepLearning_VERSION_MINOR 1)
set(CMAKE_CXX_STANDARD 11)


FOREACH(subdir ${SUBDIRS})
    ADD_SUBDIRECTORY(${subdir})
    INCLUDE_DIRECTORIES(${subdir})
ENDFOREACH()

# file(GLOB SOURCES "src/models/*.cpp")
# file(GLOB SOURCES "src/utils/*.cpp")

add_executable(deeplearning main.cpp)