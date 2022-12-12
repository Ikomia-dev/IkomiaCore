cmake_minimum_required(VERSION 3.16)
project(IkomiaCore VERSION 0.9.0 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Set up AUTOMOC and some sensible defaults for runtime execution
# When using Qt 6.3, you can replace the code block below with
# qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
include(GNUInstallDirs)
include(Versions.cmake)
include(Common.cmake)

# Qt
find_package(Qt5 REQUIRED COMPONENTS Core Concurrent Gui Network OpenGL Sql Test Widgets)

# Python
find_package(Python3 ${PYTHON_VERSION_DOT} REQUIRED)

# OpenMP
find_package(OpenMP)

# OpenCV
find_package(OpenCV REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem python${PYTHON_VERSION_NO_DOT})

# VTK
find_package(VTK)

add_subdirectory(Utils)
add_subdirectory(Core)
add_subdirectory(DataIO)
add_subdirectory(DataManagement)
add_subdirectory(DataProcess)
add_subdirectory(DataPlot)
add_subdirectory(VolumeRender)
add_subdirectory(PyDataIO)
add_subdirectory(PyCore)
add_subdirectory(PyUtils)
add_subdirectory(PyDataProcess)
add_subdirectory(UnitTests)
