cmake_minimum_required(VERSION 3.16)
project(UnitTests VERSION 1.0 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Set up AUTOMOC and some sensible defaults for runtime execution
# When using Qt 6.3, you can replace the code block below with
# qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
include(GNUInstallDirs)

find_package(QT NAMES Qt5 Qt6 REQUIRED COMPONENTS Core)

if(NOT IKOMIA_TESTS)
    return()
endif()

add_subdirectory(DataIOTests)
add_subdirectory(CoreTests)
add_subdirectory(DataProcessTests)
add_subdirectory(DataSamples)
add_subdirectory(UtilsTests)
add_subdirectory(DataPlotTests)

if(enableTestCL)
    add_subdirectory(VolumeRenderTests)
endif()
