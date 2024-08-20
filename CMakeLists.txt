cmake_minimum_required(VERSION 3.16)
set(IKOMIA_VERSION 0.11.1)
project(IkomiaCore VERSION ${IKOMIA_VERSION} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/Build)
set(CMAKE_INSTALL_IKOMIA_LIB ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_IKOMIA_INCLUDE ${CMAKE_INSTALL_PREFIX}/include)
set(CMAKE_INSTALL_IKOMIA_PYTHON_LIB ${CMAKE_CURRENT_LIST_DIR}/../IkomiaApi/ikomia)

# Set up AUTOMOC and some sensible defaults for runtime execution
# When using Qt 6.3, you can replace the code block below with
# qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
include(GNUInstallDirs)
include(Versions.cmake)
include(LocalSettings.cmake)

# Qt
find_package(Qt5 REQUIRED COMPONENTS Core Concurrent Gui Network OpenGL Sql Test Widgets LinguistTools)

# Python
if(CENTOS7)
    set(Python3_DIR "/work/shared/local")
endif()

find_package(Python3 ${PYTHON_VERSION_DOT} EXACT REQUIRED COMPONENTS Interpreter Development NumPy)

# OpenMP
find_package(OpenMP)

# OpenCV
if(WIN32)
    if(${IK_CPU})
        set(OpenCV_DIR "C:/Program Files/OpenCV/cpu/x64/vc${MSVC_VERSION}/lib")
    else()
        set(OpenCV_DIR "C:/Program Files/OpenCV/cuda/x64/vc${MSVC_VERSION}/lib")
    endif()
endif()

find_package(OpenCV ${OPENCV_VERSION} REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

if(CENTOS7)
    set(BUILD_SHARED_LIBS ON)
    set(BOOST_PYTHON_TARGET Boost::python)
    find_package(Boost REQUIRED COMPONENTS system filesystem python)
else()
    set(BOOST_PYTHON_TARGET Boost::python${PYTHON_VERSION_NO_DOT})
    find_package(Boost REQUIRED COMPONENTS system filesystem python${PYTHON_VERSION_NO_DOT})
endif()

# VTK
find_package(GDCM ${GDCM_VERSION} REQUIRED)
find_package(VTK ${VTK_MAJOR_MINOR_VERSION} REQUIRED)

include(Common.cmake)

add_subdirectory(Utils)
add_subdirectory(Core)
add_subdirectory(DataIO)
add_subdirectory(DataManagement)
add_subdirectory(DataProcess)
add_subdirectory(DataPlot)

if(NOT ${IK_CPU})
    add_subdirectory(VolumeRender)
endif()

add_subdirectory(PyDataIO)
add_subdirectory(PyCore)
add_subdirectory(PyUtils)
add_subdirectory(PyDataProcess)
add_subdirectory(UnitTests)



###############################################
##                                           ##
##     Generation of the package's files     ##
##                                           ##
###############################################

include(CMakePackageConfigHelpers)

set(IKOMIA_CORE_DIR
    ${CMAKE_CURRENT_LIST_DIR}
    CACHE PATH "Location of root directory"
)

set(IkomiaCore_UNDERSCORE_VERSION
    ${IkomiaCore_VERSION_MAJOR}_${IkomiaCore_VERSION_MINOR}_${IkomiaCore_VERSION_PATCH}
    CACHE STRING "Version of the project where dots are replaced by underscore (X.X.X -> 'X_X_X')"
)

configure_package_config_file(
    IkomiaCoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/IkomiaCoreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IkomiaCore
    PATH_VARS IKOMIA_CORE_DIR CMAKE_INSTALL_PREFIX CMAKE_INSTALL_IKOMIA_INCLUDE CMAKE_INSTALL_IKOMIA_LIB
)

write_basic_package_version_file(
    IkomiaCoreConfigVersion.cmake
    VERSION ${IKOMIA_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/IkomiaCoreConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/IkomiaCoreConfigVersion.cmake
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/IkomiaCore
)
