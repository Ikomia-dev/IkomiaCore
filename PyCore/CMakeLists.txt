set(CMAKE_INSTALL_IKOMIA_PYCORE ${CMAKE_INSTALL_IKOMIA_PYTHON_LIB}/core)

add_library(pycore SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp
    ../Core/Data/CvMatNumpyArrayConverter.h
    CGraphicsItemWrap.cpp
    CGraphicsItemWrap.h
    CTaskIOFactoryWrap.cpp
    CTaskIOFactoryWrap.h
    CWorkflowTaskIOWrap.cpp
    CWorkflowTaskIOWrap.h
    CWorkflowTaskParamWrap.cpp
    CWorkflowTaskParamWrap.h
    CWorkflowTaskWidgetWrap.cpp
    CWorkflowTaskWidgetWrap.h
    CWorkflowTaskWrap.cpp
    CWorkflowTaskWrap.h
    IterableConverter.hpp
    MapConverter.hpp
    PyCore.cpp
    PyCore.h
    PyCoreDocString.hpp
    PyCoreGlobal.h
    PyCoreTools.hpp
    VectorConverter.hpp
)

set_target_properties(pycore PROPERTIES
    VERSION ${IKOMIA_VERSION}
    SOVERSION ${IKOMIA_VERSION}
    PREFIX ""
)

if (WIN32)
    set_target_properties(pycore PROPERTIES SUFFIX ".pyd")
endif()

target_compile_definitions(pycore PRIVATE
    PYCORE_LIBRARY
)

target_compile_features(pycore PRIVATE cxx_std_14)

target_include_directories(pycore PRIVATE
    ../Core
    ../Utils
)

target_link_libraries(pycore PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Python3::Python
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui
    ikUtils
    ikCore
)

if(WIN32)
    target_compile_options(pycore PRIVATE
        /bigobj
    )
endif()

install(TARGETS pycore
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_PYCORE}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_PYCORE}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_PYCORE}
)
