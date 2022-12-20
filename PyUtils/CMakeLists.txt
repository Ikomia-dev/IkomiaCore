set(CMAKE_INSTALL_IKOMIA_PYUTILS ${CMAKE_INSTALL_IKOMIA_PYTHON_LIB}/utils)

add_library(pyutils SHARED
    PairToPythonTupleConverter.hpp
    PyUtils.cpp
    PyUtils.h
    PyUtilsDocString.hpp
    PyUtilsGlobal.h
)

set_target_properties(pyutils PROPERTIES
    VERSION ${IKOMIA_VERSION}
    SOVERSION ${IKOMIA_VERSION}
    PREFIX ""
)

target_compile_definitions(pyutils PRIVATE
    PYUTILS_LIBRARY
)

target_compile_features(pyutils PRIVATE cxx_std_14)

target_include_directories(pyutils PRIVATE
    ../Utils
)

target_link_libraries(pyutils PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
    ikUtils
)

if(WIN32)
    target_link_libraries(pyutils PRIVATE
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(pyutils PRIVATE
        opencv_core
    )
endif()

install(TARGETS pyutils
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_PYUTILS}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_PYUTILS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_PYUTILS}
)
