set(CMAKE_INSTALL_IKOMIA_PYUTILS ${CMAKE_INSTALL_IKOMIA_PYTHON_LIB}/utils)

add_library(pyutils SHARED
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

if (WIN32)
    set_target_properties(pyutils PROPERTIES SUFFIX ".pyd")
endif()

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
    Python3::Python
    Boost::system
    Boost::filesystem
    ${BOOST_PYTHON_TARGET}
    opencv_core
    ikUtils
)

install(TARGETS pyutils
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_PYUTILS}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_PYUTILS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_PYUTILS}
)
