set(CMAKE_INSTALL_IKOMIA_PYDATAIO ${CMAKE_INSTALL_IKOMIA_PYTHON_LIB}/dataio)

add_library(pydataio SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp
    ../Core/Data/CvMatNumpyArrayConverter.h
    PyDataIO.cpp
    PyDataIO.h
    PyDataIODocString.hpp
    PyDataIOGlobal.hpp
)

set_target_properties(pydataio PROPERTIES
    VERSION ${IKOMIA_VERSION}
    SOVERSION ${IKOMIA_VERSION}
    PREFIX ""
)

if (WIN32)
    set_target_properties(pydataio PROPERTIES SUFFIX ".pyd")
endif()

target_compile_definitions(pydataio PRIVATE
    PYDATAIO_LIBRARY
)

target_compile_features(pydataio PRIVATE cxx_std_14)

target_include_directories(pydataio PRIVATE
    ../Core
    ../DataIO
    ../Utils
)

target_link_libraries(pydataio PRIVATE
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
    ikUtils
    ikCore
    ikDataIO
)

install(TARGETS pydataio
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_PYDATAIO}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_PYDATAIO}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_PYDATAIO}
)
