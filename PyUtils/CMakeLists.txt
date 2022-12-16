add_library(pyutils SHARED
    PairToPythonTupleConverter.hpp
    PyUtils.cpp PyUtils.h
    PyUtilsDocString.hpp
    PyUtilsGlobal.h
)

set_target_properties(pyutils PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_include_directories(pyutils PRIVATE
    ../Utils
)

target_compile_definitions(pyutils PRIVATE
    PYUTILS_LIBRARY
)

target_link_libraries(pyutils PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
)

if(WIN32)
    target_link_libraries(pyutils PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(pyutils PUBLIC
        opencv_core
    )
endif()

install(TARGETS pyutils
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
