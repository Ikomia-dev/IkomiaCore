add_library(pydataio SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp ../Core/Data/CvMatNumpyArrayConverter.h
    PyDataIO.cpp PyDataIO.h
    PyDataIODocString.hpp
    PyDataIOGlobal.hpp
)

set_target_properties(pydataio PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_include_directories(pydataio PRIVATE
    ../Core
    ../DataIO
    ../Utils
)

target_compile_definitions(pydataio PRIVATE
    PYDATAIO_LIBRARY
)

target_link_libraries(pydataio PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
)

if(WIN32)
    target_link_libraries(pydataio PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgcodecs${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgcodecs${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(pydataio PUBLIC
        opencv_core
        opencv_imgcodecs
    )
endif()

install(TARGETS pydataio
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
