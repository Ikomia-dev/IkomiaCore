add_library(pycore SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp ../Core/Data/CvMatNumpyArrayConverter.h
    CGraphicsItemWrap.cpp CGraphicsItemWrap.h
    CTaskIOFactoryWrap.cpp CTaskIOFactoryWrap.h
    CWorkflowTaskIOWrap.cpp CWorkflowTaskIOWrap.h
    CWorkflowTaskParamWrap.cpp CWorkflowTaskParamWrap.h
    CWorkflowTaskWidgetWrap.cpp CWorkflowTaskWidgetWrap.h
    CWorkflowTaskWrap.cpp CWorkflowTaskWrap.h
    PyCore.cpp PyCore.h
    PyCoreDocString.hpp
    PyCoreGlobal.h
    PyCoreTools.hpp
)

set_target_properties(pycore PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_include_directories(pycore PRIVATE
    ../Core
    ../Utils
)

target_compile_definitions(pycore PRIVATE
    PYCORE_LIBRARY
)

target_link_libraries(pycore PRIVATE
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
    target_link_libraries(pycore PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgcodecs${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgcodecs${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(pycore PUBLIC
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
    )
endif()

if(WIN32)
    target_compile_options(pycore
        -bigobj
    )
endif()

install(TARGETS pycore
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
