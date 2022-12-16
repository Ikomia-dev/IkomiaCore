add_library(ikDataPlot SHARED
    CCurvePlot.cpp
    CCurvePlot.h
    CDataPlot.cpp
    CDataPlot.h
    CHistogramPlot.cpp
    CHistogramPlot.h
    CMultiBarPlot.cpp
    CMultiBarPlot.h
    CPieMarker.cpp
    CPieMarker.h
    CPiePlot.cpp
    CPiePlot.h
    CPlot.cpp
    CPlot.h
    CSimpleBarPlot.cpp
    CSimpleBarPlot.h
    DataPlotGlobal.h
)

set_target_properties(ikDataPlot PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_include_directories(ikDataPlot PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../Utils
    ${CMAKE_CURRENT_LIST_DIR}/../../qwt/src
)

target_compile_definitions(ikDataPlot PUBLIC
    DATAPLOT_LIBRARY
    QWT_DLL
)

target_link_libraries(ikDataPlot PUBLIC
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
)

if(WIN32)
    target_link_libraries(ikDataPlot PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikDataPlot PUBLIC
        opencv_core
    )
endif()

install(TARGETS ikDataPlot
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
