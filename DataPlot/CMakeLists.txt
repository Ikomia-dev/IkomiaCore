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

target_compile_definitions(ikDataPlot PRIVATE
    DATAPLOT_LIBRARY
    QWT_DLL
)

target_compile_features(ikDataPlot PRIVATE cxx_std_14)

target_include_directories(ikDataPlot PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../Utils
    ${CMAKE_CURRENT_LIST_DIR}/../../qwt/src
)

target_link_directories(ikDataPlot PRIVATE
    # Qwt
    ${CMAKE_CURRENT_LIST_DIR}/../../qwt/build/lib
)

target_link_libraries(ikDataPlot PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Python3::Python
    Boost::python${PYTHON_VERSION_NO_DOT}
    opencv_core
    qwt
    ikUtils
)

# Translations
qt5_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/dataplot_fr.ts)

install(TARGETS ikDataPlot
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
