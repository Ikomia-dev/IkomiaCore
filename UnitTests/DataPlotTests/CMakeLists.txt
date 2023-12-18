add_executable(CDataPlotTests WIN32
    CDataPlotTests.cpp
    CDataPlotTests.h
)

target_include_directories(CDataPlotTests PRIVATE
    ../../../qwt/src
    ../../DataPlot
    ../../Utils
)

target_compile_features(CDataPlotTests PRIVATE cxx_std_14)

target_link_libraries(CDataPlotTests PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Widgets
    Qt::Test
    OpenMP::OpenMP_CXX
    Python3::Python
    ${BOOST_PYTHON_TARGET}
    opencv_core
    ikUtils
    ikDataPlot
    qwt
)

install(TARGETS CDataPlotTests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
