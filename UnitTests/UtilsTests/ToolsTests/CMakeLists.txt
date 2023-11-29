add_executable(ToolsTests
    CToolsTests.cpp
    CToolsTests.h
)

target_include_directories(ToolsTests PRIVATE
    ../../../Utils
)

target_compile_features(ToolsTests PRIVATE cxx_std_14)

target_link_libraries(ToolsTests PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Test
    Qt::Widgets
    Python3::Python
    Boost::system
    Boost::filesystem
    ${BOOST_PYTHON_TARGET}
    OpenMP::OpenMP_CXX
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_videoio
    ikUtils
    ikCore
)

install(TARGETS ToolsTests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
