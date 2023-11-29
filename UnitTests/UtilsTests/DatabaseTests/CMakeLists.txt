add_executable(UtilsTests
    CDatabaseTests.cpp
    CDatabaseTests.h
)

target_include_directories(UtilsTests PRIVATE
    ../../../Core
    ../../../Utils
)

target_compile_features(UtilsTests PRIVATE cxx_std_14)

target_link_libraries(UtilsTests PRIVATE
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

install(TARGETS UtilsTests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
