add_executable(DataInfoTests
    CDataInfoTests.cpp
    CDataInfoTests.h
)

target_include_directories(DataInfoTests PRIVATE
    ../../../Core
    ../../../Utils
)

target_compile_features(DataInfoTests PRIVATE cxx_std_14)

target_link_libraries(DataInfoTests PRIVATE
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

install(TARGETS DataInfoTests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
