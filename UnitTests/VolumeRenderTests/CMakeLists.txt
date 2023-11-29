add_executable(volumeRenderTests WIN32
    COpenCLTests.cpp
    COpenCLTests.h
    kernels.hpp
)

target_include_directories(volumeRenderTests PRIVATE
    ../../Core
    ../../Utils
    ../../VolumeRender
)

target_compile_features(DataInfoTests PRIVATE cxx_std_14)

target_link_libraries(volumeRenderTests PRIVATE
    Qt::Concurrent
    Qt::Core
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
    ikVolumeRender
)

install(TARGETS volumeRenderTests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
