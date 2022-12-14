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
    target_link_libraries(ikUtils PUBLIC
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
    target_link_libraries(ikUtils PUBLIC
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
    )
endif()

if(WIN32)
    target_link_libraries(pycore PRIVATE
        # Remove: L$ENV{ProgramW6432}/Boost/lib
        # Remove: L$ENV{ProgramW6432}/GDCM/bin
        # Remove: L$ENV{ProgramW6432}/OpenSSL-Win64/bin
        # Remove: L$ENV{ProgramW6432}/Python38/
        # Remove: L$ENV{ProgramW6432}/Python38/libs
        # Remove: L$ENV{ProgramW6432}/VTK/bin
        # Remove: L$ENV{ProgramW6432}/VTK/lib
        # Remove: L$ENV{ProgramW6432}/gmic/bin/
        # Remove: L$ENV{ProgramW6432}/gmic/lib/
        # Remove: L$ENV{ProgramW6432}/intel_tbb/bin/intel64/vc14
        # Remove: L$ENV{ProgramW6432}/leptonica/bin
        # Remove: L$ENV{ProgramW6432}/tesseract/bin
        # Remove: L$ENV{ProgramW6432}/zlib/bin
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../quazip/build/lib
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../qwt/build/lib
        # Remove: L'C:\Program
#        (
#        )
#        Files
#        Kits\10\Lib\10.0.19041.0\um\x64'
#        \Windows
#        x86
    )

    target_compile_options(pycore
        -bigobj
    )
endif()

if(UNIX AND centos7 AND NOT MACOS)
    target_link_libraries(pycore PRIVATE
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(pycore PRIVATE
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(pycore PRIVATE
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(pycore PRIVATE
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA
#        Computing
#        GPU
#        Toolkit/CUDA/v/bin'
#        Toolkit/CUDA/v/lib/x64'
    )
endif()

install(TARGETS pycore
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)