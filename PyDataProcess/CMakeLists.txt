add_library(pydataprocess SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp ../Core/Data/CvMatNumpyArrayConverter.h
    CIkomiaRegistryWrap.cpp CIkomiaRegistryWrap.h
    CPluginProcessInterfaceWrap.cpp CPluginProcessInterfaceWrap.h
    CWidgetFactoryWrap.cpp CWidgetFactoryWrap.h
    CWorkflowWrap.cpp CWorkflowWrap.h
    IO/CArrayIOWrap.cpp IO/CArrayIOWrap.h
    IO/CDatasetIOWrap.cpp IO/CDatasetIOWrap.h
    IO/CGraphicsInputWrap.cpp IO/CGraphicsInputWrap.h
    IO/CGraphicsOutputWrap.cpp IO/CGraphicsOutputWrap.h
    IO/CImageIOWrap.cpp IO/CImageIOWrap.h
    IO/CInstanceSegIOWrap.cpp IO/CInstanceSegIOWrap.h
    IO/CNumericIOWrap.hpp
    IO/CObjectDetectionIOWrap.cpp IO/CObjectDetectionIOWrap.h
    IO/CPathIOWrap.cpp IO/CPathIOWrap.h
    IO/CSemanticSegIOWrap.cpp IO/CSemanticSegIOWrap.h
    IO/CVideoIOWrap.cpp IO/CVideoIOWrap.h
    IO/CWidgetOutputWrap.cpp IO/CWidgetOutputWrap.h
    PyDataProcess.cpp PyDataProcess.h
    PyDataProcessDocString.hpp
    PyDataProcessGlobal.h
    PyDataProcessTools.hpp
    Task/C2dImageInteractiveTaskWrap.cpp Task/C2dImageInteractiveTaskWrap.h
    Task/C2dImageTaskWrap.cpp Task/C2dImageTaskWrap.h
    Task/CDnnTrainTaskWrap.cpp Task/CDnnTrainTaskWrap.h
    Task/CTaskFactoryWrap.cpp Task/CTaskFactoryWrap.h
    Task/CVideoOFTaskWrap.cpp Task/CVideoOFTaskWrap.h
    Task/CVideoTaskWrap.cpp Task/CVideoTaskWrap.h
    Task/CVideoTrackingTaskWrap.cpp Task/CVideoTrackingTaskWrap.h
)

target_include_directories(pydataprocess PRIVATE
    ../Core
    ../DataIO
    ../DataProcess
    ../Utils
)

target_compile_definitions(pydataprocess PRIVATE
    PYDATAPROCESS_LIBRARY
)

target_link_libraries(pydataprocess PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
    gmic
)

if(WIN32)
    target_link_directories(pydataprocess PUBLIC
        # GMIC
        $ENV{ProgramW6432}/gmic/bin
        $ENV{ProgramW6432}/gmic/lib
    )

    target_link_libraries(ikUtils PUBLIC
        $<$<CONFIG:RELEASE>:opencv_bgsegm${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_bgsegm${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_bioinspired${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_bioinspired${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_dnn${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_dnn${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_face${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_face${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_features2d${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_features2d${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_fuzzy${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_fuzzy${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_hfs${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_hfs${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_objdetect${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_objdetect${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_optflow${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_optflow${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_photo${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_photo${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_saliency${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_saliency${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_superres${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_superres${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_text${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_text${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_tracking${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_tracking${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_video${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_video${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_ximgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_ximgproc${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_xphoto${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_xphoto${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_directories(pydataprocess PUBLIC
        # GMIC
        ${CMAKE_CURRENT_LIST_DIR}/../../gmic/build
    )

    target_link_libraries(ikUtils PUBLIC
        opencv_bgsegm
        opencv_bioinspired
        opencv_core
        opencv_dnn
        opencv_face
        opencv_features2d
        opencv_fuzzy
        opencv_hfs
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        opencv_objdetect
        opencv_optflow
        opencv_photo
        opencv_saliency
        opencv_superres
        opencv_text
        opencv_tracking
        opencv_video
        opencv_ximgproc
        opencv_xphoto
    )
endif()

if(WIN32 AND NOT IK_CPU)
    target_link_libraries(pydataprocess PUBLIC
        $<$<CONFIG:RELEASE>:opencv_cudawarping${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_cudawarping${OPENCV_VERSION}d>
    )
endif()

if(UNIX AND NOT IK_CPU)
    target_link_libraries(pydataprocess PUBLIC
        opencv_cudawarping
    )
endif()

if(WIN32)
    target_link_libraries(pydataprocess PRIVATE
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

    target_compile_options(pydataprocess
        -bigobj
    )
endif()

if(UNIX AND centos7 AND NOT MACOS)
    target_link_libraries(pydataprocess PRIVATE
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(pydataprocess PRIVATE
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(pydataprocess PRIVATE
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(pydataprocess PRIVATE
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA
#        Computing
#        GPU
#        Toolkit/CUDA/v/bin'
#        Toolkit/CUDA/v/lib/x64'
    )
endif()

install(TARGETS pydataprocess
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
