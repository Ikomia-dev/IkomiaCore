add_library(ikVolumeRender SHARED
    CAxisRender.cpp CAxisRender.h
    CCylinderRender.cpp CCylinderRender.h
    CSphereRender.cpp CSphereRender.h
    CTextureRender.cpp CTextureRender.h
    Kernels/raycastColor.h
    Kernels/raycastColorF2B.h
    Kernels/raycastColorIso.h
    Kernels/raycastColorMIP.h
    Kernels/raycastColorMinIP.h
    Kernels/raycastColorTools.h
    Kernels/raycastGray.h
    Kernels/raycastGrayF2B.h
    Kernels/raycastGrayIso.h
    Kernels/raycastGrayMIP.h
    Kernels/raycastGrayMinIP.h
    Kernels/raycastGrayTools.h
    VolumeRender.cpp VolumeRender.h
    VolumeRenderDefine.h
    VolumeRenderGlobal.h
    ocl.hpp
    renderUtils.hpp
)

target_include_directories(ikVolumeRender PUBLIC
    ../Core
    ../Utils
)

target_compile_definitions(ikVolumeRender PUBLIC
    VOLUMERENDER_LIBRARY
)

target_link_libraries(ikVolumeRender PUBLIC
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::OpenGL
    Qt::Sql
    Qt::Widgets
)

if(WIN32)
    target_link_libraries(ikUtils PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikUtils PUBLIC
        opencv_core
        opencv_highgui
        opencv_imgproc
    )
endif()

# Resources:
set(RESOURCES "")
qt5_add_resources(RESOURCES VolumeRender.qrc)
target_sources(ikVolumeRender PUBLIC ${RESOURCES})

#set(VolumeRender_resource_files
#    "Shaders/cylinderfshader.frag"
#    "Shaders/cylindervshader.vert"
#    "Shaders/spherefshader.frag"
#    "Shaders/spherevshader.vert"
#    "Shaders/texturefshader.frag"
#    "Shaders/texturevshader.vert"
#)

#qt_add_resources(ikVolumeRender "VolumeRender"
#    PREFIX
#        "/"
#    FILES
#        ${VolumeRender_resource_files}
#)

if(WIN32)
    target_link_libraries(ikVolumeRender PUBLIC
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
#        OpenCL
#        OpenGL32
#        \Windows
#        x86
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(ikVolumeRender PUBLIC
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
        OpenCL
    )
endif()

if(UNIX AND centos7 AND NOT MACOS)
    target_link_libraries(ikVolumeRender PUBLIC
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(ikVolumeRender PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(ikVolumeRender PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA
#        Computing
#        GPU
#        Toolkit/CUDA/v/bin'
#        Toolkit/CUDA/v/lib/x64'
    )
endif()

install(TARGETS ikVolumeRender
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
