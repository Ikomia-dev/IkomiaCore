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

set_target_properties(ikVolumeRender PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

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
    target_link_libraries(ikVolumeRender PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikVolumeRender PUBLIC
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

install(TARGETS ikVolumeRender
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
