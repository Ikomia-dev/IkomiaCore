add_library(ikVolumeRender SHARED
    CAxisRender.cpp
    CAxisRender.h
    CCylinderRender.cpp
    CCylinderRender.h
    CSphereRender.cpp
    CSphereRender.h
    CTextureRender.cpp
    CTextureRender.h
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
    VolumeRender.cpp
    VolumeRender.h
    VolumeRenderDefine.h
    VolumeRenderGlobal.h
    ocl.hpp
    renderUtils.hpp
)

set_target_properties(ikVolumeRender PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_compile_definitions(ikVolumeRender PRIVATE
    VOLUMERENDER_LIBRARY
)

target_compile_features(ikVolumeRender PRIVATE cxx_std_14)

target_include_directories(ikVolumeRender PRIVATE
    ../Core
    ../Utils
)

target_link_libraries(ikVolumeRender PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::OpenGL
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
    OpenCL
    ikUtils
    ikCore
)

if(WIN32)
    target_link_libraries(ikVolumeRender PRIVATE
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
        OpenGL32
    )
endif()

if(UNIX)
    target_link_libraries(ikVolumeRender PRIVATE
        opencv_core
        opencv_highgui
        opencv_imgproc
    )
endif()

# Translations
qt5_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/volumerender_fr.ts)

# Resources
set(RESOURCES "")
qt5_add_resources(RESOURCES VolumeRender.qrc)
target_sources(ikVolumeRender PRIVATE ${RESOURCES})

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
