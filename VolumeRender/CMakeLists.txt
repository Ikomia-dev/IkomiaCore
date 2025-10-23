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

target_link_directories(ikVolumeRender PRIVATE
    # OpenCL
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/lib/x64"
)

target_link_libraries(ikVolumeRender PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::OpenGL
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
    Python3::Python
    ${BOOST_PYTHON_TARGET}
    opencv_core
    opencv_highgui
    opencv_imgproc
    OpenCL
    ikUtils
    ikCore
)

if(WIN32)
    target_link_libraries(ikVolumeRender PRIVATE
        OpenGL32
    )
endif()

# Translations
qt6_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/volumerender_fr.ts)

# Resources
set(RESOURCES "")
qt6_add_resources(RESOURCES VolumeRender.qrc)
target_sources(ikVolumeRender PRIVATE ${RESOURCES})

install(TARGETS ikVolumeRender
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
