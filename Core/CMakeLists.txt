add_library(ikCore SHARED
    Data/CDataArrayInfo.cpp
    Data/CDataArrayInfo.h
    Data/CDataConversion.cpp
    Data/CDataConversion.h
    Data/CDataDicomInfo.cpp
    Data/CDataDicomInfo.h
    Data/CDataImageInfo.cpp
    Data/CDataImageInfo.h
    Data/CDataInfo.cpp
    Data/CDataInfo.h
    Data/CDataSubset.hpp
    Data/CDataVideoInfo.cpp
    Data/CDataVideoInfo.h
    Data/CDataset.hpp
    Data/CDatasetInfo.cpp
    Data/CDatasetInfo.h
    Data/CMat.hpp
    Data/CMeasure.cpp
    Data/CMeasure.h
    Data/CndArray.hpp
    Graphics/CGraphicsComplexPolygon.cpp
    Graphics/CGraphicsComplexPolygon.h
    Graphics/CGraphicsContext.cpp
    Graphics/CGraphicsContext.h
    Graphics/CGraphicsConversion.cpp
    Graphics/CGraphicsConversion.h
    Graphics/CGraphicsEllipse.cpp
    Graphics/CGraphicsEllipse.h
    Graphics/CGraphicsItem.hpp
    Graphics/CGraphicsJSON.cpp
    Graphics/CGraphicsJSON.h
    Graphics/CGraphicsLayer.cpp
    Graphics/CGraphicsLayer.h
    Graphics/CGraphicsPoint.cpp
    Graphics/CGraphicsPoint.h
    Graphics/CGraphicsPolygon.cpp
    Graphics/CGraphicsPolygon.h
    Graphics/CGraphicsPolyline.cpp
    Graphics/CGraphicsPolyline.h
    Graphics/CGraphicsRectangle.cpp
    Graphics/CGraphicsRectangle.h
    Graphics/CGraphicsRegistration.cpp
    Graphics/CGraphicsRegistration.h
    Graphics/CGraphicsText.cpp
    Graphics/CGraphicsText.h
    Graphics/CPoint.hpp
    Graphics/GraphicsProperty.cpp
    Graphics/GraphicsProperty.h
    Main/CoreDefine.hpp
    Main/CoreGlobal.hpp
    Main/CoreTools.hpp
    Main/forwards.hpp
    Workflow/CViewPropertyIO.cpp
    Workflow/CViewPropertyIO.h
    Workflow/CWorkflowEdge.hpp
    Workflow/CWorkflowTask.cpp
    Workflow/CWorkflowTask.h
    Workflow/CWorkflowTaskIO.cpp
    Workflow/CWorkflowTaskIO.h
    Workflow/CWorkflowTaskParam.cpp
    Workflow/CWorkflowTaskParam.h
    Workflow/CWorkflowTaskWidget.cpp
    Workflow/CWorkflowTaskWidget.h
)

set_target_properties(ikCore PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

set(PUBLIC_HEADERS_DATA
    Data/CDataConversion.h
    Data/CDataImageInfo.h
    Data/CDataset.hpp
    Data/CDataInfo.h
    Data/CDatasetInfo.h
    Data/CDataSubset.hpp
    Data/CMat.hpp
    Data/CndArray.hpp
    Data/CMeasure.h
    Data/CDataVideoInfo.h
)

set(PUBLIC_HEADERS_GRAPHICS
    Graphics/CGraphicsLayer.h
    Graphics/CGraphicsItem.hpp
    Graphics/CGraphicsJSON.h
    Graphics/CGraphicsEllipse.h
    Graphics/CGraphicsPolygon.h
    Graphics/CGraphicsComplexPolygon.h
    Graphics/CGraphicsPolyline.h
    Graphics/CGraphicsRectangle.h
    Graphics/CGraphicsText.h
    Graphics/CGraphicsPoint.h
    Graphics/CGraphicsConversion.h
    Graphics/CGraphicsContext.h
    Graphics/CPoint.hpp
    Graphics/GraphicsProperty.h
)

set(PUBLIC_HEADERS_WORKFLOW
    Workflow/CWorkflowEdge.hpp
    Workflow/CWorkflowTaskParam.h
    Workflow/CWorkflowTaskWidget.h
    Workflow/CWorkflowTask.h
    Workflow/CWorkflowTaskIO.h
    Workflow/CViewPropertyIO.h
)

set(PUBLIC_HEADERS_MAIN
    Main/CoreDefine.hpp
    Main/CoreTools.hpp
    Main/CoreGlobal.hpp
)

target_include_directories(ikCore PUBLIC
    ../Utils
)

target_compile_definitions(ikCore PUBLIC
    CORE_LIBRARY
)

target_link_libraries(ikCore PUBLIC
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
    avcodec
)

if(WIN32)
    target_link_libraries(ikCore PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgcodecs${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgcodecs${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_videoio${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_videoio${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikCore PUBLIC
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_videoio
    )
endif()

if(WIN32)
    target_link_libraries(ikCore PUBLIC
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
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../../quazip/build/lib
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../../qwt/build/lib
        # Remove: L'C:\Program
#        (
#        )
#        Files
#        Kits\10\Lib\10.0.19041.0\um\x64'
#        \Windows
#        x86
    )
endif()

if(UNIX AND centos7 AND NOT MACOS)
    target_link_libraries(ikCore PUBLIC
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(ikCore PUBLIC
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(ikCore PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(ikCore PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA
#        Computing
#        GPU
#        Toolkit/CUDA/v/bin'
#        Toolkit/CUDA/v/lib/x64'
    )
endif()

install(TARGETS ikCore
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)

install(FILES ${PUBLIC_HEADERS_DATA}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Core/Data
)

install(FILES ${PUBLIC_HEADERS_GRAPHICS}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Core/Graphics
)

install(FILES ${PUBLIC_HEADERS_WORKFLOW}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Core/Workflow
)

install(FILES ${PUBLIC_HEADERS_MAIN}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Core/Main
)
