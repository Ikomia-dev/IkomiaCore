add_library(ikDataIO SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp
    ../Core/Data/CvMatNumpyArrayConverter.h
    CDataIO.hpp
    CDataImageIO.cpp
    CDataImageIO.h
    CDataVideoBuffer.cpp
    CDataVideoBuffer.h
    CDataVideoIO.cpp
    CDataVideoIO.h
    CDicomImageIO.cpp
    CDicomImageIO.h
    CNumpyImageIO.cpp
    CNumpyImageIO.h
    COpencvImageIO.cpp
    COpencvImageIO.h
    COpencvVideoIO.cpp
    COpencvVideoIO.h
    CVirtualImageIO.cpp
    CVirtualImageIO.h
    CVirtualVideoIO.cpp
    CVirtualVideoIO.h
    DataIOGlobal.hpp
    DataIOTools.hpp
)

set_target_properties(ikDataIO PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_include_directories(ikDataIO PUBLIC
    ../Core
    ../Utils
)

target_compile_definitions(ikDataIO PUBLIC
    DATAIO_LIBRARY
)

target_link_libraries(ikDataIO PUBLIC
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
    vtkCommonCore-${VTK_MAJOR_MINOR_VERSION}
    vtkCommonDataModel-${VTK_MAJOR_MINOR_VERSION}
    vtkCommonExecutionModel-${VTK_MAJOR_MINOR_VERSION}
    vtkCommonMath-${VTK_MAJOR_MINOR_VERSION}
    vtkCommonSystem-${VTK_MAJOR_MINOR_VERSION}
    vtkCommonTransforms-${VTK_MAJOR_MINOR_VERSION}
    vtkFiltersCore-${VTK_MAJOR_MINOR_VERSION}
    vtkFiltersGeneral-${VTK_MAJOR_MINOR_VERSION}
    vtkFiltersImaging-${VTK_MAJOR_MINOR_VERSION}
    vtkImagingColor-${VTK_MAJOR_MINOR_VERSION}
    vtkImagingCore-${VTK_MAJOR_MINOR_VERSION}
    vtkImagingGeneral-${VTK_MAJOR_MINOR_VERSION}
    vtkImagingMath-${VTK_MAJOR_MINOR_VERSION}
    vtkImagingSources-${VTK_MAJOR_MINOR_VERSION}
    vtkInteractionImage-${VTK_MAJOR_MINOR_VERSION}
    vtkInteractionStyle-${VTK_MAJOR_MINOR_VERSION}
    vtkIOCore-${VTK_MAJOR_MINOR_VERSION}
    vtkIOImage-${VTK_MAJOR_MINOR_VERSION}
    vtkRenderingContext2D-${VTK_MAJOR_MINOR_VERSION}
    vtkRenderingCore-${VTK_MAJOR_MINOR_VERSION}
    vtkRenderingImage-${VTK_MAJOR_MINOR_VERSION}
    vtkRenderingOpenGL2-${VTK_MAJOR_MINOR_VERSION}
    vtksys-${VTK_MAJOR_MINOR_VERSION}
    vtkViewsCore-${VTK_MAJOR_MINOR_VERSION}
    vtkDICOM-${VTK_MAJOR_MINOR_VERSION}
)

if(WIN32)
    target_link_libraries(ikDataIO PUBLIC
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
    target_link_libraries(ikDataIO PUBLIC
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_videoio
    )
endif()

install(TARGETS ikDataIO
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
