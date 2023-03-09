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

target_compile_definitions(ikDataIO PRIVATE
    DATAIO_LIBRARY
)

target_compile_features(ikDataIO PRIVATE cxx_std_14)

target_include_directories(ikDataIO PRIVATE
    ../Core
    ../Utils
)

target_link_libraries(ikDataIO PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
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

if(UNIX)
    target_link_libraries(ikDataIO PRIVATE
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
endif()

if(WIN32)
    target_link_libraries(ikDataIO PRIVATE
        vtkCommonCore
        vtkCommonDataModel
        vtkCommonExecutionModel
        vtkCommonMath
        vtkCommonSystem
        vtkCommonTransforms
        vtkFiltersCore
        vtkFiltersGeneral
        vtkFiltersImaging
        vtkImagingColor
        vtkImagingCore
        vtkImagingGeneral
        vtkImagingMath
        vtkImagingSources
        vtkInteractionImage
        vtkInteractionStyle
        vtkIOCore
        vtkIOImage
        vtkRenderingContext2D
        vtkRenderingCore
        vtkRenderingImage
        vtkRenderingOpenGL2
        vtksys
        vtkViewsCore
        vtkDICOM
    )
endif()

# Translations
qt5_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/utils_fr.ts)

install(TARGETS ikDataIO
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
