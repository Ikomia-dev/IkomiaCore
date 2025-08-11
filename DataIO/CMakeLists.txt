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
        # GDCM
        gdcmMSFF
        gdcmDICT
        gdcmIOD
        gdcmDSED
        gdcmCommon
        gdcmjpeg8
        gdcmjpeg12
        gdcmjpeg16
        gdcmopenjp2
        gdcmcharls
        gdcmuuid
        gdcmexpat
        # VTK
        vtkCommonColor-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonComputationalGeometry-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonCore-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonDataModel-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonExecutionModel-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonMath-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonMisc-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonSystem-${VTK_MAJOR_MINOR_VERSION}
        vtkCommonTransforms-${VTK_MAJOR_MINOR_VERSION}
        vtkDICOM-${VTK_MAJOR_MINOR_VERSION}
        vtkDICOMParser-${VTK_MAJOR_MINOR_VERSION}
        vtkexpat-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersCore-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersExtraction-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersGeneral-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersGeometry-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersHybrid-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersImaging-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersModeling-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersSources-${VTK_MAJOR_MINOR_VERSION}
        vtkFiltersStatistics-${VTK_MAJOR_MINOR_VERSION}
        vtkfreetype-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingColor-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingCore-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingFourier-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingGeneral-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingHybrid-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingMath-${VTK_MAJOR_MINOR_VERSION}
        vtkImagingSources-${VTK_MAJOR_MINOR_VERSION}
        vtkInteractionImage-${VTK_MAJOR_MINOR_VERSION}
        vtkInteractionStyle-${VTK_MAJOR_MINOR_VERSION}
        vtkInteractionWidgets-${VTK_MAJOR_MINOR_VERSION}
        vtkIOCore-${VTK_MAJOR_MINOR_VERSION}
        vtkIOImage-${VTK_MAJOR_MINOR_VERSION}
        vtkIOSQL-${VTK_MAJOR_MINOR_VERSION}
        vtkIOXML-${VTK_MAJOR_MINOR_VERSION}
        vtkIOXMLParser-${VTK_MAJOR_MINOR_VERSION}
        vtklz4-${VTK_MAJOR_MINOR_VERSION}
        vtkmetaio-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingAnnotation-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingContext2D-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingCore-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingFreeType-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingImage-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingOpenGL2-${VTK_MAJOR_MINOR_VERSION}
        vtkRenderingVolume-${VTK_MAJOR_MINOR_VERSION}
        vtksqlite-${VTK_MAJOR_MINOR_VERSION}
        vtksys-${VTK_MAJOR_MINOR_VERSION}
        vtkViewsCore-${VTK_MAJOR_MINOR_VERSION}
    )
endif()

if(WIN32)
    target_link_libraries(ikDataIO PRIVATE
        # GDCM
        gdcmMSFF
        gdcmDICT
        gdcmIOD
        gdcmDSED
        gdcmCommon
        gdcmjpeg8
        gdcmjpeg12
        gdcmjpeg16
        gdcmopenjp2
        gdcmcharls
        gdcmexpat
        # VTK
        vtkalglib
        vtkCommonColor
        vtkCommonComputationalGeometry
        vtkCommonCore
        vtkCommonDataModel
        vtkCommonExecutionModel
        vtkCommonMath
        vtkCommonMisc
        vtkCommonSystem
        vtkCommonTransforms
        vtkDICOM
        vtkDICOMParser
        vtkexpat
        vtkFiltersCore
        vtkFiltersExtraction
        vtkFiltersGeneral
        vtkFiltersGeometry
        vtkFiltersHybrid
        vtkFiltersImaging
        vtkFiltersModeling
        vtkFiltersSources
        vtkFiltersStatistics
        vtkfreetype
        vtkglew
        vtkImagingColor
        vtkImagingCore
        vtkImagingFourier
        vtkImagingGeneral
        vtkImagingHybrid
        vtkImagingMath
        vtkImagingSources
        vtkInteractionImage
        vtkInteractionStyle
        vtkInteractionWidgets
        vtkIOCore
        vtkIOImage
        vtkIOSQL
        vtkIOXML
        vtkIOXMLParser
        vtklz4
        vtkmetaio
        vtkRenderingAnnotation
        vtkRenderingContext2D
        vtkRenderingCore
        vtkRenderingFreeType
        vtkRenderingImage
        vtkRenderingOpenGL2
        vtkRenderingVolume
        vtksqlite
        vtksys
        vtkViewsCore
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
