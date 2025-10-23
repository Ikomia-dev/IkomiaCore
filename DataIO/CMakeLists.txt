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
        # gdcmopenjp2
        gdcmcharls
        gdcmuuid
        gdcmexpat
        # VTK
        ${VTK_LIBRARIES}
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
        ${VTK_LIBRARIES}
        # vtkalglib
        # vtkCommonColor
        # vtkCommonComputationalGeometry
        # vtkCommonCore
        # vtkCommonDataModel
        # vtkCommonExecutionModel
        # vtkCommonMath
        # vtkCommonMisc
        # vtkCommonSystem
        # vtkCommonTransforms
        # vtkDICOM
        # vtkDICOMParser
        # vtkexpat
        # vtkFiltersCore
        # vtkFiltersExtraction
        # vtkFiltersGeneral
        # vtkFiltersGeometry
        # vtkFiltersHybrid
        # vtkFiltersImaging
        # vtkFiltersModeling
        # vtkFiltersSources
        # vtkFiltersStatistics
        # vtkfreetype
        # vtkglew
        # vtkImagingColor
        # vtkImagingCore
        # vtkImagingFourier
        # vtkImagingGeneral
        # vtkImagingHybrid
        # vtkImagingMath
        # vtkImagingSources
        # vtkInteractionImage
        # vtkInteractionStyle
        # vtkInteractionWidgets
        # vtkIOCore
        # vtkIOImage
        # vtkIOSQL
        # vtkIOXML
        # vtkIOXMLParser
        # vtklz4
        # vtkmetaio
        # vtkRenderingAnnotation
        # vtkRenderingContext2D
        # vtkRenderingCore
        # vtkRenderingFreeType
        # vtkRenderingImage
        # vtkRenderingOpenGL2
        # vtkRenderingVolume
        # vtksqlite
        # vtksys
        # vtkViewsCore
    )
endif()

# Translations
qt6_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/utils_fr.ts)

install(TARGETS ikDataIO
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
