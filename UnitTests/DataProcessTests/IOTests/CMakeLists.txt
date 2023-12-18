add_executable(IOTests WIN32
    CIOTests.cpp
    CIOTests.h
)

target_include_directories(IOTests PRIVATE
    ../../../Core
    ../../../DataIO
    ../../../DataProcess
    ../../../Utils
)

target_compile_features(IOTests PRIVATE cxx_std_14)

target_link_libraries(IOTests PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Test
    Qt::Widgets
    Python3::Python
    Boost::system
    Boost::filesystem
    ${BOOST_PYTHON_TARGET}
    OpenMP::OpenMP_CXX
    # OpenCV
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
    gmic
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
    # Ikomia
    ikUtils
    ikCore
    ikDataIO
    ikDataProcess
)

if(UNIX)
    target_link_libraries(IOTests PRIVATE
        # VTK
        vtkalglib-${VTK_MAJOR_MINOR_VERSION}
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
        vtkglew-${VTK_MAJOR_MINOR_VERSION}
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
    target_link_libraries(IOTests PRIVATE
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

install(TARGETS IOTests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
