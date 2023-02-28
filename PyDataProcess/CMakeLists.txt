set(CMAKE_INSTALL_IKOMIA_PYDATAPROCESS ${CMAKE_INSTALL_IKOMIA_PYTHON_LIB}/dataprocess)

add_library(pydataprocess SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp
    ../Core/Data/CvMatNumpyArrayConverter.h
    ../PyCore/VectorConverter.hpp
    ../PyCore/MapConverter.hpp
    CIkomiaRegistryWrap.cpp
    CIkomiaRegistryWrap.h
    CPluginProcessInterfaceWrap.cpp
    CPluginProcessInterfaceWrap.h
    CWidgetFactoryWrap.cpp
    CWidgetFactoryWrap.h
    CWorkflowWrap.cpp
    CWorkflowWrap.h
    IO/CArrayIOWrap.cpp
    IO/CArrayIOWrap.h
    IO/CDatasetIOWrap.cpp
    IO/CDatasetIOWrap.h
    IO/CGraphicsInputWrap.cpp
    IO/CGraphicsInputWrap.h
    IO/CGraphicsOutputWrap.cpp
    IO/CGraphicsOutputWrap.h
    IO/CImageIOWrap.cpp
    IO/CImageIOWrap.h
    IO/CInstanceSegIOWrap.cpp
    IO/CInstanceSegIOWrap.h
    IO/CKeyptsIOWrap.cpp
    IO/CKeyptsIOWrap.h
    IO/CNumericIOWrap.hpp
    IO/CObjectDetectionIOWrap.cpp
    IO/CObjectDetectionIOWrap.h
    IO/CPathIOWrap.cpp
    IO/CPathIOWrap.h
    IO/CSemanticSegIOWrap.cpp
    IO/CSemanticSegIOWrap.h
    IO/CVideoIOWrap.cpp
    IO/CVideoIOWrap.h
    IO/CWidgetOutputWrap.cpp
    IO/CWidgetOutputWrap.h
    PyDataProcess.cpp
    PyDataProcess.h
    PyDataProcessDocString.hpp
    PyDataProcessGlobal.h
    PyDataProcessTools.hpp
    Task/C2dImageInteractiveTaskWrap.cpp
    Task/C2dImageInteractiveTaskWrap.h
    Task/C2dImageTaskWrap.cpp
    Task/C2dImageTaskWrap.h
    Task/CDnnTrainTaskWrap.cpp
    Task/CDnnTrainTaskWrap.h
    Task/CKeyptsDetectTaskWrap.cpp
    Task/CKeyptsDetectTaskWrap.h
    Task/CTaskFactoryWrap.cpp
    Task/CTaskFactoryWrap.h
    Task/CVideoOFTaskWrap.cpp
    Task/CVideoOFTaskWrap.h
    Task/CVideoTaskWrap.cpp
    Task/CVideoTaskWrap.h
    Task/CVideoTrackingTaskWrap.cpp
    Task/CVideoTrackingTaskWrap.h
    Task/CClassifTaskWrap.cpp
    Task/CClassifTaskWrap.h
    Task/CObjDetectTaskWrap.cpp
    Task/CObjDetectTaskWrap.h
    Task/CSemanticSegTaskWrap.cpp
    Task/CSemanticSegTaskWrap.h
    Task/CInstanceSegTaskWrap.cpp
    Task/CInstanceSegTaskWrap.h
)

set_target_properties(pydataprocess PROPERTIES
    VERSION ${IKOMIA_VERSION}
    SOVERSION ${IKOMIA_VERSION}
    PREFIX ""
)

if (WIN32)
    set_target_properties(pydataprocess PROPERTIES SUFFIX ".pyd")
endif()

target_compile_definitions(pydataprocess PRIVATE
    PYDATAPROCESS_LIBRARY
)

target_compile_features(pydataprocess PRIVATE cxx_std_14)

target_include_directories(pydataprocess PRIVATE
    ../Core
    ../DataIO
    ../DataProcess
    ../Utils
    ../PyCore
)

target_link_libraries(pydataprocess PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Python3::Python
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
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
    ikUtils
    ikCore
    ikDataIO
    ikDataProcess
)

if(WIN32)
    target_link_directories(pydataprocess PRIVATE
        # GMIC
        $ENV{ProgramW6432}/gmic/bin
        $ENV{ProgramW6432}/gmic/lib
    )
endif()

if(UNIX)
    target_link_directories(pydataprocess PRIVATE
        # GMIC
        ${CMAKE_CURRENT_LIST_DIR}/../../gmic/build
    )
endif()

if(NOT IK_CPU)
    target_link_libraries(pydataprocess PRIVATE
        opencv_cudawarping
    )
endif()

if(WIN32)
    target_compile_options(pydataprocess PRIVATE
        /bigobj
    )
endif()

install(TARGETS pydataprocess
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_PYDATAPROCESS}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_PYDATAPROCESS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_PYDATAPROCESS}
)
