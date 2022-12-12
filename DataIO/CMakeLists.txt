add_library(ikDataIO SHARED
    ../Core/Data/CvMatNumpyArrayConverter.cpp ../Core/Data/CvMatNumpyArrayConverter.h
    CDataIO.hpp
    CDataImageIO.cpp CDataImageIO.h
    CDataVideoBuffer.cpp CDataVideoBuffer.h
    CDataVideoIO.cpp CDataVideoIO.h
    CDicomImageIO.cpp CDicomImageIO.h
    CNumpyImageIO.cpp CNumpyImageIO.h
    COpencvImageIO.cpp COpencvImageIO.h
    COpencvVideoIO.cpp COpencvVideoIO.h
    CVirtualImageIO.cpp CVirtualImageIO.h
    CVirtualVideoIO.cpp CVirtualVideoIO.h
    DataIOGlobal.hpp
    DataIOTools.hpp
)
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
    target_link_libraries(ikUtils PUBLIC
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
    target_link_libraries(ikUtils PUBLIC
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_videoio
    )
endif()

if(WIN32)
    target_link_libraries(ikDataIO PUBLIC
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
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../quazip/build/lib
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../qwt/build/lib
        # Remove: L'C:\Program
#        (
#        )
#        Files
#        Kits\10\Lib\10.0.19041.0\um\x64'
#        \Windows
#        x86
    )
endif()

if(UNIX)
    target_link_libraries(ikDataIO PUBLIC
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../gmic/build/
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(ikDataIO PUBLIC
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
    )
endif()

if(UNIX AND centos7 AND NOT MACOS)
    target_link_libraries(ikDataIO PUBLIC
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(ikDataIO PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(ikDataIO PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA
#        Computing
#        GPU
#        Toolkit/CUDA/v/bin'
#        Toolkit/CUDA/v/lib/x64'
    )
endif()

install(TARGETS ikDataIO
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
