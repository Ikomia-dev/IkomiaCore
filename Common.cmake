include(LocalSettings.cmake)

add_compile_definitions(
    QT_DEPRECATED_WARNINGS
    BOOST_ALL_NO_LIB
)

add_compile_options(
    "${OpenMP_CXX_FLAGS}"
)

# ------------------- #
# ----- INCLUDE ----- #
# ------------------- #
if(WIN32)
    include_directories(
        # Boost
        $ENV{ProgramW6432}/Boost/include/boost-${BOOST_VERSION}
        # Python
        $ENV{ProgramW6432}/Python${PYTHON_VERSION_NO_DOT}/include
        # Numpy
        ../../numpy/numpy/core/include
        # VTK
        $ENV{ProgramW6432}/VTK/include/vtk-${VTK_VERSION}
        # OpenCL
        $ENV{ProgramW6432}/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include
        #OpenCV
        $<$<IK_CPU:1>:$ENV{ProgramW6432}/OpenCV/cpu/include>
        $<$<IK_CPU:0>:$ENV{ProgramW6432}/OpenCV/cuda/include>
    )
endif()

if(UNIX AND NOT CENTOS7)
    include_directories(
        # Python
        /usr/include/python${PYTHON_VERSION_DOT}
        /usr/local/include/python${PYTHON_VERSION_DOT}
        # Numpy
        /usr/lib/python${PYTHON_VERSION_DOT}/site-packages/numpy/core/include
        /usr/local/lib/python${PYTHON_VERSION_DOT}/dist-packages/numpy/core/include
        # VTK
        /usr/local/include/vtk-${VTK_MAJOR_MINOR_VERSION}
        #OpenCV
        /usr/local/include/opencv4
    )
endif()

if(UNIX AND CENTOS7)
    include_directories(
        # Global include
        /work/shared/local/include
        # Python
        /work/shared/local/include/python${PYTHON_VERSION_DOT_M}
        # Numpy
        /work/shared/local/lib/python$${PYTHON_VERSION_DOT}/site-packages/numpy/core/include
        # VTK
        /work/shared/local/include/vtk-${VTK_MAJOR_MINOR_VERSION}
        #OpenCV
        /work/shared/local/include/opencv4
    )
endif()

if(MSVC)
    target_compile_options(ikUtils
        -arch:AVX2
        _CRT_SECURE_NO_WARNINGS
    )
endif()
