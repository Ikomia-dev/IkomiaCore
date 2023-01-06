add_compile_definitions(
    QT_DEPRECATED_WARNINGS
    BOOST_ALL_NO_LIB
)

if(MSVC)
    add_compile_options(
        /arch:AVX2
        -D_CRT_SECURE_NO_WARNINGS
    )
endif()

# ------------------- #
# ----- INCLUDE ----- #
# ------------------- #
include_directories(
    # Python
    ${Python3_INCLUDE_DIRS}
    # Numpy
    ${Python3_NumPy_INCLUDE_DIRS}
)

if(WIN32)
    include_directories(
        # Boost
        ${Boost_INCLUDE_DIRS}/Boost/include/boost-${BOOST_VERSION}
        # VTK
        ${VTK_INCLUDE_DIRS}
        # OpenCL
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/include"
        #OpenCV
        ${OpenCV_INCLUDE_DIRS}
    )
endif()

if(UNIX AND NOT CENTOS7)
    include_directories(
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
        #/work/shared/local/include/python${PYTHON_VERSION_DOT_M}
        # Numpy
        #/work/shared/local/lib/python$${PYTHON_VERSION_DOT}/site-packages/numpy/core/include
        # VTK
        /work/shared/local/include/vtk-${VTK_MAJOR_MINOR_VERSION}
        #OpenCV
        /work/shared/local/include/opencv4
    )
endif()

# --------------- #
# ----- LIB ----- #
# --------------- #
link_directories(
    ${CMAKE_CURRENT_LIST_DIR}/Build/lib
)
