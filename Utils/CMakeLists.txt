add_library(ikUtils SHARED
    CException.cpp
    CException.h
    CLogManager.cpp
    CLogManager.h
    CMemoryInfo.cpp
    CMemoryInfo.h
    CNetworkManager.cpp
    CNetworkManager.h
    CProgressSignalHandler.cpp
    CProgressSignalHandler.h
    CTimer.hpp
    CTreeDbManager.hpp
    CWindowsDeviceEnumerator.cpp
    CWindowsDeviceEnumerator.h
    Containers/CQueue.hpp
    DesignPattern/CAbstractFactory.hpp
    DesignPattern/CFactoryRegister.hpp
    DesignPattern/CObjectLocker.hpp
    DesignPattern/CSingleton.hpp
    ExceptionCode.hpp
    PythonThread.hpp
    UnitTestUtils.hpp
    UtilsDefine.hpp
    UtilsGlobal.hpp
    UtilsTools.hpp
    Widgets/CBrowseFileWidget.cpp
    Widgets/CBrowseFileWidget.h
    Widgets/CColorPushButton.cpp
    Widgets/CColorPushButton.h
    Widgets/FancyTabBar/fancytab.cpp
    Widgets/FancyTabBar/fancytab.h
    Widgets/FancyTabBar/fancytabbar.cpp
    Widgets/FancyTabBar/fancytabbar.h
    Widgets/FancyTabBar/stylehelper.cpp
    Widgets/FancyTabBar/stylehelper.h
    base64.hpp
    qaesencryption.cpp qaesencryption.h
)

set_target_properties(ikUtils PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

set(PUBLIC_HEADERS_UTILS
    CException.h
    CLogManager.h
    UtilsTools.hpp
    ExceptionCode.hpp
    CMemoryInfo.h
    CTimer.hpp
    UtilsGlobal.hpp
    UtilsDefine.hpp
    CProgressSignalHandler.h
    CWindowsDeviceEnumerator.h
    PythonThread.hpp
)

set(PUBLIC_HEADERS_DESIGN_PATTERN
    DesignPattern/CFactoryRegister.hpp
    DesignPattern/CAbstractFactory.hpp
    DesignPattern/CSingleton.hpp
    DesignPattern/CObjectLocker.hpp
)

set(PUBLIC_HEADERS_CONTAINERS
    Containers/CQueue.hpp
)

set(PUBLIC_HEADERS_WIDGETS
    Widgets/CColorPushButton.h
    Widgets/CBrowseFileWidget.h
)

target_include_directories(ikUtils PUBLIC
    Widgets/FancyTabBar
    Widgets/QtPropertyBrowser
)

target_compile_definitions(ikUtils PUBLIC
    QT_QTPROPERTYBROWSER_EXPORT
    UTILS_LIBRARY
)

target_link_libraries(ikUtils PUBLIC
    Qt::Core
    Qt::Gui
    Qt::Network
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
)

if(WIN32)
    target_link_libraries(ikUtils PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikUtils PUBLIC
        opencv_core
    )
endif()

if(WIN32)
    target_link_libraries(ikUtils PUBLIC
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
    target_link_libraries(ikUtils PUBLIC
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(ikUtils PUBLIC
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(ikUtils PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(ikUtils PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA/Computing GPU
        # Toolkit/CUDA/v/bin'
        # Toolkit/CUDA/v/lib/x64'
    )
endif()

if(qtpropertybrowser-buildlib OR NOT qtpropertybrowser-uselib)
    list(APPEND QTPROP_SOURCES
        Widgets/QtPropertyBrowser/qtbuttonpropertybrowser.cpp Widgets/QtPropertyBrowser/qtbuttonpropertybrowser.h
        Widgets/QtPropertyBrowser/qteditorfactory.cpp Widgets/QtPropertyBrowser/qteditorfactory.h
        Widgets/QtPropertyBrowser/qtgroupboxpropertybrowser.cpp Widgets/QtPropertyBrowser/qtgroupboxpropertybrowser.h
        Widgets/QtPropertyBrowser/qtpropertybrowser.cpp Widgets/QtPropertyBrowser/qtpropertybrowser.h
        Widgets/QtPropertyBrowser/qtpropertybrowserutils.cpp Widgets/QtPropertyBrowser/qtpropertybrowserutils_p.h
        Widgets/QtPropertyBrowser/qtpropertymanager.cpp Widgets/QtPropertyBrowser/qtpropertymanager.h
        Widgets/QtPropertyBrowser/qttreepropertybrowser.cpp Widgets/QtPropertyBrowser/qttreepropertybrowser.h
        Widgets/QtPropertyBrowser/qtvariantproperty.cpp Widgets/QtPropertyBrowser/qtvariantproperty.h
    )

    qt5_add_resources(QTPROP_SOURCES Widgets/QtPropertyBrowser/qtpropertybrowser.qrc)
    target_sources(ikUtils PUBLIC ${QTPROP_SOURCES})

    # Resources:
#    set(qtpropertybrowser_resource_files
#        "Widgets/QtPropertyBrowser/images/cursor-arrow.png"
#        "Widgets/QtPropertyBrowser/images/cursor-busy.png"
#        "Widgets/QtPropertyBrowser/images/cursor-closedhand.png"
#        "Widgets/QtPropertyBrowser/images/cursor-cross.png"
#        "Widgets/QtPropertyBrowser/images/cursor-forbidden.png"
#        "Widgets/QtPropertyBrowser/images/cursor-hand.png"
#        "Widgets/QtPropertyBrowser/images/cursor-hsplit.png"
#        "Widgets/QtPropertyBrowser/images/cursor-ibeam.png"
#        "Widgets/QtPropertyBrowser/images/cursor-openhand.png"
#        "Widgets/QtPropertyBrowser/images/cursor-sizeall.png"
#        "Widgets/QtPropertyBrowser/images/cursor-sizeb.png"
#        "Widgets/QtPropertyBrowser/images/cursor-sizef.png"
#        "Widgets/QtPropertyBrowser/images/cursor-sizeh.png"
#        "Widgets/QtPropertyBrowser/images/cursor-sizev.png"
#        "Widgets/QtPropertyBrowser/images/cursor-uparrow.png"
#        "Widgets/QtPropertyBrowser/images/cursor-vsplit.png"
#        "Widgets/QtPropertyBrowser/images/cursor-wait.png"
#        "Widgets/QtPropertyBrowser/images/cursor-whatsthis.png"
#    )

#    qt_add_resources(ikUtils "qtpropertybrowser"
#        PREFIX
#            "/trolltech/qtpropertybrowser"
#        BASE
#            "Widgets/QtPropertyBrowser"
#        FILES
#            ${qtpropertybrowser_resource_files}
#    )
endif()

install(TARGETS ikUtils
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)

install(FILES ${PUBLIC_HEADERS_UTILS}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Utils
)

install(FILES ${PUBLIC_HEADERS_DESIGN_PATTERN}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Utils/DesignPattern
)

install(FILES ${PUBLIC_HEADERS_CONTAINERS}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Utils/Containers
)

install(FILES ${PUBLIC_HEADERS_WIDGETS}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/Utils/Widgets
)
