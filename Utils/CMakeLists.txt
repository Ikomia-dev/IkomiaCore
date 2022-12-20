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
    Widgets/QtPropertyBrowser/qtbuttonpropertybrowser.cpp
    Widgets/QtPropertyBrowser/qtbuttonpropertybrowser.h
    Widgets/QtPropertyBrowser/qteditorfactory.cpp
    Widgets/QtPropertyBrowser/qteditorfactory.h
    Widgets/QtPropertyBrowser/qtgroupboxpropertybrowser.cpp
    Widgets/QtPropertyBrowser/qtgroupboxpropertybrowser.h
    Widgets/QtPropertyBrowser/qtpropertybrowser.cpp
    Widgets/QtPropertyBrowser/qtpropertybrowser.h
    Widgets/QtPropertyBrowser/qtpropertybrowserutils.cpp
    Widgets/QtPropertyBrowser/qtpropertybrowserutils_p.h
    Widgets/QtPropertyBrowser/qtpropertymanager.cpp
    Widgets/QtPropertyBrowser/qtpropertymanager.h
    Widgets/QtPropertyBrowser/qttreepropertybrowser.cpp
    Widgets/QtPropertyBrowser/qttreepropertybrowser.h
    Widgets/QtPropertyBrowser/qtvariantproperty.cpp
    Widgets/QtPropertyBrowser/qtvariantproperty.h
    base64.hpp
    qaesencryption.cpp
    qaesencryption.h
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

target_include_directories(ikUtils PRIVATE
    Widgets/FancyTabBar
    Widgets/QtPropertyBrowser
)

target_compile_definitions(ikUtils PRIVATE
    QT_QTPROPERTYBROWSER_EXPORT
    UTILS_LIBRARY
)

target_compile_features(ikUtils PRIVATE cxx_std_14)

target_link_libraries(ikUtils PRIVATE
    Qt::Core
    Qt::Gui
    Qt::Network
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
)

if(WIN32)
    target_link_libraries(ikUtils PRIVATE
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikUtils PRIVATE
        opencv_core
    )
endif()

set(QTPROP_RESOURCES "")
qt5_add_resources(QTPROP_RESOURCES Widgets/QtPropertyBrowser/qtpropertybrowser.qrc)
target_sources(ikUtils PRIVATE ${QTPROP_RESOURCES})

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
