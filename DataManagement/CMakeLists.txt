add_library(ikDataManagement SHARED
    CDataManager.hpp
    CImageDataManager.cpp
    CImageDataManager.h
    CVideoDataManager.cpp
    CVideoDataManager.h
    DataManagementGlobal.hpp
)

set_target_properties(ikDataManagement PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_include_directories(ikDataManagement PUBLIC
    ../Core
    ../DataIO
    ../Utils
)

target_compile_definitions(ikDataManagement PUBLIC
    DATAMANAGEMENT_LIBRARY
)

target_link_libraries(ikDataManagement PUBLIC
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
)

if(WIN32)
    target_link_libraries(ikDataManagement PUBLIC
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_libraries(ikDataManagement PUBLIC
        opencv_core
        opencv_highgui
    )
endif()

install(TARGETS ikDataManagement
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
