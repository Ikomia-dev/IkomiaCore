add_library(ikDataManagement SHARED
    CDataManager.hpp
    CImageDataManager.cpp
    CImageDataManager.h
    CVideoDataManager.cpp
    CVideoDataManager.h
    DataManagementGlobal.hpp
)

set_target_properties(ikDataManagement PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

target_compile_definitions(ikDataManagement PRIVATE
    DATAMANAGEMENT_LIBRARY
)

target_compile_features(ikDataManagement PRIVATE cxx_std_14)

target_include_directories(ikDataManagement PRIVATE
    ../Core
    ../DataIO
    ../Utils
)

target_link_libraries(ikDataManagement PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    OpenMP::OpenMP_CXX
    Boost::filesystem
    ${BOOST_PYTHON_TARGET}
    Python3::Python
    opencv_core
    opencv_highgui
    ikUtils
    ikCore
    ikDataIO
)

# Translations
qt5_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/datamanagement_fr.ts)

install(TARGETS ikDataManagement
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)
