add_executable(DataSamples WIN32 MACOSX_BUNDLE
    main.cpp
)

target_link_libraries(DataSamples PRIVATE
    Qt::Core
    Qt::Gui
)

# Resources
set(RESOURCES "")
qt5_add_resources(RESOURCES samples.qrc)
target_sources(DataSamples PRIVATE ${RESOURCES})

install(TARGETS DataSamples
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
