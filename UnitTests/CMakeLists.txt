if(IKOMIA_TESTS)
    find_package(Qt5 REQUIRED COMPONENTS Test)

    add_subdirectory(DataIOTests)
    add_subdirectory(CoreTests)
    add_subdirectory(DataProcessTests)
    add_subdirectory(DataSamples)
    add_subdirectory(UtilsTests)
    add_subdirectory(DataPlotTests)

    if(IKOMIA_TESTS_CL)
        add_subdirectory(VolumeRenderTests)
    endif()
endif()
