include(../LocalSettings.pri)

# If IKOMIA_TESTS is not defined, all test projects are excluded from build
requires(IKOMIA_TESTS)

TEMPLATE = subdirs

SUBDIRS += \
    DataIOTests \
    CoreTests \
    DataProcessTests \
    DataSamples \
    UtilsTests \
    DataPlotTests

enableTestCL {
    SUBDIRS += VolumeRenderTests
}

QMAKE_EXTRA_TARGETS += vera++
