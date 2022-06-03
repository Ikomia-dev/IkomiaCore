include(../../../IkomiaCore.pri)

QT       += core gui testlib widgets sql concurrent

TARGET = IOTests

CONFIG   -= app_bundle

# For Jenkins
CONFIG += testcase

DEFINES += BOOST_ALL_NO_LIB

unix {
    target.path = ../../../Build/Tests/
    INSTALLS += target
    QMAKE_RPATHDIR += $$PWD/../../../Build/Lib
    QMAKE_RPATHDIR += /usr/local/lib
}

TEMPLATE = app

HEADERS += \
    CIOTests.h

SOURCES += \
    CIOTests.cpp

LIBS += $$link_boost()

#Dynamic link to gmic
LIBS += -lgmic

#Dynamic link to Utils
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../../Utils/release/ -likUtils
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../../Utils/debug/ -likUtils
else:unix: LIBS += -L$$OUT_PWD/../../../Utils/ -likUtils
INCLUDEPATH += $$PWD/../../../Utils
DEPENDPATH += $$PWD/../../../Utils

#Dynamic link to Core
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../../Core/release/ -likCore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../../Core/debug/ -likCore
else:unix: LIBS += -L$$OUT_PWD/../../../Core/ -likCore
INCLUDEPATH += $$PWD/../../../Core
DEPENDPATH += $$PWD/../../../Core

#Dynamic link to DataIO
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../../DataIO/release/ -likDataIO
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../../DataIO/debug/ -likDataIO
else:unix: LIBS += -L$$OUT_PWD/../../../DataIO/ -likDataIO
INCLUDEPATH += $$PWD/../../../DataIO
DEPENDPATH += $$PWD/../../../DataIO

#Dynamic link to DataProcess
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../../DataProcess/release/ -likDataProcess
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../../DataProcess/debug/ -likDataProcess
else:unix: LIBS += -L$$OUT_PWD/../../../DataProcess/ -likDataProcess
INCLUDEPATH += $$PWD/../../../DataProcess
DEPENDPATH += $$PWD/../../../DataProcess
