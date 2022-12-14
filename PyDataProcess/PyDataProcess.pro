#-------------------------------------------------
#
# Project created by QtCreator 2018-02-27T11:45:29
#
#-------------------------------------------------
include(../IkomiaCore.pri)

QT += core gui widgets sql concurrent
CONFIG += plugin no_plugin_name_prefix
TARGET = pydataprocess
TEMPLATE = lib
VERSION = $$IKOMIA_LIB_VERSION
win32:QMAKE_EXTENSION_SHLIB = pyd
macx:QMAKE_EXTENSION_SHLIB = so

win32:QMAKE_CXXFLAGS += -bigobj

DEFINES += PYDATAPROCESS_LIBRARY BOOST_ALL_NO_LIB

SOURCES += \
        ../Core/Data/CvMatNumpyArrayConverter.cpp \
        CIkomiaRegistryWrap.cpp \
        CWorkflowWrap.cpp \
        IO/CArrayIOWrap.cpp \
        IO/CDatasetIOWrap.cpp \
        IO/CGraphicsInputWrap.cpp \
        IO/CGraphicsOutputWrap.cpp \
        IO/CImageIOWrap.cpp \
        IO/CInstanceSegIOWrap.cpp \
        IO/CObjectDetectionIOWrap.cpp \
        IO/CPathIOWrap.cpp \
        IO/CSemanticSegIOWrap.cpp \
        IO/CVideoIOWrap.cpp \
        IO/CWidgetOutputWrap.cpp \
        PyDataProcess.cpp \
        CWidgetFactoryWrap.cpp \
        CPluginProcessInterfaceWrap.cpp \
        Task/C2dImageInteractiveTaskWrap.cpp \
        Task/C2dImageTaskWrap.cpp \
        Task/CDnnTrainTaskWrap.cpp \
        Task/CTaskFactoryWrap.cpp \
        Task/CVideoOFTaskWrap.cpp \
        Task/CVideoTaskWrap.cpp \
        Task/CVideoTrackingTaskWrap.cpp

HEADERS += \
        ../Core/Data/CvMatNumpyArrayConverter.h \
        CIkomiaRegistryWrap.h \
        CWorkflowWrap.h \
        IO/CArrayIOWrap.h \
        IO/CDatasetIOWrap.h \
        IO/CGraphicsInputWrap.h \
        IO/CGraphicsOutputWrap.h \
        IO/CImageIOWrap.h \
        IO/CInstanceSegIOWrap.h \
        IO/CNumericIOWrap.hpp \
        IO/CObjectDetectionIOWrap.h \
        IO/CPathIOWrap.h \
        IO/CSemanticSegIOWrap.h \
        IO/CVideoIOWrap.h \
        IO/CWidgetOutputWrap.h \
        PyDataProcess.h \
        PyDataProcessDocString.hpp \
        PyDataProcessGlobal.h \
        PyDataProcessTools.hpp \
        CWidgetFactoryWrap.h \
        CPluginProcessInterfaceWrap.h \
        Task/C2dImageInteractiveTaskWrap.h \
        Task/C2dImageTaskWrap.h \
        Task/CDnnTrainTaskWrap.h \
        Task/CTaskFactoryWrap.h \
        Task/CVideoOFTaskWrap.h \
        Task/CVideoTaskWrap.h \
        Task/CVideoTrackingTaskWrap.h

#Make install directive
target.path = ../../IkomiaApi/ikomia/dataprocess
INSTALLS += target

LIBS += $$link_python()

LIBS += $$link_boost()

# Dynamic link to OpenCV
win32:CONFIG(release, debug|release): LIBS += -lopencv_core$${OPENCV_VERSION} -lopencv_imgproc$${OPENCV_VERSION} -lopencv_objdetect$${OPENCV_VERSION} -lopencv_photo$${OPENCV_VERSION} -lopencv_ximgproc$${OPENCV_VERSION} -lopencv_highgui$${OPENCV_VERSION}
win32:CONFIG(release, debug|release): LIBS += -lopencv_xphoto$${OPENCV_VERSION} -lopencv_fuzzy$${OPENCV_VERSION} -lopencv_hfs$${OPENCV_VERSION} -lopencv_dnn$${OPENCV_VERSION} -lopencv_tracking$${OPENCV_VERSION} -lopencv_video$${OPENCV_VERSION}
win32:CONFIG(release, debug|release): LIBS += -lopencv_bgsegm$${OPENCV_VERSION} -lopencv_optflow$${OPENCV_VERSION} -lopencv_bioinspired$${OPENCV_VERSION} -lopencv_saliency$${OPENCV_VERSION} -lopencv_superres$${OPENCV_VERSION} -lopencv_text$${OPENCV_VERSION}
win32:CONFIG(release, debug|release): LIBS += -lopencv_features2d$${OPENCV_VERSION}
win32:!ik_cpu:CONFIG(release, debug|release): LIBS += -lopencv_cudawarping$${OPENCV_VERSION}
win32:CONFIG(debug, debug|release):LIBS += -lopencv_core$${OPENCV_VERSION}d -lopencv_imgproc$${OPENCV_VERSION}d -lopencv_objdetect$${OPENCV_VERSION}d -lopencv_photo$${OPENCV_VERSION}d -lopencv_ximgproc$${OPENCV_VERSION}d -lopencv_highgui$${OPENCV_VERSION}d
win32:CONFIG(debug, debug|release):LIBS += -lopencv_xphoto$${OPENCV_VERSION}d -lopencv_fuzzy$${OPENCV_VERSION}d -lopencv_hfs$${OPENCV_VERSION}d -lopencv_dnn$${OPENCV_VERSION}d -lopencv_tracking$${OPENCV_VERSION}d -lopencv_video$${OPENCV_VERSION}d
win32:CONFIG(debug, debug|release):LIBS += -lopencv_bgsegm$${OPENCV_VERSION}d -lopencv_optflow$${OPENCV_VERSION}d -lopencv_bioinspired$${OPENCV_VERSION}d -lopencv_saliency$${OPENCV_VERSION}d -lopencv_superres$${OPENCV_VERSION}d -lopencv_text$${OPENCV_VERSION}d
win32:CONFIG(debug, debug|release):LIBS += -lopencv_features2d$${OPENCV_VERSION}d
win32:!ik_cpu:CONFIG(debug, debug|release): LIBS += -lopencv_cudawarping$${OPENCV_VERSION}d

unix: LIBS += -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_photo -lopencv_ximgproc -lopencv_highgui
unix: LIBS += -lopencv_xphoto -lopencv_fuzzy -lopencv_hfs -lopencv_dnn -lopencv_tracking -lopencv_video
unix: LIBS += -lopencv_bgsegm -lopencv_optflow -lopencv_bioinspired -lopencv_saliency -lopencv_superres -lopencv_text
unix: LIBS += -lopencv_face -lopencv_features2d
unix: !ik_cpu: LIBS += -lopencv_cudawarping

#Dynamic link to gmic
LIBS += -lgmic

LIBS += $$link_utils()
INCLUDEPATH += $$PWD/../Utils
DEPENDPATH += $$PWD/../Utils

LIBS += $$link_core()
INCLUDEPATH += $$PWD/../Core
DEPENDPATH += $$PWD/../Core

LIBS += $$link_dataio()
INCLUDEPATH += $$PWD/../DataIO
DEPENDPATH += $$PWD/../DataIO

LIBS += $$link_dataprocess()
INCLUDEPATH += $$PWD/../DataProcess
DEPENDPATH += $$PWD/../DataProcess
