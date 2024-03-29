#include <QtTest>
#include "COcvProcessTests.h"
#include "UnitTestUtils.hpp"

COcvProcessTests::COcvProcessTests(QObject *parent) : QObject(parent)
{
}

void COcvProcessTests::initTestCase()
{
    m_image = loadSampleImage();

    std::string imagePath = UnitTest::getDataPath() + "/Videos/highway.avi";
    m_videoInputPtr = std::make_shared<CVideoIO>();
    m_videoInputPtr->setVideoPath(imagePath);
}

void COcvProcessTests::init()
{
    initImageTypes();
}

void COcvProcessTests::bckSubCnt()
{
    enableImageTypeFromDepth({CV_8U});
    runStdTest1("ocv_bck_substractor_cnt", std::make_shared<COcvBckgndSubCntParam>(), 0, 1);
}

void COcvProcessTests::bckSubGmg()
{
    enableImageTypeFromDepth({CV_8U, CV_16U, CV_32F});
    runStdTest1("ocv_bck_substractor_gmg", std::make_shared<COcvBckgndSubGmgParam>(), 0, 1);
}

void COcvProcessTests::bckSubGsoc()
{
    enableImageTypeFromDepth({CV_8U, CV_32F});
    runStdTest1("ocv_bck_substractor_gsoc", std::make_shared<COcvBckgndSubGsocParam>(), 0, 1);
}

void COcvProcessTests::bckSubLsbp()
{
    enableImageTypeFromDepth({CV_8U, CV_32F});
    runStdTest1("ocv_bck_substractor_lspb", std::make_shared<COcvBckgndSubLsbpParam>(), 0, 1);
}

void COcvProcessTests::bckSubMog()
{
    enableImageTypeFromDepth({CV_8U});
    runStdTest1("ocv_bck_substractor_mog", std::make_shared<COcvBckgndSubMogParam>(), 0, 1);
}

void COcvProcessTests::retina()
{
    enableAllImageTypes();
    auto paramPtr = std::make_shared<COcvRetinaParam>();
    runStdTest1("ocv_retina", paramPtr, 0, 1);
    paramPtr->m_bUseOCL = true;
    runOpenCL("ocv_retina", paramPtr, 0);
}

void COcvProcessTests::retinaSegmentation()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_8SC1, CV_8SC3, CV_16UC1, CV_16SC1, CV_16FC1, CV_16FC3, CV_32SC1, CV_32FC1, CV_32FC3, CV_64FC1});
    runStdTest1("ocv_retina_segmentation", std::make_shared<COcvRetinaSegmentationParam>(), 0, 1);
}

void COcvProcessTests::dft()
{
    const std::string name = "ocv_dft";
    auto paramPtr = std::make_shared<COcvDftParam>();
    enableImageTypeFromChannels(1);
    runNoInput(name, paramPtr);
    runTypeImageTest(name, paramPtr, 0);
}

void COcvProcessTests::exp()
{
    enableAllImageTypes();
    runStdTest1("ocv_exp", std::make_shared<CWorkflowTaskParam>(), 0, 1);
}

void COcvProcessTests::extractChannel()
{
    const std::string name = "ocv_extract_channel";
    auto paramPtr = std::make_shared<COcvExtractChannelParam>();

    enableAllImageTypes();
    runImage(name, paramPtr, 0);
    runTypeImageTest(name, paramPtr, 0);
    runVideo(name, paramPtr, 0);
    paramPtr->m_index = -1;
    QVERIFY_EXCEPTION_THROWN(runImage(name, paramPtr, 0), std::exception);
    paramPtr->m_index = 4;
    QVERIFY_EXCEPTION_THROWN(runImage(name, paramPtr, 0), std::exception);
}

void COcvProcessTests::flip()
{
    enableAllImageTypes();
    runStdTest1("ocv_flip", std::make_shared<COcvFlipParam>(), 0, 1);
}

void COcvProcessTests::inRange()
{
    enableAllImageTypes();
    runStdTest1("ocv_in_range", std::make_shared<COcvInRangeParam>(), 0, 1);
}

void COcvProcessTests::kmeans()
{
    enableAllImageTypes();
    runStdTest1("ocv_kmeans", std::make_shared<COcvKMeansParam>(), 0, 1);
}

void COcvProcessTests::log()
{
    enableAllImageTypes();
    runStdTest1("ocv_log", std::make_shared<CWorkflowTaskParam>(), 0, 1);
}

void COcvProcessTests::negative()
{
    enableAllImageTypes();
    runStdTest1("ocv_negative", std::make_shared<COcvNegativeParam>(), 0, 1);
}

void COcvProcessTests::normalize()
{
    enableAllImageTypes();
    runStdTest1("ocv_normalize", std::make_shared<COcvNormalizeParam>(), 0, 1);
}

void COcvProcessTests::rotate()
{
    enableAllImageTypes();
    runStdTest1("ocv_rotate", std::make_shared<COcvRotateParam>(), 0, 1);
}

void COcvProcessTests::agast()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_16UC3, CV_32FC3});
    runStdTest1("ocv_agast", std::make_shared<COcvAGASTParam>(), 0, 1);
}

void COcvProcessTests::akaze()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_32FC1, CV_32FC3});
    runStdTest1("ocv_akaze", std::make_shared<COcvAKAZEParam>(), 0, 1);
}

void COcvProcessTests::brisk()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3});
    runStdTest1("ocv_brisk", std::make_shared<COcvBRISKParam>(), 0, 1);
}

void COcvProcessTests::fast()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_16UC3, CV_32FC3});
    runStdTest1("ocv_fast", std::make_shared<COcvFASTParam>(), 0, 1);
}

void COcvProcessTests::gftt()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_32FC3});
    runStdTest1("ocv_gftt_detector", std::make_shared<COcvGFTTParam>(), 0, 1);
}

void COcvProcessTests::kaze()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_32FC1, CV_32FC3});
    runStdTest1("ocv_kaze", std::make_shared<COcvKAZEParam>(), 0, 1);
}

void COcvProcessTests::orb()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3, CV_16UC3, CV_32FC3});
    runStdTest1("ocv_orb", std::make_shared<COcvORBParam>(), 0, 1);
}

void COcvProcessTests::simpleBlobDetector()
{
    enableImageTypeFromType({CV_8UC1, CV_8UC3});
    runStdTest1("ocv_simple_blob_detector", std::make_shared<COcvSimpleBlobDetectorParam>(), 0, 1);
}

void COcvProcessTests::initImageTypes()
{
    m_imgTypes[CV_8UC1] = false;
    m_imgTypes[CV_8UC3] = false;
    //m_imgTypes[CV_8UC4] = false;
    m_imgTypes[CV_8SC1] = false;
    m_imgTypes[CV_8SC3] = false;
    //m_imgTypes[CV_8SC4] = false;
    m_imgTypes[CV_16UC1] = false;
    m_imgTypes[CV_16UC3] = false;
    //m_imgTypes[CV_16UC4] = false;
    m_imgTypes[CV_16SC1] = false;
    m_imgTypes[CV_16SC3] = false;
    //m_imgTypes[CV_16SC4] = false;
    //m_imgTypes[CV_16FC1] = false;
    //m_imgTypes[CV_16FC3] = false;
    //m_imgTypes[CV_16FC4] = false;
    m_imgTypes[CV_32SC1] = false;
    m_imgTypes[CV_32SC3] = false;
    //m_imgTypes[CV_32SC4] = false;
    m_imgTypes[CV_32FC1] = false;
    m_imgTypes[CV_32FC3] = false;
    //m_imgTypes[CV_32FC4] = false;
    m_imgTypes[CV_64FC1] = false;
    m_imgTypes[CV_64FC3] = false;
    //m_imgTypes[CV_64FC4] = false;
}

int COcvProcessTests::getChannelsFromType(int type)
{
    int channels = 1;
    switch(type)
    {
        case CV_8UC1:
        case CV_16UC1:
        case CV_16SC1:
        case CV_32SC1:
        case CV_32FC1:
        case CV_64FC1:
            channels = 1;
            break;
        case CV_8UC3:
        case CV_16UC3:
        case CV_16SC3:
        case CV_32SC3:
        case CV_32FC3:
        case CV_64FC3:
            channels = 3;
            break;
        case CV_8UC4:
        case CV_16UC4:
        case CV_16SC4:
        case CV_32SC4:
        case CV_32FC4:
        case CV_64FC4:
            channels = 4;
            break;
    }
    return channels;
}

void COcvProcessTests::enableAllImageTypes()
{
    for(auto it=m_imgTypes.begin(); it!=m_imgTypes.end(); ++it)
        m_imgTypes[it.key()] = true;
}

void COcvProcessTests::enableImageTypeFromDepth(std::vector<int> depthList)
{
    for(size_t i=0; i<depthList.size(); ++i)
    {
        for(auto it=m_imgTypes.begin(); it!=m_imgTypes.end(); ++it)
        {
            if(CV_MAT_DEPTH(it.key()) == depthList[i])
                m_imgTypes[it.key()] = true;
        }
    }
}

void COcvProcessTests::enableImageTypeFromChannels(int channels)
{
    for(auto it=m_imgTypes.begin(); it!=m_imgTypes.end(); ++it)
    {
        if(getChannelsFromType(it.key()) == channels)
            m_imgTypes[it.key()] = true;
    }
}

void COcvProcessTests::enableImageTypeFromType(std::vector<int> types)
{
    for(size_t i=0; i<types.size(); ++i)
    {
        for(auto it=m_imgTypes.begin(); it!=m_imgTypes.end(); ++it)
        {
            if(it.key() == types[i])
                m_imgTypes[it.key()] = true;
        }
    }
}

CMat COcvProcessTests::convertImage(const CMat &src, int newType)
{
    int type = src.type();
    if(type != newType)
    {
        CMat newImage;
        int srcChannels = src.channels();
        if(srcChannels == getChannelsFromType(newType))
            newImage = src.clone();
        else
        {
            if(srcChannels == 3)
                cv::cvtColor(src, newImage, cv::COLOR_RGB2GRAY);
            else
                cv::cvtColor(src, newImage, cv::COLOR_GRAY2RGB);
        }

        int srcDepth = src.depth();
        if(srcDepth == CV_MAT_DEPTH(newType))
            return newImage;
        else
        {
            CMat newImage2;
            newImage.convertTo(newImage2, newType);
            return newImage2;
        }
    }
    return src;
}

void COcvProcessTests::addGraphicsRect(std::shared_ptr<CGraphicsInput>& graphicsInputPtr)
{
    if(graphicsInputPtr == nullptr)
        return;

    std::vector<ProxyGraphicsItemPtr> items;
    auto graphicsRectPtr = std::make_shared<CProxyGraphicsRect>(50, 50, 150, 100);
    items.push_back(graphicsRectPtr);
    graphicsInputPtr->setItems(items);
}

CMat COcvProcessTests::loadSampleImage()
{
    std::string imagePath = UnitTest::getDataPath() + "/Images/Lena.png";
    CMat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    return img;
}

void COcvProcessTests::runNoInput(const std::string algoName, const WorkflowTaskParamPtr &paramPtr)
{
    auto factory = m_processRegister.getProcessFactory();
    auto processPtr = factory.createObject(algoName, std::move(paramPtr));
    QVERIFY(processPtr != nullptr);

    //Apply with no input data
    QVERIFY_EXCEPTION_THROWN(processPtr->run(), CException);
}

void COcvProcessTests::runImage(const std::string algoName, const WorkflowTaskParamPtr& paramPtr, int inputIndex, int graphicsInputIndex)
{
    auto factory = m_processRegister.getProcessFactory();
    auto processPtr = factory.createObject(algoName, std::move(paramPtr));
    QVERIFY(processPtr != nullptr);
    auto inputPtr = std::dynamic_pointer_cast<CImageIO>(processPtr->getInput(inputIndex));
    QVERIFY(inputPtr != nullptr);
    inputPtr->setImage(m_image);

    if(graphicsInputIndex != -1)
    {
        auto graphicsInputPtr = std::dynamic_pointer_cast<CGraphicsInput>(processPtr->getInput(graphicsInputIndex));
        addGraphicsRect(graphicsInputPtr);
    }

    processPtr->run();
}

void COcvProcessTests::runVideo(const std::string algoName, const WorkflowTaskParamPtr& paramPtr, int inputIndex, int graphicsInputIndex)
{
    const int timeout = 3000;
    auto factory = m_processRegister.getProcessFactory();
    auto processPtr = factory.createObject(algoName, std::move(paramPtr));
    QVERIFY(processPtr != nullptr);
    processPtr->setInput(m_videoInputPtr, inputIndex);

    if(graphicsInputIndex != -1)
    {
        auto graphicsInputPtr = std::dynamic_pointer_cast<CGraphicsInput>(processPtr->getInput(graphicsInputIndex));
        addGraphicsRect(graphicsInputPtr);
    }

    // Set video position to the first image for processing all the video
    m_videoInputPtr->setVideoPos(0);
    // Start acquisition
    m_videoInputPtr->startVideo(timeout);

    for(size_t i=0; i<m_videoInputPtr->getVideoFrameCount() && i<m_maxFrames; ++i)
    {
        m_videoInputPtr->setFrameToRead(i);
        processPtr->run();
    }
    m_videoInputPtr->stopVideo();
}

void COcvProcessTests::runTypeImageTest(const std::string algoName, const WorkflowTaskParamPtr& paramPtr, int inputIndex)
{
    auto factory = m_processRegister.getProcessFactory();
    auto processPtr = factory.createObject(algoName, std::move(paramPtr));
    QVERIFY(processPtr != nullptr);
    auto inputPtr = std::dynamic_pointer_cast<CImageIO>(processPtr->getInput(inputIndex));
    QVERIFY(inputPtr != nullptr);

    //Apply all image types
    for(auto it=m_imgTypes.begin(); it!=m_imgTypes.end(); ++it)
    {
        CMat image = convertImage(m_image, it.key());
        if(image.empty() == false)
        {
            inputPtr->setImage(image);
            if(it.value() == true)
                processPtr->run();
            else
                QVERIFY_EXCEPTION_THROWN(processPtr->run(), CException);
        }
    }
}

void COcvProcessTests::runStdTest1(const std::string algoName, const WorkflowTaskParamPtr& paramPtr, int inputIndex, int graphicsInputIndex)
{
    try
    {
        runNoInput(algoName, paramPtr);
        runImage(algoName, paramPtr, inputIndex, -1);
        runImage(algoName, paramPtr, inputIndex, graphicsInputIndex);
        runTypeImageTest(algoName, paramPtr, inputIndex);
        runVideo(algoName, paramPtr, inputIndex, -1);
        runVideo(algoName, paramPtr, inputIndex, graphicsInputIndex);
    }
    catch(std::exception& e)
    {
        QFAIL(e.what());
    }
}

void COcvProcessTests::runOpenCL(const std::string algoName, const WorkflowTaskParamPtr &paramPtr, int inputIndex)
{
    auto factory = m_processRegister.getProcessFactory();
    auto processPtr = factory.createObject(algoName, std::move(paramPtr));
    QVERIFY(processPtr != nullptr);

    //Apply with image data
    auto inputPtr = std::dynamic_pointer_cast<CImageIO>(processPtr->getInput(inputIndex));
    QVERIFY(inputPtr != nullptr);
    inputPtr->setImage(m_image);

    if(cv::ocl::useOpenCL() == false)
        QVERIFY_EXCEPTION_THROWN(processPtr->run(), CException);
    else
        processPtr->run();
}

QTEST_GUILESS_MAIN(COcvProcessTests)
