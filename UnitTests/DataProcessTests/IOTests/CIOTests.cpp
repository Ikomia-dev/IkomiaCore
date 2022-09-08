#include <QtTest>
#include "CIOTests.h"
#include "UnitTestUtils.hpp"
#include "IO/CObjectDetectionIO.h"
#include "IO/CInstanceSegIO.h"

CIOTests::CIOTests(QObject *parent) : QObject(parent)
{
}

void CIOTests::initTestCase()
{
}

void CIOTests::blobMeasureIOSave()
{
    CBlobMeasureIO blobIO;
    fillBlobMeasureIO(blobIO);

    std::string path = UnitTest::getDataPath() + "/IO/blobMeasureIOTmp.csv";
    blobIO.save(path);
    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::blobMeasureIOLoad()
{
    const int objCount = 5;
    CBlobMeasureIO blobIO;
    std::string path = UnitTest::getDataPath() + "/IO/blobMeasureIO.csv";
    blobIO.load(path);

    auto measures = blobIO.getMeasures();
    QVERIFY(measures.size() == objCount);
    QVERIFY(measures[0].size() == 2);
    auto measure1 = measures[0][0].getMeasureInfo();
    auto measure2 = measures[0][1].getMeasureInfo();
    QVERIFY(measure1.m_id == CMeasure::BBOX);
    QVERIFY(measure2.m_id == CMeasure::CUSTOM);
    QVERIFY(measure2.m_name == "Confidence");
    auto values = measures[0][0].getValues();
    QVERIFY(values.size() == 4);
    values = measures[0][1].getValues();
    QVERIFY(values.size() == 1);
    QVERIFY(values[0] == 0.1);
}

void CIOTests::blobMeasureIOToJson()
{
    CBlobMeasureIO blobIO;
    fillBlobMeasureIO(blobIO);
    std::string jsonStr = blobIO.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());
}

void CIOTests::blobMeasureIOFromJson()
{
    const int objCount = 5;
    std::string path = UnitTest::getDataPath() + "/IO/blobMeasureIO.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CBlobMeasureIO blobIO;
    blobIO.fromJson(jsonStr);

    auto measures = blobIO.getMeasures();
    QVERIFY(measures.size() == objCount);
    QVERIFY(measures[0].size() == 2);

    auto measure1 = measures[0][0].getMeasureInfo();
    QVERIFY(measure1.m_id == CMeasure::CUSTOM);
    QVERIFY(measure1.m_name == "Confidence");
    auto values = measures[0][0].getValues();
    QVERIFY(values.size() == 1);
    QVERIFY(values[0] == 0.1);

    auto measure2 = measures[0][1].getMeasureInfo();
    QVERIFY(measure2.m_id == CMeasure::BBOX);
    values = measures[0][1].getValues();
    QVERIFY(values.size() == 4);
}

void CIOTests::fillBlobMeasureIO(CBlobMeasureIO &io)
{
    const int objCount = 5;
    std::vector<float> confidences = { 0.1, 0.9, 0.5, 0.8, 0.6};
    std::vector<std::string> labels = { "Dog", "Cat", "Lion", "Mouse", "Snake"};

    for (int i=0; i<objCount; ++i)
    {
        std::vector<CObjectMeasure> results;
        results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, "Confidence"), confidences[i], i, labels[i]));
        results.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {0, 0, 200, 300}, i, labels[i]));
        io.addObjectMeasures(results);
    }
}

void CIOTests::graphicsInputSave()
{
    CGraphicsInput graphicsIn;
    graphicsIn.setItems(createGraphics());

    std::string path = UnitTest::getDataPath() + "/graphicsInputTmp.json";
    graphicsIn.save(path);
    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::graphicsInputLoad()
{
    CGraphicsInput graphicsIn;
    std::string path = UnitTest::getDataPath() + "/IO/graphicsInput.json";
    graphicsIn.load(path);
    auto items = graphicsIn.getItems();
    QVERIFY(items.size() == 7);
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsEllipse>(items[0]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsRect>(items[1]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolyline>(items[2]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolygon>(items[3]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsComplexPoly>(items[4]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPoint>(items[5]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsText>(items[6]));
}

void CIOTests::graphicsInputToJson()
{
    CGraphicsInput graphicsIn;
    graphicsIn.setItems(createGraphics());
    std::string jsonStr = graphicsIn.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());
}

void CIOTests::graphicsInputFromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/graphicsInput.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CGraphicsInput graphicsIn;
    graphicsIn.fromJson(jsonStr);

    auto items = graphicsIn.getItems();
    QVERIFY(items.size() == 7);
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsEllipse>(items[0]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsRect>(items[1]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolyline>(items[2]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolygon>(items[3]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsComplexPoly>(items[4]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPoint>(items[5]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsText>(items[6]));
}

void CIOTests::graphicsOutputSave()
{
    CGraphicsOutput graphicsOut;
    graphicsOut.setItems(createGraphics());

    std::string path = UnitTest::getDataPath() + "/IO/graphicsOutputTmp.json";
    graphicsOut.save(path);
    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::graphicsOutputLoad()
{
    CGraphicsOutput graphicsOut;
    std::string path = UnitTest::getDataPath() + "/IO/graphicsOutput.json";
    graphicsOut.load(path);

    auto items = graphicsOut.getItems();
    QVERIFY(items.size() == 7);
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsEllipse>(items[0]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsRect>(items[1]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolyline>(items[2]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolygon>(items[3]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsComplexPoly>(items[4]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPoint>(items[5]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsText>(items[6]));
}

void CIOTests::graphicsOutputToJson()
{
    CGraphicsInput graphicsOut;
    graphicsOut.setItems(createGraphics());
    std::string jsonStr = graphicsOut.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());
}

void CIOTests::graphicsOutputFromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/graphicsInput.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CGraphicsInput graphicsOut;
    graphicsOut.fromJson(jsonStr);

    auto items = graphicsOut.getItems();
    QVERIFY(items.size() == 7);
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsEllipse>(items[0]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsRect>(items[1]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolyline>(items[2]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPolygon>(items[3]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsComplexPoly>(items[4]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsPoint>(items[5]));
    QVERIFY(std::dynamic_pointer_cast<CProxyGraphicsText>(items[6]));
}

std::vector<ProxyGraphicsItemPtr> CIOTests::createGraphics()
{
    std::vector<ProxyGraphicsItemPtr> items;

    //Ellipse
    auto ellipse = std::make_shared<CProxyGraphicsEllipse>(10, 10, 50, 75);
    items.push_back(ellipse);

    //Rectangle
    auto rectangle = std::make_shared<CProxyGraphicsRect>(120, 10, 50, 50);
    items.push_back(rectangle);

    //Polyline
    std::vector<CPointF> pts1 = {CPointF(300, 30), CPointF(500, 600), CPointF(700, 100)};
    auto polyline = std::make_shared<CProxyGraphicsPolyline>(pts1);
    items.push_back(polyline);

    //Polygon
    std::vector<CPointF> pts2 = {CPointF(50, 350), CPointF(150, 450), CPointF(200, 700), CPointF(50, 750), CPointF(30, 500)};
    auto polygon = std::make_shared<CProxyGraphicsPolygon>(pts2);
    items.push_back(polygon);

    //Complex polygon
    PolygonF outer = {CPointF(750, 350), CPointF(850, 450), CPointF(900, 700), CPointF(800, 750), CPointF(730, 450)};
    std::vector<PolygonF> inners = {{CPointF(750, 400), CPointF(800, 700), CPointF(810, 450)}};
    auto complexPoly = std::make_shared<CProxyGraphicsComplexPoly>(outer, inners);
    items.push_back(complexPoly);

    //Point
    auto point = std::make_shared<CProxyGraphicsPoint>(CPointF(160, 120));
    items.push_back(point);

    //Text
    auto text = std::make_shared<CProxyGraphicsText>("Test", 100, 100);
    items.push_back(text);

    return items;
}

void CIOTests::numericIODoubleSave()
{
    CNumericIO<double> numericIO;
    fillNumericIO(numericIO);
    std::string path = UnitTest::getDataPath() + "/IO/numericIOTmp.csv";
    numericIO.save(path);

    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::numericIODoubleLoad()
{
    CNumericIO<double> numericIO;
    std::string path = UnitTest::getDataPath() + "/IO/numericIODouble.csv";
    numericIO.load(path);

    QVERIFY(numericIO.getOutputType() == NumericOutputType::TABLE);
    QVERIFY(numericIO.getPlotType() == PlotType::CURVE);
    QVERIFY(numericIO.getAllHeaderLabels().size() == 4);
    QVERIFY(numericIO.getAllValueLabels().size() == 0);
    auto values = numericIO.getAllValues();
    QVERIFY(values.size() == 4);

    for (size_t i=0; i<values.size(); ++i)
        QVERIFY(values[i].size() == 6);
}

void CIOTests::numericIODoubleToJson()
{
    CNumericIO<double> numericIO;
    fillNumericIO(numericIO);
    std::string jsonStr = numericIO.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());
}

void CIOTests::numericIODoubleFromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/numericIODouble.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CNumericIO<double> numericIO;
    numericIO.fromJson(jsonStr);

    QVERIFY(numericIO.getOutputType() == NumericOutputType::TABLE);
    QVERIFY(numericIO.getPlotType() == PlotType::CURVE);
    QVERIFY(numericIO.getAllHeaderLabels().size() == 4);
    QVERIFY(numericIO.getAllValueLabels().size() == 0);
    auto values = numericIO.getAllValues();
    QVERIFY(values.size() == 4);

    for (size_t i=0; i<values.size(); ++i)
        QVERIFY(values[i].size() == 6);
}

void CIOTests::fillNumericIO(CNumericIO<double> &io)
{
    std::vector<std::string> headerLabels = {"id", "surface", "perimeter", "diameter"};
    std::vector<double> ids = {1, 2, 3, 4, 5, 6};
    std::vector<double> surfaces = {17.2, 33, 5.6, 107, 89.8, 0.2};
    std::vector<double> perimeters = {21, 28.5, 10, 221, 79.5, 1.1};
    std::vector<double> diameters = {6, 4, 2, 33, 14, 0.1};

    io.addValueList(ids, headerLabels[0]);
    io.addValueList(surfaces, headerLabels[1]);
    io.addValueList(perimeters, headerLabels[2]);
    io.addValueList(diameters, headerLabels[3]);
}

void CIOTests::numericIOStringSave()
{
    CNumericIO<std::string> numericIO;
    fillNumericIO(numericIO);
    std::string path = UnitTest::getDataPath() + "/IO/numericIOTmp.csv";
    numericIO.save(path);

    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::numericIOStringLoad()
{
    CNumericIO<std::string> numericIO;
    std::string path = UnitTest::getDataPath() + "/IO/numericIOString.csv";
    numericIO.load(path);

    QVERIFY(numericIO.getOutputType() == NumericOutputType::TABLE);
    QVERIFY(numericIO.getPlotType() == PlotType::CURVE);
    QVERIFY(numericIO.getAllHeaderLabels().size() == 4);
    QVERIFY(numericIO.getAllValueLabels().size() == 0);
    auto values = numericIO.getAllValues();
    QVERIFY(values.size() == 4);

    for (size_t i=0; i<values.size(); ++i)
        QVERIFY(values[i].size() == 6);
}

void CIOTests::numericIOStringToJson()
{
    CNumericIO<std::string> numericIO;
    fillNumericIO(numericIO);
    std::string jsonStr = numericIO.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());
}

void CIOTests::numericIOStringFromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/numericIOString.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CNumericIO<std::string> numericIO;
    numericIO.fromJson(jsonStr);

    QVERIFY(numericIO.getOutputType() == NumericOutputType::TABLE);
    QVERIFY(numericIO.getPlotType() == PlotType::CURVE);
    QVERIFY(numericIO.getAllHeaderLabels().size() == 4);
    QVERIFY(numericIO.getAllValueLabels().size() == 0);
    auto values = numericIO.getAllValues();
    QVERIFY(values.size() == 4);

    for (size_t i=0; i<values.size(); ++i)
        QVERIFY(values[i].size() == 6);
}

void CIOTests::imageIOToJson()
{
    std::string path = UnitTest::getDataPath() + "/Images/Lena.png";
    CImageIO imgIO(IODataType::IMAGE, "ColorImage", path);
    std::vector<std::string> options = {"image_format", "jpg"};
    std::string jsonStr = imgIO.toJson(options);
    QVERIFY(!jsonStr.empty());

    options = {"format", "png"};
    jsonStr = imgIO.toJson(options);
    QVERIFY(!jsonStr.empty());

//    QFile file(QString::fromStdString(UnitTest::getDataPath() + "/IO/imageIO.json"));
//    file.open(QFile::WriteOnly);
//    file.write(QString::fromStdString(jsonStr).toUtf8());
//    file.close();
}

void CIOTests::imageIOFromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/imageIOjpg.json";
    QFile jsonFileJpg(QString::fromStdString(path));
    jsonFileJpg.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFileJpg.readAll()).toStdString();

    CImageIO imgIO;
    imgIO.fromJson(jsonStr);

    CMat img = imgIO.getImage();
    QVERIFY(img.data != nullptr);
    QVERIFY(img.getNbRows() == 512);
    QVERIFY(img.getNbCols() == 512);
    QVERIFY(img.channels() == 3);

    //imgIO.save(UnitTest::getDataPath() + "/IO/imageJpg.png");

    path = UnitTest::getDataPath() + "/IO/imageIOpng.json";
    QFile jsonFilePng(QString::fromStdString(path));
    jsonFilePng.open(QFile::ReadOnly | QFile::Text);
    jsonStr = QString(jsonFilePng.readAll()).toStdString();

    imgIO.fromJson(jsonStr);

    img = imgIO.getImage();
    QVERIFY(img.data != nullptr);
    QVERIFY(img.getNbRows() == 512);
    QVERIFY(img.getNbCols() == 512);
    QVERIFY(img.channels() == 3);

    //imgIO.save(UnitTest::getDataPath() + "/IO/imagePng.png");
}

void CIOTests::objDetectIOToJson()
{
    CObjectDetectionIO io;
    fillObjectDetectionIO(io);
    std::string jsonStr = io.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());

//    QFile file(QString::fromStdString(UnitTest::getDataPath() + "/IO/objectDetectionIO.json"));
//    file.open(QFile::WriteOnly);
//    file.write(QString::fromStdString(jsonStr).toUtf8());
//    file.close();
}

void CIOTests::objDetectIOfromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/objectDetectionIO.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CObjectDetectionIO io;
    io.fromJson(jsonStr);

    QVERIFY(io.isDataAvailable());
    QVERIFY(io.getObjectCount() == 5);
    QVERIFY(io.getGraphicsIO() != nullptr);
    QVERIFY(io.getBlobMeasureIO() != nullptr);

    auto objects = io.getObjects();
    QVERIFY(objects.size() == io.getObjectCount());

    for (size_t i=0; i<objects.size(); ++i)
    {
        auto obj = io.getObject(i);
        QVERIFY(obj.m_box.size() == 4);
        QVERIFY(obj.m_label.empty() == false);
        QVERIFY(obj.m_confidence > 0);
        QVERIFY(obj.m_color.size() == 4);
    }
}

void CIOTests::instanceSegIOToJson()
{
    CInstanceSegIO io;
    fillInstanceSegIO(io);
    std::string jsonStr = io.toJson(std::vector<std::string>());
    QVERIFY(!jsonStr.empty());

    QFile file(QString::fromStdString(UnitTest::getDataPath() + "/IO/instanceSegIO.json"));
    file.open(QFile::WriteOnly);
    file.write(QString::fromStdString(jsonStr).toUtf8());
    file.close();
}

void CIOTests::instanceSegIOFromJson()
{
    std::string path = UnitTest::getDataPath() + "/IO/instanceSegIO.json";
    QFile jsonFile(QString::fromStdString(path));
    jsonFile.open(QFile::ReadOnly | QFile::Text);
    std::string jsonStr = QString(jsonFile.readAll()).toStdString();

    CInstanceSegIO io;
    io.fromJson(jsonStr);

    QVERIFY(io.isDataAvailable());
    QVERIFY(io.getInstanceCount() == 3);
    QVERIFY(io.getMaskImageIO() != nullptr);
    QVERIFY(io.getGraphicsIO() != nullptr);
    QVERIFY(io.getBlobMeasureIO() != nullptr);
    QVERIFY(io.getMergeMask().empty() == false);

    auto instances = io.getInstances();
    QVERIFY(instances.size() == io.getInstanceCount());

    for (size_t i=0; i<instances.size(); ++i)
    {
        auto inst = io.getInstance(i);
        QVERIFY(inst.m_box.size() == 4);
        QVERIFY(inst.m_label.empty() == false);
        QVERIFY(inst.m_confidence > 0);
        QVERIFY(inst.m_color.size() == 4);
        QVERIFY(inst.m_classIndex >= 0);
        QVERIFY(inst.m_mask.empty() == false);
    }
}

void CIOTests::fillNumericIO(CNumericIO<std::string> &io)
{
    std::vector<std::string> headerLabels = {"name", "category", "description", "date"};
    std::vector<std::string> names = {"titi", "tata", "tutu", "toto", "toutou", "toitoi"};
    std::vector<std::string> categories = {"cat", "dog", "lion", "elefant", "tiger", "horse"};
    std::vector<std::string> descriptions = {"animal", "animal", "sauvage animal", "sauvage animal", "sauvage animal", "animal"};
    std::vector<std::string> dates = {"10/10/2020", "02/08/1999", "23/05/2011", "17/03/2018", "09/12/2000", "14/07/2021"};

    io.addValueList(names, headerLabels[0]);
    io.addValueList(categories, headerLabels[1]);
    io.addValueList(descriptions, headerLabels[2]);
    io.addValueList(dates, headerLabels[3]);
}

void CIOTests::fillObjectDetectionIO(CObjectDetectionIO &io)
{
    io.init("YOLO", 0);
    io.addObject("Object1", 0.75, 5, 10, 125, 150, {255, 0, 0});
    io.addObject("Object2", 0.25, 30, 100, 75, 50, {0, 255, 0});
    io.addObject("Object3", 0.99, 120, 10, 50, 50, {0, 0, 255});
    io.addObject("Object4", 0.80, 5, 100, 125, 150, {255, 255, 0});
    io.addObject("Object5", 0.60, 250, 100, 80, 100, {255, 0, 255});
}

void CIOTests::fillInstanceSegIO(CInstanceSegIO &io)
{
    std::string path = UnitTest::getDataPath() + "/Images/Lena.png";
    auto srcImg = cv::imread(path);
    cv::cvtColor(srcImg, srcImg, cv::COLOR_RGB2GRAY);
    io.init("MaskRCNN", 0, srcImg.cols, srcImg.rows);

    cv::Rect rc1(150, 150, 100, 100);
    cv::Mat mask1(srcImg.rows, srcImg.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat binImg1 = srcImg > 50;
    cv::Mat roiBinImg1(binImg1, rc1);
    cv::Mat roiMask1(mask1, rc1);
    roiBinImg1.copyTo(roiMask1);
    io.addInstance(CInstanceSegmentation::ObjectType::THING, 0, "Object1", 0.85, rc1.x, rc1.y, rc1.width, rc1.height, mask1, {255,0,0});

    cv::Rect rc2(200, 100, 75, 150);
    cv::Mat mask2(srcImg.rows, srcImg.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat binImg2 = srcImg > 100;
    cv::Mat roiBinImg2(binImg2, rc2);
    cv::Mat roiMask2(mask2, rc2);
    roiBinImg2.copyTo(roiMask2);
    io.addInstance(CInstanceSegmentation::ObjectType::THING, 1, "Object2", 0.60, rc2.x, rc2.y, rc2.width, rc2.height, mask2, {0,255,0});

    cv::Rect rc3(150, 250, 150, 80);
    cv::Mat mask3(srcImg.rows, srcImg.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat binImg3 = srcImg > 150;
    cv::Mat roiBinImg3(binImg3, rc3);
    cv::Mat roiMask3(mask3, rc3);
    roiBinImg3.copyTo(roiMask3);
    io.addInstance(CInstanceSegmentation::ObjectType::THING, 0, "Object3", 0.45, rc3.x, rc3.y, rc3.width, rc3.height, mask3, {255,0,0});
}

QTEST_GUILESS_MAIN(CIOTests)
