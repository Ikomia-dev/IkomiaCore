#include <QtTest>
#include "CIOTests.h"
#include "UnitTestUtils.hpp"

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

    std::string path = UnitTest::getDataPath() + "/blobMeasureIO.csv";
    blobIO.save(path);
    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::blobMeasureIOLoad()
{
    const int objCount = 5;
    CBlobMeasureIO blobIO;
    fillBlobMeasureIO(blobIO);
    std::string path = UnitTest::getDataPath() + "/blobMeasureIO.csv";
    blobIO.save(path);

    CBlobMeasureIO blobIOLoad;
    blobIOLoad.load(path);

    auto measures = blobIOLoad.getMeasures();
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

    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
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

    std::string path = UnitTest::getDataPath() + "/graphicsInput.json";
    graphicsIn.save(path);
    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::graphicsInputLoad()
{
    CGraphicsInput graphicsIn;
    graphicsIn.setItems(createGraphics());
    std::string path = UnitTest::getDataPath() + "/graphicsInput.json";
    graphicsIn.save(path);

    CGraphicsInput graphicsInLoad;
    graphicsInLoad.load(path);
    auto items = graphicsInLoad.getItems();
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

    std::string path = UnitTest::getDataPath() + "/graphicsInput.json";
    graphicsOut.save(path);
    QVERIFY(boost::filesystem::exists(path));
    boost::filesystem::path boostPath(path);
    boost::filesystem::remove(boostPath);
}

void CIOTests::graphicsOutputLoad()
{
    CGraphicsOutput graphicsOut;
    graphicsOut.setItems(createGraphics());
    std::string path = UnitTest::getDataPath() + "/graphicsInput.json";
    graphicsOut.save(path);

    CGraphicsOutput graphicsOutLoad;
    graphicsOutLoad.load(path);
    auto items = graphicsOutLoad.getItems();
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

QTEST_GUILESS_MAIN(CIOTests)
