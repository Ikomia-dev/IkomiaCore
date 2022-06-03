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
    auto blobIO = CBlobMeasureIO();
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
    auto blobIO = CBlobMeasureIO();
    fillBlobMeasureIO(blobIO);
    std::string path = UnitTest::getDataPath() + "/blobMeasureIO.csv";
    blobIO.save(path);

    auto blobIOLoad = CBlobMeasureIO();
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

QTEST_GUILESS_MAIN(CIOTests)
