#include "CObjectDetectionIO.h"
#include "Main/CoreTools.hpp"
#include <QJsonArray>

CObjectDetectionIO::CObjectDetectionIO() : CWorkflowTaskIO(IODataType::OBJECT_DETECTION, "CObjectDetectionIO")
{
    m_description = QObject::tr("Object detection data: label, confidence, box and color.\n").toStdString();
    m_saveFormat = DataFileFormat::JSON;
    m_graphicsIOPtr = std::make_shared<CGraphicsOutput>();
    m_blobMeasureIOPtr = std::make_shared<CBlobMeasureIO>();
}

CObjectDetectionIO::CObjectDetectionIO(const CObjectDetectionIO &io): CWorkflowTaskIO(io)
{
    m_objects = io.m_objects;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
}

CObjectDetectionIO::CObjectDetectionIO(const CObjectDetectionIO &&io): CWorkflowTaskIO(io)
{
    m_objects = std::move(io.m_objects);
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
}

CObjectDetectionIO &CObjectDetectionIO::operator=(const CObjectDetectionIO &io)
{
    m_objects = io.m_objects;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
    return *this;
}

CObjectDetectionIO &CObjectDetectionIO::operator=(const CObjectDetectionIO &&io)
{
    m_objects = std::move(io.m_objects);
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
    return *this;
}

size_t CObjectDetectionIO::getObjectCount() const
{
    return m_objects.size();
}

CObjectDetection CObjectDetectionIO::getObject(size_t index) const
{
    if (index >= m_objects.size())
        throw CException(CoreExCode::INDEX_OVERFLOW, "No object detection at given index: index overflow", __func__, __FILE__, __LINE__);

    return m_objects[index];
}

std::vector<CObjectDetection> CObjectDetectionIO::getObjects() const
{
    return m_objects;
}

CDataInfoPtr CObjectDetectionIO::getDataInfo()
{
    if (m_infoPtr == nullptr)
    {
        m_infoPtr = std::make_shared<CDataInfo>(this->m_dataType);
        m_infoPtr->metadata().insert(std::make_pair("Object count", std::to_string(m_objects.size())));
    }
    return m_infoPtr;
}

std::shared_ptr<CGraphicsOutput> CObjectDetectionIO::getGraphicsIO() const
{
    return m_graphicsIOPtr;
}

std::shared_ptr<CBlobMeasureIO> CObjectDetectionIO::getBlobMeasureIO() const
{
    return m_blobMeasureIOPtr;
}

bool CObjectDetectionIO::isDataAvailable() const
{
    return m_objects.size() > 0;
}

void CObjectDetectionIO::init(const std::string &taskName, int imageIndex)
{
    clearData();
    m_graphicsIOPtr->setNewLayer(taskName);
    m_graphicsIOPtr->setImageIndex(imageIndex);
}

void CObjectDetectionIO::addObject(const std::string &label, double confidence, double boxX, double boxY, double boxWidth, double boxHeight, const CColor &color)
{
    CObjectDetection obj;
    obj.m_label = label;
    obj.m_confidence = confidence;
    obj.m_box = {boxX, boxY, boxWidth, boxHeight};
    obj.m_color = color;
    m_objects.push_back(obj);

    //Set integrated I/O
    //Create rectangle graphics of bbox
    CGraphicsRectProperty rectProp;
    rectProp.m_category = label;
    rectProp.m_penColor = color;
    auto graphicsObj = m_graphicsIOPtr->addRectangle(boxX, boxY, boxWidth, boxHeight, rectProp);

    //Class label
    std::string graphicsLabel = label + " : " + std::to_string(confidence);
    CGraphicsTextProperty textProperty;
    textProperty.m_color = color;
    textProperty.m_fontSize = 8;
    m_graphicsIOPtr->addText(graphicsLabel, boxX + 5, boxY + 5, textProperty);

    //Store values to be shown in results table
    std::vector<CObjectMeasure> results;
    results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), label));
    results.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {boxX, boxY, boxWidth, boxHeight}, graphicsObj->getId(), label));
    m_blobMeasureIOPtr->addObjectMeasures(results);
}

void CObjectDetectionIO::clearData()
{
    m_objects.clear();
    m_graphicsIOPtr->clearData();
    m_blobMeasureIOPtr->clearData();
    m_infoPtr = nullptr;
}

void CObjectDetectionIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonFile.readAll()));
    if(jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading object detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJson(jsonDoc);
}

void CObjectDetectionIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJson());
    jsonFile.write(jsonDoc.toJson());
}

std::string CObjectDetectionIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJson());
    return toFormattedJson(doc, options);
}

QJsonObject CObjectDetectionIO::toJson() const
{
    QJsonArray objects;
    for (size_t i=0; i<m_objects.size(); ++i)
    {
        QJsonObject obj;
        obj["label"] = QString::fromStdString(m_objects[i].m_label);
        obj["confidence"] = m_objects[i].m_confidence;

        QJsonObject box;
        box["x"] = m_objects[i].m_box[0];
        box["y"] = m_objects[i].m_box[1];
        box["width"] = m_objects[i].m_box[2];
        box["height"] = m_objects[i].m_box[3];
        obj["box"] = box;

        obj["color"] = CGraphicsJSON::toJsonObject(m_objects[i].m_color);
        objects.append(obj);
    }
    QJsonObject root;
    root["detections"] = objects;
    return root;
}

void CObjectDetectionIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading object detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJson(jsonDoc);
}

void CObjectDetectionIO::fromJson(const QJsonDocument &jsonDoc)
{
    QJsonObject root = jsonDoc.object();
    QJsonArray objects = root["detections"].toArray();
    m_objects.clear();

    for (int i=0; i<objects.size(); ++i)
    {
        CObjectDetection objDetection;
        QJsonObject obj = objects[i].toObject();
        objDetection.m_label = obj["label"].toString().toStdString();
        objDetection.m_confidence = obj["confidence"].toDouble();
        QJsonObject box = obj["box"].toObject();
        objDetection.m_box = {box["x"].toDouble(), box["y"].toDouble(), box["width"].toDouble(), box["height"].toDouble()};
        objDetection.m_color = Utils::Graphics::colorFromJson(obj["color"].toObject());
        m_objects.push_back(objDetection);
    }
}

std::shared_ptr<CObjectDetectionIO> CObjectDetectionIO::clone() const
{
    return std::static_pointer_cast<CObjectDetectionIO>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CObjectDetectionIO::cloneImp() const
{
    return std::shared_ptr<CObjectDetectionIO>(new CObjectDetectionIO(*this));
}
