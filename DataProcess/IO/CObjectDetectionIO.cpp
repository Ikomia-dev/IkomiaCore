#include "CObjectDetectionIO.h"
#include <QJsonArray>
#include "Main/CoreTools.hpp"
#include "IO/CInstanceSegIO.h"

//----------------------------//
//----- CObjectDetection -----//
//----------------------------//
int CObjectDetection::getId() const
{
    return m_id;
}

std::string CObjectDetection::getLabel() const
{
    return m_label;
}

double CObjectDetection::getConfidence() const
{
    return m_confidence;
}

std::vector<double> CObjectDetection::getBox() const
{
    return m_box;
}

CColor CObjectDetection::getColor() const
{
    return m_color;
}

void CObjectDetection::setId(int id)
{
    m_id = id;
}

void CObjectDetection::setLabel(const std::string &label)
{
    m_label = label;
}

void CObjectDetection::setConfidence(double confidence)
{
    m_confidence = confidence;
}

void CObjectDetection::setBox(const std::vector<double> &box)
{
    m_box = box;
}

void CObjectDetection::setColor(const CColor &color)
{
    m_color = color;
}

std::string CObjectDetection::repr() const
{
    return "CObjectDetection()";
}

std::ostream& operator<<(std::ostream& os, const CObjectDetection& obj)
{
    os << "----- Object: " << std::to_string(obj.m_id) << " -----" << std::endl;
    os << "Label: " << obj.m_label << std::endl;
    os << "Confidence: " << std::to_string(obj.m_confidence) << std::endl;

    os << "Box: [";
    for (size_t i=0; i<obj.m_box.size(); ++i)
    {
        os << std::to_string(obj.m_box[i]);
        if (i < obj.m_box.size() - 1)
            os << ", ";
    }
    os << "]" << std::endl;

    os << "Color: [" << std::to_string(obj.m_color[0]) << ", " << std::to_string(obj.m_color[0]) << ", " << std::to_string(obj.m_color[0]) << "]" << std::endl;
    return os;
}

//------------------------------//
//----- CObjectDetectionIO -----//
//------------------------------//
CObjectDetectionIO::CObjectDetectionIO() : CWorkflowTaskIO(IODataType::OBJECT_DETECTION, "ObjectDetectionIO")
{
    m_description = QObject::tr("Object detection data: label, confidence, box and color.").toStdString();
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
    CWorkflowTaskIO::operator=(io);
    m_objects = io.m_objects;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
    return *this;
}

std::string CObjectDetectionIO::repr() const
{
    std::stringstream s;
    s << "CObjectDetectionIO()";
    return s.str();
}

CObjectDetectionIO &CObjectDetectionIO::operator=(const CObjectDetectionIO &&io)
{
    CWorkflowTaskIO::operator=(io);
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

GraphicsOutputPtr CObjectDetectionIO::getGraphicsIO() const
{
    return m_graphicsIOPtr;
}

BlobMeasureIOPtr CObjectDetectionIO::getBlobMeasureIO() const
{
    return m_blobMeasureIOPtr;
}

InputOutputVect CObjectDetectionIO::getSubIOList(const std::set<IODataType> &dataTypes) const
{
    InputOutputVect ioList;

    auto it = dataTypes.find(IODataType::OUTPUT_GRAPHICS);
    if(it != dataTypes.end())
        ioList.push_back(m_graphicsIOPtr);

    it = dataTypes.find(IODataType::BLOB_VALUES);
    if(it != dataTypes.end())
        ioList.push_back(m_blobMeasureIOPtr);

    return ioList;
}

CMat CObjectDetectionIO::getImageWithGraphics(const CMat &image) const
{
    auto graphicsIOPtr = getGraphicsIO();
    if (graphicsIOPtr)
        return graphicsIOPtr->getImageWithGraphics(image);
    else
        return image;
}

CMat CObjectDetectionIO::getImageWithMaskAndGraphics(const CMat &image) const
{
    return getImageWithGraphics(image);
}

bool CObjectDetectionIO::isDataAvailable() const
{
    return m_objects.size() > 0;
}

bool CObjectDetectionIO::isComposite() const
{
    return true;
}

void CObjectDetectionIO::init(const std::string &taskName, int imageIndex)
{
    clearData();
    m_graphicsIOPtr->setNewLayer(taskName);
    m_graphicsIOPtr->setImageIndex(imageIndex);
}

void CObjectDetectionIO::addObject(int id, const std::string &label, double confidence, double boxX, double boxY, double boxWidth, double boxHeight, const CColor &color)
{
    CObjectDetection obj;
    obj.m_id = id;
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
    std::string graphicsLabel = label + " #" + std::to_string(id) + ": " + std::to_string(confidence);
    CGraphicsTextProperty textProperty;
    textProperty.m_color = color;
    textProperty.m_fontSize = 8;
    m_graphicsIOPtr->addText(graphicsLabel, boxX + 5, boxY + 5, textProperty);

    //Store values to be shown in results table
    std::vector<CObjectMeasure> results;
    results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Identifier").toStdString()), id, graphicsObj->getId(), label));
    results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), label));
    results.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {boxX, boxY, boxWidth, boxHeight}, graphicsObj->getId(), label));
    m_blobMeasureIOPtr->addObjectMeasures(results);
}

void CObjectDetectionIO::addObject(int id, const std::string &label, double confidence,
                                   double cx, double cy, double width, double height, double angle, const CColor &color)
{
    CObjectDetection obj;
    obj.m_id = id;
    obj.m_label = label;
    obj.m_confidence = confidence;
    obj.m_box = {cx, cy, width, height, angle};
    obj.m_color = color;
    m_objects.push_back(obj);

    //Set integrated I/O
    //Create polygon graphics of rotated bbox
    CGraphicsPolygonProperty polyProp;
    polyProp.m_category = label;
    polyProp.m_penColor = color;
    cv::RotatedRect box(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
    cv::Point2f vertices[4];
    box.points(vertices);

    PolygonF poly;
    for(int j=0; j<4; ++j)
        poly.push_back(CPointF(vertices[j].x, vertices[j].y));

    auto graphicsObj = m_graphicsIOPtr->addPolygon(poly);

    //Class label
    std::string graphicsLabel = label + " #" + std::to_string(id) + ": " + std::to_string(confidence);
    CGraphicsTextProperty textProperty;
    textProperty.m_color = color;
    textProperty.m_fontSize = 8;
    m_graphicsIOPtr->addText(graphicsLabel, cx + 5, cy + 5, textProperty);

    //Store values to be shown in results table
    std::vector<CObjectMeasure> results;
    results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Identifier").toStdString()), id, graphicsObj->getId(), label));
    results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), label));
    results.emplace_back(CObjectMeasure(CMeasure::Id::ORIENTED_BBOX, {cx, cy, width, height, angle}, graphicsObj->getId(), label));
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

    fromJsonInternal(jsonDoc);
}

void CObjectDetectionIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson());
}

std::string CObjectDetectionIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CObjectDetectionIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

QJsonObject CObjectDetectionIO::toJsonInternal() const
{
    QJsonArray objects;
    for (size_t i=0; i<m_objects.size(); ++i)
    {
        QJsonObject obj;
        obj["id"] = m_objects[i].m_id;
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
    root["referenceImageIndex"] = m_graphicsIOPtr->getImageIndex();
    return root;
}

void CObjectDetectionIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading object detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

void CObjectDetectionIO::fromJsonInternal(const QJsonDocument &jsonDoc)
{
    QJsonObject root = jsonDoc.object();
    QJsonArray objects = root["detections"].toArray();
    clearData();

    for (int i=0; i<objects.size(); ++i)
    {
        QJsonObject obj = objects[i].toObject();
        int id = obj["id"].toInt();
        std::string label = obj["label"].toString().toStdString();
        double confidence = obj["confidence"].toDouble();
        QJsonObject box = obj["box"].toObject();
        double x = box["x"].toDouble();
        double y = box["y"].toDouble();
        double width = box["width"].toDouble();
        double height = box["height"].toDouble();
        CColor color = Utils::Graphics::colorFromJson(obj["color"].toObject());
        addObject(id, label, confidence, x, y, width, height, color);
    }
}

std::shared_ptr<CObjectDetectionIO> CObjectDetectionIO::clone() const
{
    return std::static_pointer_cast<CObjectDetectionIO>(cloneImp());
}

void CObjectDetectionIO::copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    auto type = ioPtr->getDataType();
    if (type == IODataType::OBJECT_DETECTION)
    {
        //Should not be called in this case
        auto pIO = dynamic_cast<const CObjectDetectionIO*>(ioPtr.get());
        if(pIO)
            *this = *pIO;
    }
    else if (type == IODataType::INSTANCE_SEGMENTATION)
    {
        auto instanceIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(ioPtr);
        if (instanceIOPtr)
        {
            clearData();
            std::vector<CInstanceSegmentation> objects = instanceIOPtr->getObjects();

            for (size_t i=0; i<objects.size(); ++i)
            {
                addObject(objects[i].m_id, objects[i].m_label, objects[i].m_confidence,
                          objects[i].m_box[0], objects[i].m_box[1], objects[i].m_box[2], objects[i].m_box[3],
                          objects[i].m_color);
            }
        }
    }
}

WorkflowTaskIOPtr CObjectDetectionIO::cloneImp() const
{
    return std::shared_ptr<CObjectDetectionIO>(new CObjectDetectionIO(*this));
}
