#include "CInstanceSegIO.h"
#include "Data/CDataConversion.h"
#include "Main/CoreTools.hpp"
#include "DataProcessTools.hpp"
#include <QJsonArray>

CInstanceSegIO::CInstanceSegIO() : CWorkflowTaskIO(IODataType::INSTANCE_SEGMENTATION, "CInstanceSegIO")
{
    m_description = QObject::tr("Instance segmentation data: label, confidence, box, mask and color.\n").toStdString();
    m_saveFormat = DataFileFormat::JSON;
    m_imgIOPtr = std::make_shared<CImageIO>();
    m_graphicsIOPtr = std::make_shared<CGraphicsOutput>();
    m_blobMeasureIOPtr = std::make_shared<CBlobMeasureIO>();
}

CInstanceSegIO::CInstanceSegIO(const CInstanceSegIO &io): CWorkflowTaskIO(io)
{
    m_instances = io.m_instances;
    m_mergeMask = io.m_mergeMask;
    m_imgIOPtr = io.m_imgIOPtr->clone();
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
}

CInstanceSegIO::CInstanceSegIO(const CInstanceSegIO &&io): CWorkflowTaskIO(io)
{
    m_instances = std::move(io.m_instances);
    m_mergeMask = std::move(io.m_mergeMask);
    m_imgIOPtr = io.m_imgIOPtr->clone();
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
}

CInstanceSegIO &CInstanceSegIO::operator=(const CInstanceSegIO &io)
{
    m_instances = io.m_instances;
    m_mergeMask = io.m_mergeMask;
    m_imgIOPtr = io.m_imgIOPtr->clone();
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
    return *this;
}

CInstanceSegIO &CInstanceSegIO::operator=(const CInstanceSegIO &&io)
{
    m_instances = std::move(io.m_instances);
    m_mergeMask = std::move(io.m_mergeMask);
    m_imgIOPtr = io.m_imgIOPtr->clone();
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_blobMeasureIOPtr = io.m_blobMeasureIOPtr->clone();
    return *this;
}

size_t CInstanceSegIO::getInstanceCount() const
{
    return m_instances.size();
}

CInstanceSegmentation CInstanceSegIO::getInstance(size_t index) const
{
    if (index >= m_instances.size())
        throw CException(CoreExCode::INDEX_OVERFLOW, "No instance segmentation at given index: index overflow", __func__, __FILE__, __LINE__);

    return m_instances[index];
}

std::vector<CInstanceSegmentation> CInstanceSegIO::getInstances() const
{
    return m_instances;
}

CDataInfoPtr CInstanceSegIO::getDataInfo()
{
    if (m_infoPtr == nullptr)
    {
        m_infoPtr = std::make_shared<CDataInfo>(this->m_dataType);
        m_infoPtr->metadata().insert(std::make_pair("Object count", std::to_string(m_instances.size())));
    }
    return m_infoPtr;
}

std::shared_ptr<CImageIO> CInstanceSegIO::getMaskImageIO() const
{
    return m_imgIOPtr;
}

std::shared_ptr<CGraphicsOutput> CInstanceSegIO::getGraphicsIO() const
{
    return m_graphicsIOPtr;
}

std::shared_ptr<CBlobMeasureIO> CInstanceSegIO::getBlobMeasureIO() const
{
    return m_blobMeasureIOPtr;
}

CMat CInstanceSegIO::getMergeMask() const
{
    return m_imgIOPtr->getImage();
}

bool CInstanceSegIO::isDataAvailable() const
{
    return m_instances.size() > 0;
}

void CInstanceSegIO::init(const std::string &taskName, int refImageIndex, int imageWidth, int imageHeight)
{
    clearData();
    m_graphicsIOPtr->setNewLayer(taskName);
    m_graphicsIOPtr->setImageIndex(refImageIndex);

    cv::Mat mergeMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    m_imgIOPtr->setImage(mergeMask);
}

void CInstanceSegIO::addObject(int classIndex, const std::string &label,
                               double confidence, double boxX, double boxY, double boxWidth, double boxHeight,
                               const CMat &mask, const CColor &color)
{
    CInstanceSegmentation obj;
    obj.m_classIndex = classIndex;
    obj.m_label = label;
    obj.m_confidence = confidence;
    obj.m_box = {boxX, boxY, boxWidth, boxHeight};
    obj.m_color = color;

    if (mask.depth() !=  CV_8U)
        CDataConversion::to8Bits(mask, obj.m_mask);
    else
        obj.m_mask = mask.clone();

    m_instances.push_back(obj);

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

    //Merge mask
    auto mergeMask = m_imgIOPtr->getImage();
    //Create label mask according to the object index (we do index + 1 because 0 is the background label)
    cv::Mat labelImg(boxHeight, boxWidth, CV_8UC1, cv::Scalar(obj.m_classIndex + 1));
    cv::Mat labelMask(boxHeight, boxWidth, CV_8UC1, cv::Scalar(0));
    cv::Mat roiMask(obj.m_mask, cv::Rect(boxX, boxY, boxWidth, boxHeight));
    labelImg.copyTo(labelMask, roiMask);
    cv::Mat roiMerge(mergeMask, cv::Rect(boxX, boxY, boxWidth, boxHeight));
    cv::bitwise_or(labelMask, roiMerge, roiMerge);
    m_imgIOPtr->setImage(mergeMask);
}

void CInstanceSegIO::clearData()
{
    m_instances.clear();
    m_imgIOPtr->clearData();
    m_graphicsIOPtr->clearData();
    m_blobMeasureIOPtr->clearData();
    m_infoPtr = nullptr;
}

void CInstanceSegIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonFile.readAll()));
    if(jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading instance segmentation: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJson(jsonDoc);
}

void CInstanceSegIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal({"image_format", "jpg"}));
    jsonFile.write(jsonDoc.toJson());
}

std::string CInstanceSegIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal(options));
    return toFormattedJson(doc, options);
}

void CInstanceSegIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading object detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJson(jsonDoc);
}

std::shared_ptr<CInstanceSegIO> CInstanceSegIO::clone() const
{
    return std::static_pointer_cast<CInstanceSegIO>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CInstanceSegIO::cloneImp() const
{
    return std::shared_ptr<CInstanceSegIO>(new CInstanceSegIO(*this));
}

QJsonObject CInstanceSegIO::toJsonInternal(const std::vector<std::string> &options) const
{
    QJsonArray objects;
    for (size_t i=0; i<m_instances.size(); ++i)
    {
        QJsonObject obj;
        obj["label"] = QString::fromStdString(m_instances[i].m_label);
        obj["confidence"] = m_instances[i].m_confidence;

        QJsonObject box;
        box["x"] = m_instances[i].m_box[0];
        box["y"] = m_instances[i].m_box[1];
        box["width"] = m_instances[i].m_box[2];
        box["height"] = m_instances[i].m_box[3];
        obj["box"] = box;

        obj["mask"] = QString::fromStdString(Utils::Image::toJson(m_instances[i].m_mask, options));
        obj["color"] = CGraphicsJSON::toJsonObject(m_instances[i].m_color);
        objects.append(obj);
    }
    QJsonObject root;
    root["detections"] = objects;
    return root;
}

void CInstanceSegIO::fromJson(const QJsonDocument &doc)
{
    QJsonObject root = doc.object();
    QJsonArray instances = root["detections"].toArray();
    m_instances.clear();

    for (int i=0; i<instances.size(); ++i)
    {
        CInstanceSegmentation instance;
        QJsonObject obj = instances[i].toObject();
        instance.m_label = obj["label"].toString().toStdString();
        instance.m_confidence = obj["confidence"].toDouble();
        QJsonObject box = obj["box"].toObject();
        instance.m_box = {box["x"].toDouble(), box["y"].toDouble(), box["width"].toDouble(), box["height"].toDouble()};
        instance.m_mask = Utils::Image::fromJson(obj["mask"].toString().toStdString());
        instance.m_color = Utils::Graphics::colorFromJson(obj["color"].toObject());
        m_instances.push_back(instance);
    }
}
