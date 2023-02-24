#include "CKeypointsIO.h"

//----------------------------//
//----- CObjectKeypoints -----//
//----------------------------//

QJsonObject CObjectKeypoints::toJson() const
{
    QJsonObject obj;
    obj["id"] = m_id;
    obj["label"] = QString::fromStdString(m_label);
    obj["confidence"] = m_confidence;
    obj["color"] = CGraphicsJSON::toJsonObject(m_color);

    QJsonObject box;
    box["x"] = m_box[0];
    box["y"] = m_box[1];
    box["width"] = m_box[2];
    box["height"] = m_box[3];
    obj["box"] = box;

    QJsonArray pts;
    for (size_t i=0; i<m_pts.size(); ++i)
    {
        QJsonObject pt;
        pt["x"] = m_pts[i].m_x;
        pt["y"] = m_pts[i].m_x;
        pts.append(pt);
    }
    obj["points"] = pts;
    return obj;
}

//-------------------------//
//----- CKeypointLink -----//
//-------------------------//

QJsonObject CKeypointLink::toJson() const
{
    QJsonObject obj;
    obj["index1"] = m_ptIndex1;
    obj["index2"] = m_ptIndex2;
    obj["label"] = QString::fromStdString(m_label);
    obj["color"] = CGraphicsJSON::toJsonObject(m_color);
    return obj;
}

void CKeypointLink::fromJson(const QJsonObject& jsonObj)
{
    m_ptIndex1 = jsonObj["index1"].toInt();
    m_ptIndex1 = jsonObj["index2"].toInt();
    m_label = jsonObj["label"].toString().toStdString();
    m_color = Utils::Graphics::colorFromJson(jsonObj["color"].toObject());
}

//------------------------//
//----- CKeypointsIO -----//
//------------------------//

CKeypointsIO::CKeypointsIO(): CWorkflowTaskIO(IODataType::KEYPOINTS, "CKeypointsIO")
{
    m_description = QObject::tr("Keypoints detection data: object, label, confidence, box, keypoints data.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
    m_graphicsIOPtr = std::make_shared<CGraphicsOutput>();
    m_objMeasureIOPtr = std::make_shared<CBlobMeasureIO>();
    m_keyptsLinkIOPtr = std::make_shared<CNumericIO<std::string>>();
}

CKeypointsIO::CKeypointsIO(const CKeypointsIO &io): CWorkflowTaskIO(io)
{
    m_objects = io.m_objects;
    m_keyptsNames = io.m_keyptsNames;
    m_links = io.m_links;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_objMeasureIOPtr = io.m_objMeasureIOPtr->clone();
    m_keyptsLinkIOPtr = io.m_keyptsLinkIOPtr->clone();
}

CKeypointsIO::CKeypointsIO(const CKeypointsIO &&io): CWorkflowTaskIO(io)
{
    m_objects = std::move(io.m_objects);
    m_keyptsNames = std::move(io.m_keyptsNames);
    m_links = std::move(io.m_links);
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_objMeasureIOPtr = io.m_objMeasureIOPtr->clone();
    m_keyptsLinkIOPtr = io.m_keyptsLinkIOPtr->clone();
}

CKeypointsIO &CKeypointsIO::operator=(const CKeypointsIO &io)
{
    CWorkflowTaskIO::operator=(io);
    m_objects = io.m_objects;
    m_keyptsNames = io.m_keyptsNames;
    m_links = io.m_links;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_objMeasureIOPtr = io.m_objMeasureIOPtr->clone();
    m_keyptsLinkIOPtr = io.m_keyptsLinkIOPtr->clone();
    return *this;
}

CKeypointsIO &CKeypointsIO::operator=(const CKeypointsIO &&io)
{
    CWorkflowTaskIO::operator=(io);
    m_objects = std::move(io.m_objects);
    m_keyptsNames = std::move(io.m_keyptsNames);
    m_links = std::move(io.m_links);
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_objMeasureIOPtr = io.m_objMeasureIOPtr->clone();
    m_keyptsLinkIOPtr = io.m_keyptsLinkIOPtr->clone();
    return *this;
}

void CKeypointsIO::addObject(int id, const std::string &label, double confidence, double x, double y, double width, double height, const std::vector<CPointF> keypts, CColor color)
{
    CObjectKeypoints obj;
    obj.m_id = id;
    obj.m_label = label;
    obj.m_confidence = confidence;
    obj.m_box = {x, y, width, height};
    obj.m_pts = keypts;
    obj.m_color = color;
    m_objects.push_back(obj);

    //Set integrated I/O

    //Create rectangle graphics of bbox
    CGraphicsRectProperty rectProp;
    rectProp.m_category = label;
    rectProp.m_penColor = color;
    auto graphicsObj = m_graphicsIOPtr->addRectangle(x, y, width, height, rectProp);

    //Class label
    std::string graphicsLabel = label + " #" + std::to_string(id) + ": " + std::to_string(confidence);
    CGraphicsTextProperty textProperty;
    textProperty.m_color = color;
    textProperty.m_fontSize = 8;
    m_graphicsIOPtr->addText(graphicsLabel, x + 5, y + 5, textProperty);

    //Keypoints graphics
    CGraphicsPointProperty ptProp;
    ptProp.m_brushColor = color;
    ptProp.m_penColor = color;
    ptProp.m_size = 5;

    for (size_t i=0; i<keypts.size(); ++i)
        m_graphicsIOPtr->addPoint(keypts[i], ptProp);

    //Keypoints links graphics
    for (size_t i=0; i<m_links.size(); ++i)
    {
        CGraphicsPolylineProperty lineProp;
        lineProp.m_penColor = m_links[i].m_color;
        CPoint p1 = keypts[m_links[i].m_ptIndex1];
        CPoint p2 = keypts[m_links[i].m_ptIndex2];
        m_graphicsIOPtr->addPolyline({p1, p2}, lineProp);
    }

    //Store object values to be shown in results table
    std::vector<CObjectMeasure> objRes;
    std::vector<double> coords;

    for (size_t i=0; i<keypts.size(); ++i)
    {
        coords.push_back(keypts[i].m_x);
        coords.push_back(keypts[i].m_y);
    }

    objRes.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Identifier").toStdString()), id, graphicsObj->getId(), label));
    objRes.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), label));
    objRes.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {x, y, width, height}, graphicsObj->getId(), label));
    objRes.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Keypoints(x,y)").toStdString()), coords, graphicsObj->getId(), label));
    m_objMeasureIOPtr->addObjectMeasures(objRes);

    //Store keypoints links to be shown in results table
    std::vector<std::string> startPtIndices;
    std::vector<std::string> endPtIndices;
    std::vector<std::string> labels;

    for (size_t i=0; i<m_links.size(); ++i)
    {
        startPtIndices.push_back(std::to_string(m_links[i].m_ptIndex1));
        endPtIndices.push_back(std::to_string(m_links[i].m_ptIndex1));
        labels.push_back(m_links[i].m_label);
    }

    m_keyptsLinkIOPtr->addValueList(startPtIndices, "Start point");
    m_keyptsLinkIOPtr->addValueList(endPtIndices, "End point");
    m_keyptsLinkIOPtr->addValueList(labels, "Label");

}

void CKeypointsIO::clearData()
{
    m_objects.clear();
    m_keyptsNames.clear();
    m_links.clear();
    m_graphicsIOPtr->clearData();
    m_objMeasureIOPtr->clearData();
    m_keyptsLinkIOPtr->clearData();
    m_infoPtr = nullptr;
}

std::shared_ptr<CKeypointsIO> CKeypointsIO::clone() const
{
    return std::static_pointer_cast<CKeypointsIO>(cloneImp());
}

std::shared_ptr<CBlobMeasureIO> CKeypointsIO::getBlobMeasureIO() const
{
    return m_objMeasureIOPtr;
}

CDataInfoPtr CKeypointsIO::getDataInfo()
{
    if (m_infoPtr == nullptr)
    {
        m_infoPtr = std::make_shared<CDataInfo>(this->m_dataType);
        m_infoPtr->metadata().insert(std::make_pair("Object count", std::to_string(m_objects.size())));
        m_infoPtr->metadata().insert(std::make_pair("Keypoints per object", std::to_string(m_keyptsNames.size())));
    }
    return m_infoPtr;
}

CKeypointsIO::DataStringIOPtr CKeypointsIO::getDataStringIO() const
{
    return m_keyptsLinkIOPtr;
}

std::shared_ptr<CGraphicsOutput> CKeypointsIO::getGraphicsIO() const
{
    return m_graphicsIOPtr;
}

size_t CKeypointsIO::getObjectCount() const
{
    return m_objects.size();
}

CObjectKeypoints CKeypointsIO::getObject(size_t index) const
{
    if (index >= m_objects.size())
        throw CException(CoreExCode::INDEX_OVERFLOW, "No keypoints detection at given index: index overflow", __func__, __FILE__, __LINE__);

    return m_objects[index];
}

std::vector<CObjectKeypoints> CKeypointsIO::getObjects() const
{
    return m_objects;
}

std::vector<CKeypointLink> CKeypointsIO::getKeypointLinks() const
{
    return m_links;
}

std::vector<std::string> CKeypointsIO::getKeypointNames() const
{
    return m_keyptsNames;
}

bool CKeypointsIO::isComposite() const
{
    return true;
}

bool CKeypointsIO::isDataAvailable() const
{
    return m_objects.size() > 0;
}

void CKeypointsIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonFile.readAll()));
    if(jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading keypoint detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

void CKeypointsIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson());
}

std::string CKeypointsIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CKeypointsIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

QJsonObject CKeypointsIO::toJsonInternal() const
{
    QJsonArray objects;
    for (size_t i=0; i<m_objects.size(); ++i)
        objects.append(m_objects[i].toJson());

    QJsonArray keyptNames;
    for (size_t i=0; i<m_keyptsNames.size(); ++i)
        keyptNames.append(QString::fromStdString(m_keyptsNames[i]));

    QJsonArray keyptLinks;
    for (size_t i=0; i<m_links.size(); ++i)
        keyptLinks.append(m_links[i].toJson());

    QJsonObject root;
    root["objects"] = objects;
    root["names"] = keyptNames;
    root["links"] = keyptLinks;
    return root;
}

void CKeypointsIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading keypoint detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

void CKeypointsIO::fromJsonInternal(const QJsonDocument &doc)
{
    clearData();
    QJsonObject root = doc.object();
    QJsonArray objects = root["detections"].toArray();

    // Objects with keypoints
    for (int i=0; i<objects.size(); ++i)
    {
        QJsonObject obj = objects[i].toObject();
        int id = obj["id"].toInt();
        std::string label = obj["label"].toString().toStdString();
        double confidence = obj["confidence"].toDouble();
        CColor color = Utils::Graphics::colorFromJson(obj["color"].toObject());

        QJsonObject box = obj["box"].toObject();
        double x = box["x"].toDouble();
        double y = box["y"].toDouble();
        double width = box["width"].toDouble();
        double height = box["height"].toDouble();

        std::vector<CPointF> keypts;
        QJsonArray jsonPts = obj["points"].toArray();

        for (int i=0; i<jsonPts.size(); ++i)
        {
            QJsonObject pt = jsonPts[i].toObject();
            keypts.push_back(CPointF(pt["x"].toDouble(), pt["y"].toDouble()));
        }

        addObject(id, label, confidence, x, y, width, height, keypts, color);
    }

    //Keypoints names
    QJsonArray jsonNames = root["names"].toArray();
    for (int i=0; i<jsonNames.size(); ++i)
        m_keyptsNames.push_back(jsonNames[i].toString().toStdString());

    //Keypoints links
    QJsonArray jsonLinks = root["links"].toArray();
    for (int i=0; i<jsonLinks.size(); ++i)
    {
        CKeypointLink link;
        link.fromJson(jsonLinks[i].toObject());
        m_links.push_back(link);
    }
}

void CKeypointsIO::setKeypointNames(const std::vector<std::string> &names)
{
    m_keyptsNames = names;
}

void CKeypointsIO::setKeypointLinks(const std::vector<CKeypointLink> &links)
{
    m_links = links;
}

std::shared_ptr<CWorkflowTaskIO> CKeypointsIO::cloneImp() const
{
    return std::shared_ptr<CKeypointsIO>(new CKeypointsIO(*this));
}
