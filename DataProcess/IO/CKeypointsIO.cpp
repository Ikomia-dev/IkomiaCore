#include "CKeypointsIO.h"

//----------------------------//
//----- CObjectKeypoints -----//
//----------------------------//

int CObjectKeypoints::getId() const
{
    return m_id;
}

std::string CObjectKeypoints::getLabel() const
{
    return m_label;
}

double CObjectKeypoints::getConfidence() const
{
    return m_confidence;
}

std::vector<double> CObjectKeypoints::getBox() const
{
    return m_box;
}

CColor CObjectKeypoints::getColor() const
{
    return m_color;
}

std::vector<Keypoint> CObjectKeypoints::getKeypoints() const
{
    return m_keypts;
}

CPointF CObjectKeypoints::getKeypoint(int index) const
{
    for (size_t i=0; i<m_keypts.size(); ++i)
    {
        if (m_keypts[i].first == index)
            return m_keypts[i].second;
    }
    //No point found
    throw CException(CoreExCode::INVALID_PARAMETER, "Invalid keypoint index", __func__, __FILE__, __LINE__);
}

void CObjectKeypoints::setId(int id)
{
    m_id = id;
}

void CObjectKeypoints::setLabel(const std::string &label)
{
    m_label = label;
}

void CObjectKeypoints::setConfidence(double conf)
{
    m_confidence = conf;
}

void CObjectKeypoints::setBox(const std::vector<double> &box)
{
    m_box = box;
}

void CObjectKeypoints::setColor(const CColor &color)
{
    m_color = color;
}

void CObjectKeypoints::setKeypoints(const std::vector<Keypoint> &pts)
{
    m_keypts = pts;
}

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
    for (size_t i=0; i<m_keypts.size(); ++i)
    {
        QJsonObject pt;
        pt["index"] = m_keypts[i].first;
        pt["x"] = m_keypts[i].second.m_x;
        pt["y"] = m_keypts[i].second.m_y;
        pts.append(pt);
    }
    obj["points"] = pts;
    return obj;
}

std::string CObjectKeypoints::repr() const
{
    return "CObjectKeypoints()";
}

std::ostream& operator<<(std::ostream& os, const CObjectKeypoints& obj)
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

    os << "Keypoints: [";
    for (size_t i=0; i<obj.m_keypts.size(); ++i)
    {
        os << std::to_string(obj.m_keypts[i].first) << ": " << "(" << std::to_string(obj.m_keypts[i].second.m_x) << "," << std::to_string(obj.m_keypts[i].second.m_y) << ")";
        if (i < obj.m_keypts.size() - 1)
            os << ", ";
    }
    os << "]" << std::endl;
    return os;
}

//-------------------------//
//----- CKeypointLink -----//
//-------------------------//

int CKeypointLink::getStartPointIndex() const
{
    return m_ptIndex1;
}

int CKeypointLink::getEndPointIndex() const
{
    return m_ptIndex2;
}

std::string CKeypointLink::getLabel() const
{
    return m_label;
}

CColor CKeypointLink::getColor() const
{
    return m_color;
}

void CKeypointLink::setStartPointIndex(int index)
{
    m_ptIndex1 = index;
}

void CKeypointLink::setEndPointIndex(int index)
{
    m_ptIndex2 = index;
}

void CKeypointLink::setLabel(const std::string &label)
{
    m_label = label;
}

void CKeypointLink::setColor(const CColor &color)
{
    m_color = color;
}

void CKeypointLink::fromJson(const QJsonObject& jsonObj)
{
    m_ptIndex1 = jsonObj["index1"].toInt();
    m_ptIndex1 = jsonObj["index2"].toInt();
    m_label = jsonObj["label"].toString().toStdString();
    m_color = Utils::Graphics::colorFromJson(jsonObj["color"].toObject());
}

QJsonObject CKeypointLink::toJson() const
{
    QJsonObject obj;
    obj["index1"] = m_ptIndex1;
    obj["index2"] = m_ptIndex2;
    obj["label"] = QString::fromStdString(m_label);
    obj["color"] = CGraphicsJSON::toJsonObject(m_color);
    return obj;
}

std::string CKeypointLink::repr() const
{
    return "CKeypointLink()";
}

std::ostream& operator<<(std::ostream& os, const CKeypointLink& link)
{
    os << "----- keypoints link -----" << std::endl;
    os << "Keypoint index #1: " << std::to_string(link.m_ptIndex1) << std::endl;
    os << "Keypoint index #2: " << std::to_string(link.m_ptIndex2) << std::endl;
    os << "Label: " << link.m_label << std::endl;
    os << "Color: [" << std::to_string(link.m_color[0]) << ", " << std::to_string(link.m_color[0]) << ", " << std::to_string(link.m_color[0]) << "]" << std::endl;
    return os;
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

std::string CKeypointsIO::repr() const
{
    std::stringstream s;
    s << "CKeypointsIO()";
    return s.str();
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

void CKeypointsIO::addObject(int id, const std::string &label, double confidence, double x, double y, double width, double height, const std::vector<Keypoint> keypts, CColor color)
{
    CObjectKeypoints obj;
    obj.m_id = id;
    obj.m_label = label;
    obj.m_confidence = confidence;
    obj.m_box = {x, y, width, height};
    obj.m_keypts = keypts;
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
    if (keypts.size() > 0)
    {
        CGraphicsPointProperty ptProp;
        ptProp.m_brushColor = color;
        ptProp.m_penColor = color;
        ptProp.m_size = 5;

        for (size_t i=0; i<keypts.size(); ++i)
            m_graphicsIOPtr->addPoint(keypts[i].second, ptProp);

        //Keypoints links graphics
        for (size_t i=0; i<m_links.size(); ++i)
        {
            CGraphicsPolylineProperty lineProp;
            lineProp.m_penColor = m_links[i].m_color;

            try
            {
                CPointF p1 = obj.getKeypoint(m_links[i].m_ptIndex1);
                CPointF p2 = obj.getKeypoint(m_links[i].m_ptIndex2);
                m_graphicsIOPtr->addPolyline({p1, p2}, lineProp);
            }
            catch(CException&)
            {
                //Skip invalid link
            }
        }
    }

    //Store object values to be shown in results table
    std::vector<CObjectMeasure> objRes;
    std::vector<double> keyptsInfo;

    for (size_t i=0; i<keypts.size(); ++i)
    {
        keyptsInfo.push_back(keypts[i].first);
        keyptsInfo.push_back(keypts[i].second.m_x);
        keyptsInfo.push_back(keypts[i].second.m_y);
    }

    objRes.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Identifier").toStdString()), id, graphicsObj->getId(), label));
    objRes.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), label));
    objRes.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {x, y, width, height}, graphicsObj->getId(), label));
    objRes.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Keypoints").toStdString()), keyptsInfo, graphicsObj->getId(), label));
    m_objMeasureIOPtr->addObjectMeasures(objRes);
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

BlobMeasureIOPtr CKeypointsIO::getBlobMeasureIO() const
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

GraphicsOutputPtr CKeypointsIO::getGraphicsIO() const
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

void CKeypointsIO::init(const std::string &taskName, int imageIndex)
{
    clearData();
    m_graphicsIOPtr->setNewLayer(taskName);
    m_graphicsIOPtr->setImageIndex(imageIndex);
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

        std::vector<Keypoint> keypts;
        QJsonArray jsonPts = obj["points"].toArray();

        for (int i=0; i<jsonPts.size(); ++i)
        {
            QJsonObject pt = jsonPts[i].toObject();
            keypts.push_back(std::make_pair(pt["index"].toInt(), CPointF(pt["x"].toDouble(), pt["y"].toDouble())));
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
    m_keyptsLinkIOPtr->clearData();

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

WorkflowTaskIOPtr CKeypointsIO::cloneImp() const
{
    return std::shared_ptr<CKeypointsIO>(new CKeypointsIO(*this));
}
