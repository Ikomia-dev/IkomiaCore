#include "CTextIO.h"
#include "Main/CoreTools.hpp"
#include <QJsonArray>

//----------------------//
//----- CTextField -----//
//----------------------//
int CTextField::getId() const
{
    return m_id;
}

std::string CTextField::getLabel() const
{
    return m_label;
}

std::string CTextField::getText() const
{
    return m_text;
}

double CTextField::getConfidence() const
{
    return m_confidence;
}

PolygonF CTextField::getPolygon() const
{
    return m_polygon;
}

CColor CTextField::getColor() const
{
    return m_color;
}

void CTextField::setId(int id)
{
    m_id = id;
}

void CTextField::setLabel(const std::string &label)
{
    m_label = label;
}

void CTextField::setText(const std::string &text)
{
    m_text = text;
}

void CTextField::setConfidence(double confidence)
{
    m_confidence = confidence;
}

void CTextField::setPolygon(const PolygonF &poly)
{
    m_polygon = poly;
}

void CTextField::setColor(const CColor &color)
{
    m_color = color;
}

QJsonObject CTextField::toJson() const
{
    QJsonObject obj;
    obj["id"] = m_id;
    obj["label"] = QString::fromStdString(m_label);
    obj["text"] = QString::fromStdString(m_text);
    obj["confidence"] = m_confidence;
    obj["color"] = CGraphicsJSON::toJsonObject(m_color);

    QJsonArray pts;
    for (size_t i=0; i<m_polygon.size(); ++i)
    {
        QJsonObject pt;
        pt["x"] = m_polygon[i].m_x;
        pt["y"] = m_polygon[i].m_y;
        pts.append(pt);
    }

    obj["polygon"] = pts;
    return obj;
}

//-------------------//
//----- CTextIO -----//
//-------------------//
CTextIO::CTextIO() : CWorkflowTaskIO(IODataType::TEXT, "CTextIO")
{
    m_description = QObject::tr("Text detection data: label, text, confidence, polygon and color.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
    m_graphicsIOPtr = std::make_shared<CGraphicsOutput>();
    m_textDataIOPtr = std::make_shared<CNumericIO<std::string>>();
}

CTextIO::CTextIO(const CTextIO &io): CWorkflowTaskIO(io)
{
    m_fields = io.m_fields;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_textDataIOPtr = io.m_textDataIOPtr->clone();
}

CTextIO::CTextIO(const CTextIO &&io): CWorkflowTaskIO(io)
{
    m_fields = std::move(io.m_fields);
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_textDataIOPtr = io.m_textDataIOPtr->clone();
}

CTextIO &CTextIO::operator=(const CTextIO &io)
{
    CWorkflowTaskIO::operator=(io);
    m_fields = io.m_fields;
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_textDataIOPtr = io.m_textDataIOPtr->clone();
    return *this;
}

CTextIO &CTextIO::operator=(const CTextIO &&io)
{
    CWorkflowTaskIO::operator=(io);
    m_fields = std::move(io.m_fields);
    m_graphicsIOPtr = io.m_graphicsIOPtr->clone();
    m_textDataIOPtr = io.m_textDataIOPtr->clone();
    return *this;
}

size_t CTextIO::getTextFieldCount() const
{
    return m_fields.size();
}

CTextField CTextIO::getTextField(size_t index) const
{
    if (index >= m_fields.size())
        throw CException(CoreExCode::INDEX_OVERFLOW, "No text field at given index: index overflow", __func__, __FILE__, __LINE__);

    return m_fields[index];
}

std::vector<CTextField> CTextIO::getTextFields() const
{
    return m_fields;
}

CDataInfoPtr CTextIO::getDataInfo()
{
    if (m_infoPtr == nullptr)
    {
        m_infoPtr = std::make_shared<CDataInfo>(this->m_dataType);
        m_infoPtr->metadata().insert(std::make_pair("Text fields count", std::to_string(m_fields.size())));
    }
    return m_infoPtr;
}

GraphicsOutputPtr CTextIO::getGraphicsIO() const
{
    return m_graphicsIOPtr;
}

CTextIO::DataStringIOPtr CTextIO::getDataStringIO() const
{
    return m_textDataIOPtr;
}

bool CTextIO::isDataAvailable() const
{
    return m_fields.size() > 0;
}

bool CTextIO::isComposite() const
{
    return true;
}

void CTextIO::init(const std::string &taskName, int imageIndex)
{
    clearData();
    m_graphicsIOPtr->setNewLayer(taskName);
    m_graphicsIOPtr->setImageIndex(imageIndex);
}

void CTextIO::finalize()
{
    m_textDataIOPtr->clearData();
    std::vector<std::string> ids;
    std::vector<std::string> labels;
    std::vector<std::string> texts;
    std::vector<std::string> confidences;
    std::vector<std::string> polygons;

    for (size_t i=0; i<m_fields.size(); ++i)
    {
        ids.push_back(std::to_string(m_fields[i].m_id));
        labels.push_back(m_fields[i].m_label);
        texts.push_back(m_fields[i].m_text);
        confidences.push_back(std::to_string(m_fields[i].m_confidence));

        std::string strPts;
        PolygonF poly =  m_fields[i].m_polygon;

        for (size_t j=0; j<poly.size(); ++j)
            strPts += "(" + std::to_string(poly[j].m_x) + "," + std::to_string(poly[j].m_y) + ");";

        polygons.push_back(strPts);
    }
    m_textDataIOPtr->addValueList(ids, "Id");
    m_textDataIOPtr->addValueList(labels, "Label");
    m_textDataIOPtr->addValueList(texts, "Text");
    m_textDataIOPtr->addValueList(confidences, "Confidence");
    m_textDataIOPtr->addValueList(polygons, "Polygon");
}

void CTextIO::addTextField(int id, const std::string &label, const std::string& text,
                           double confidence, double x, double y, double width, double height, const CColor &color)
{
    CTextField field;
    field.m_id = id;
    field.m_label = label;
    field.m_text = text;
    field.m_confidence = confidence;
    field.m_polygon = {CPointF(x, y), CPointF(x+width, y), CPointF(x+width, y+height), CPointF(x, y+height)};
    field.m_color = color;
    m_fields.push_back(field);

    //Set integrated I/O
    //Create rectangle graphics of bbox
    CGraphicsPolygonProperty prop;
    prop.m_category = label;
    prop.m_penColor = color;
    m_graphicsIOPtr->addPolygon(field.m_polygon, prop);

    //Class label
    std::string graphicsLabel = label + " #" + std::to_string(id) + ": " + text + " (" + std::to_string(confidence) + ")";
    CGraphicsTextProperty textProperty;
    textProperty.m_color = color;
    textProperty.m_fontSize = 8;
    m_graphicsIOPtr->addText(graphicsLabel, x + 5, y + 5, textProperty);
}

void CTextIO::addTextField(int id, const std::string& label, const std::string& text,
                           double confidence, const PolygonF& polygon, const CColor& color)
{
    CTextField field;
    field.m_id = id;
    field.m_label = label;
    field.m_text = text;
    field.m_confidence = confidence;
    field.m_polygon = polygon;
    field.m_color = color;
    m_fields.push_back(field);

    //Set integrated I/O
    //Create rectangle graphics of bbox
    CGraphicsPolygonProperty prop;
    prop.m_category = label;
    prop.m_penColor = color;
    m_graphicsIOPtr->addPolygon(polygon, prop);

    //Class label
    std::string graphicsLabel = label + " #" + std::to_string(id) + ": " + text + " (" + std::to_string(confidence) + ")";
    CGraphicsTextProperty textProperty;
    textProperty.m_color = color;
    textProperty.m_fontSize = 8;
    m_graphicsIOPtr->addText(graphicsLabel, polygon[0].m_x + 5, polygon[0].m_y + 5, textProperty);
}

void CTextIO::clearData()
{
    m_fields.clear();
    m_graphicsIOPtr->clearData();
    m_textDataIOPtr->clearData();
    m_infoPtr = nullptr;
}

void CTextIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonFile.readAll()));
    if(jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading text detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

void CTextIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson());
}

std::string CTextIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CTextIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

QJsonObject CTextIO::toJsonInternal() const
{
    QJsonArray jsonFields;
    for (size_t i=0; i<m_fields.size(); ++i)
        jsonFields.append(m_fields[i].toJson());

    QJsonObject root;
    root["fields"] = jsonFields;
    return root;
}

void CTextIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading text detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
    finalize();
}

void CTextIO::fromJsonInternal(const QJsonDocument &jsonDoc)
{
    QJsonObject root = jsonDoc.object();
    QJsonArray jsonFields = root["fields"].toArray();
    clearData();

    for (int i=0; i<jsonFields.size(); ++i)
    {
        QJsonObject field = jsonFields[i].toObject();
        int id = field["id"].toInt();
        std::string label = field["label"].toString().toStdString();
        std::string text = field["text"].toString().toStdString();
        double confidence = field["confidence"].toDouble();
        CColor color = Utils::Graphics::colorFromJson(field["color"].toObject());

        PolygonF poly;
        QJsonArray jsonPts = field["polygon"].toArray();

        for (int j=0; j<jsonPts.size(); ++j)
        {
            QJsonObject jsonPt = jsonPts[i].toObject();
            poly.push_back(CPointF(jsonPt["x"].toDouble(), jsonPt["y"].toDouble()));
        }

        addTextField(id, label, text, confidence, poly, color);
    }
}

std::shared_ptr<CTextIO> CTextIO::clone() const
{
    return std::static_pointer_cast<CTextIO>(cloneImp());
}

WorkflowTaskIOPtr CTextIO::cloneImp() const
{
    return std::shared_ptr<CTextIO>(new CTextIO(*this));
}
