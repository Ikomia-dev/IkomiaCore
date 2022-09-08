#include "CSemanticSegIO.h"
#include "Main/CoreTools.hpp"
#include "DataProcessTools.hpp"
#include <QJsonArray>

CSemanticSegIO::CSemanticSegIO() : CWorkflowTaskIO(IODataType::SEMANTIC_SEGMENTATION, "CSemanticSegIO")
{
    m_description = QObject::tr("Semantic segmentation data: mask and class names.\n").toStdString();
    m_saveFormat = DataFileFormat::JSON;
    m_imgMaskIOPtr = std::make_shared<CImageIO>(IODataType::IMAGE_LABEL);
    m_imgLegendIOPtr = std::make_shared<CImageIO>(IODataType::IMAGE);
}

CSemanticSegIO::CSemanticSegIO(const CSemanticSegIO &io): CWorkflowTaskIO(io)
{
    m_classes = io.m_classes;
    m_colors = io.m_colors;
    m_imgMaskIOPtr = io.m_imgMaskIOPtr->clone();
    m_imgLegendIOPtr = io.m_imgLegendIOPtr->clone();
}

CSemanticSegIO::CSemanticSegIO(const CSemanticSegIO &&io): CWorkflowTaskIO(io)
{
    m_classes = std::move(io.m_classes);
    m_colors = std::move(io.m_colors);
    m_imgMaskIOPtr = io.m_imgMaskIOPtr->clone();
    m_imgLegendIOPtr = io.m_imgLegendIOPtr->clone();
}

CSemanticSegIO &CSemanticSegIO::operator=(const CSemanticSegIO &io)
{
    CWorkflowTaskIO::operator=(io);
    m_classes = io.m_classes;
    m_colors = io.m_colors;
    m_imgMaskIOPtr = io.m_imgMaskIOPtr->clone();
    m_imgLegendIOPtr = io.m_imgLegendIOPtr->clone();
    return *this;
}

CSemanticSegIO &CSemanticSegIO::operator=(const CSemanticSegIO &&io)
{
    CWorkflowTaskIO::operator=(io);
    m_classes = std::move(io.m_classes);
    m_colors = std::move(io.m_colors);
    m_imgMaskIOPtr = io.m_imgMaskIOPtr->clone();
    m_imgLegendIOPtr = io.m_imgLegendIOPtr->clone();
    return *this;
}

CMat CSemanticSegIO::getMask() const
{
    return m_imgMaskIOPtr->getImage();
}

std::vector<std::string> CSemanticSegIO::getClassNames() const
{
    return m_classes;
}

std::shared_ptr<CImageIO> CSemanticSegIO::getMaskImageIO() const
{
    return m_imgMaskIOPtr;
}

std::shared_ptr<CImageIO> CSemanticSegIO::getLegendImageIO() const
{
    return m_imgLegendIOPtr;
}

void CSemanticSegIO::setMask(const CMat &mask)
{
    m_imgMaskIOPtr->setImage(mask);
}

void CSemanticSegIO::setClassNames(const std::vector<std::string> &names, const std::vector<cv::Vec3b> &colors)
{
    m_classes = names;
    m_colors = colors;
    generateLegend();
}

bool CSemanticSegIO::isDataAvailable() const
{
    return m_imgMaskIOPtr->isDataAvailable();
}

void CSemanticSegIO::clearData()
{
    m_classes.clear();
    m_colors.clear();
    m_imgMaskIOPtr->clearData();
    m_imgLegendIOPtr->clearData();
}

void CSemanticSegIO::load(const std::string &path)
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

void CSemanticSegIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal({"image_format", "jpg"}));
    jsonFile.write(jsonDoc.toJson());
}

std::string CSemanticSegIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal(options));
    return toFormattedJson(doc, options);
}

void CSemanticSegIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading object detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJson(jsonDoc);
}

std::shared_ptr<CSemanticSegIO> CSemanticSegIO::clone() const
{
    return std::static_pointer_cast<CSemanticSegIO>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CSemanticSegIO::cloneImp() const
{
    return std::shared_ptr<CSemanticSegIO>(new CSemanticSegIO(*this));
}

QJsonObject CSemanticSegIO::toJsonInternal(const std::vector<std::string> &options) const
{
    QJsonObject root;
    root["mask"] = QString::fromStdString(Utils::Image::toJson(m_imgMaskIOPtr->getImage(), options));

    QJsonArray classes;
    for (size_t i=0; i<m_classes.size(); ++i)
        classes.append(QString::fromStdString(m_classes[i]));

    root["classes"] = classes;

    QJsonArray colors;
    for (size_t i=0; i<m_colors.size(); ++i)
    {
        QJsonObject obj;
        obj["r"] = m_colors[i][0];
        obj["g"] = m_colors[i][1];
        obj["b"] = m_colors[i][2];
        colors.append(obj);
    }
    root["colors"] = colors;
    return root;
}

void CSemanticSegIO::fromJson(const QJsonDocument &doc)
{
    clearData();
    QJsonObject root = doc.object();
    auto mask = Utils::Image::fromJson(root["mask"].toString().toStdString());
    m_imgMaskIOPtr->setImage(mask);

    QJsonArray classes = root["classes"].toArray();
    for (int i=0; i<classes.size(); ++i)
        m_classes.push_back(classes[i].toString().toStdString());

    QJsonArray colors = root["colors"].toArray();
    for (int i=0; i<colors.size(); ++i)
    {
        QJsonObject obj = colors[i].toObject();
        cv::Vec3b color;
        color[0] = obj["r"].toInt();
        color[1] = obj["g"].toInt();
        color[2] = obj["b"].toInt();
        m_colors.push_back(color);
    }
    generateLegend();
}

void CSemanticSegIO::generateLegend()
{
    const int imgH = 1024;
    const int imgW = 1024;
    size_t nbColors = m_colors.size();
    const int offsetX = 10;
    const int offsetY = 10;
    const int interline = 5;
    int rectHeight = (int)((imgH - (2*offsetY) - ((nbColors-1)*interline)) / nbColors);
    int rectWidth = imgW / 3;

    CMat legend(imgH, imgW, CV_8UC3, cv::Scalar(255,255,255));
    int font = cv::FONT_HERSHEY_SIMPLEX;
    const int fontScale = 1;
    const int thickness = 2;

    for (size_t i=0; i<nbColors; ++i)
    {
        // Color frame
        cv::Rect colorFrameRect = cv::Rect(offsetX, offsetY + (i * (rectHeight + interline)), rectWidth, rectHeight);
        cv::rectangle(legend, colorFrameRect, m_colors[i], -1);
        // Class name
        cv::Point textOrigin(3 * offsetX + rectWidth, offsetY + (i * (rectHeight + interline)) + (rectHeight / 2));
        cv::putText(legend, m_classes[i], textOrigin, font, fontScale, {0, 0, 0}, thickness);

    }
    m_imgLegendIOPtr->setImage(legend);
}
