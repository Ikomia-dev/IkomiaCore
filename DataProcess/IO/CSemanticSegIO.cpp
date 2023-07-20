#include <QJsonArray>
#include "CSemanticSegIO.h"
#include "Main/CoreTools.hpp"
#include "DataProcessTools.hpp"
#include "CInstanceSegIO.h"

CSemanticSegIO::CSemanticSegIO() : CWorkflowTaskIO(IODataType::SEMANTIC_SEGMENTATION, "CSemanticSegIO")
{
    m_description = QObject::tr("Semantic segmentation data: mask and class names.").toStdString();
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

std::string CSemanticSegIO::repr() const
{
    std::stringstream s;
    s << "CSemanticSegIO()";
    return s.str();
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

CMat CSemanticSegIO::getLegend() const
{
    return m_imgLegendIOPtr->getImage();
}

std::vector<std::string> CSemanticSegIO::getClassNames() const
{
    return m_classes;
}

std::vector<CColor> CSemanticSegIO::getColors() const
{
    return m_colors;
}

ImageIOPtr CSemanticSegIO::getMaskImageIO() const
{
    return m_imgMaskIOPtr;
}

ImageIOPtr CSemanticSegIO::getLegendImageIO() const
{
    return m_imgLegendIOPtr;
}

InputOutputVect CSemanticSegIO::getSubIOList(const std::set<IODataType> &dataTypes) const
{
    InputOutputVect ioList;

    auto it = dataTypes.find(IODataType::IMAGE);
    if(it != dataTypes.end())
    {
        ioList.push_back(m_imgMaskIOPtr);
        ioList.push_back(m_imgLegendIOPtr);
    }
    return ioList;
}

int CSemanticSegIO::getReferenceImageIndex() const
{
    return m_refImageIndex;
}

void CSemanticSegIO::setMask(const CMat &mask)
{
    m_imgMaskIOPtr->setImage(mask);
    std::vector<cv::Mat> inputs;
    inputs.push_back(mask);
    cv::calcHist(inputs, {0}, cv::Mat(), m_histo, {256}, {0, 256}, false);
    generateLegend();
}

void CSemanticSegIO::setClassNames(const std::vector<std::string> &names)
{
    if (m_colors.size() != 0 && names.size() != m_colors.size())
        throw CException(CoreExCode::INVALID_SIZE, "Semantic segmentation output error: there must be the same number of classes and colors.", __func__, __FILE__, __LINE__);

    m_classes = names;
    if (m_colors.empty())
        generateRandomColors();
}

void CSemanticSegIO::setClassColors(const std::vector<CColor> &colors)
{
    if (colors.size() < m_classes.size())
        throw CException(CoreExCode::INVALID_SIZE, "Colors count must be greater or equal of class names count", __func__, __FILE__, __LINE__);

    m_colors = colors;
}

void CSemanticSegIO::setReferenceImageIndex(int index)
{
    m_refImageIndex = index;
}

bool CSemanticSegIO::isDataAvailable() const
{
    return m_imgMaskIOPtr->isDataAvailable();
}

bool CSemanticSegIO::isComposite() const
{
    return true;
}

void CSemanticSegIO::clearData()
{
    m_classes.clear();
    m_colors.clear();
    m_histo.release();
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

    fromJsonInternal(jsonDoc);
}

void CSemanticSegIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal({"image_format", "jpg"}));
    jsonFile.write(jsonDoc.toJson());
}

std::string CSemanticSegIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact", "image_format", "jpg"};
    return toJson(options);
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

    fromJsonInternal(jsonDoc);
}

std::shared_ptr<CSemanticSegIO> CSemanticSegIO::clone() const
{
    return std::static_pointer_cast<CSemanticSegIO>(cloneImp());
}

void CSemanticSegIO::copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    auto type = ioPtr->getDataType();
    if (type == IODataType::SEMANTIC_SEGMENTATION)
    {
        //Should not be called in this case
        auto pIO = dynamic_cast<const CSemanticSegIO*>(ioPtr.get());
        if(pIO)
            *this = *pIO;
    }
    else if (type == IODataType::INSTANCE_SEGMENTATION)
    {
        auto instanceIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(ioPtr);
        if (instanceIOPtr)
        {
            clearData();
            std::vector<std::string> instanceNames = instanceIOPtr->getClassNames();
            CMat mask = instanceIOPtr->getMergeMask();
            // Substract 1 as instance segmentation IO adds background class for zero-pixels in merge mask
            mask = mask - 1;
            // Set names and mask
            setClassNames(instanceNames);
            setMask(mask);
        }
    }
}

WorkflowTaskIOPtr CSemanticSegIO::cloneImp() const
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
    root["referenceImageIndex"] = m_refImageIndex;
    return root;
}

void CSemanticSegIO::fromJsonInternal(const QJsonDocument &doc)
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
        CColor color = {(uchar)(obj["r"].toInt()), (uchar)(obj["g"].toInt()), (uchar)(obj["b"].toInt())};
        m_colors.push_back(color);
    }
    generateLegend();
}

void CSemanticSegIO::generateLegend()
{
    if (m_histo.data == nullptr)
        return;

    std::vector<int> colorIndices;
    for (size_t i=0; i<m_colors.size(); ++i)
    {
        if (m_histo.at<float>(i) > 0)
            colorIndices.push_back(i);
    }

    const int imgH = 1024;
    const int imgW = 1024;
    CMat legend(imgH, imgW, CV_8UC3, cv::Scalar(255,255,255));

    size_t nbColors = colorIndices.size();
    if (nbColors > 0)
    {
        const int offsetX = 10;
        const int offsetY = 10;
        const int interline = 5;
        int rectHeight = (int)((imgH - (2*offsetY) - ((nbColors-1)*interline)) / nbColors);
        int rectWidth = imgW / 3;
        int font = cv::FONT_HERSHEY_SIMPLEX;
        const int fontScale = 1;
        const int thickness = 2;

        for (size_t i=0; i<nbColors; ++i)
        {
            // Color frame
            cv::Vec3b color = {m_colors[colorIndices[i]][0], m_colors[colorIndices[i]][1], m_colors[colorIndices[i]][2]};
            cv::Rect colorFrameRect = cv::Rect(offsetX, offsetY + (i * (rectHeight + interline)), rectWidth, rectHeight);
            cv::rectangle(legend, colorFrameRect, color, -1);
            // Class name
            cv::Point textOrigin(3 * offsetX + rectWidth, offsetY + (i * (rectHeight + interline)) + (rectHeight / 2));
            cv::putText(legend, m_classes[colorIndices[i]], textOrigin, font, fontScale, {0, 0, 0}, thickness);
        }
    }
    m_imgLegendIOPtr->setImage(legend);
}

void CSemanticSegIO::generateRandomColors()
{
    std::srand(RANDOM_COLOR_SEED);
    double factor = 255.0 / (double)RAND_MAX;

    for (size_t i=0; i<m_classes.size(); ++i)
    {
        CColor color = {
            (uchar)((double)std::rand() * factor),
            (uchar)((double)std::rand() * factor),
            (uchar)((double)std::rand() * factor)
        };
        m_colors.push_back(color);
    }
}
