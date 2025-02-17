// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the Ikomia API libraries.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "CBlobMeasureIO.h"
#include "CObjectDetectionIO.h"
#include "CInstanceSegIO.h"
#include "CKeypointsIO.h"
#include "CException.h"
#include "Main/CoreTools.hpp"
#include <QJsonArray>

//--------------------------------//
//----- class CObjectMeasure -----//
//--------------------------------//
CObjectMeasure::CObjectMeasure()
{
}

CObjectMeasure::CObjectMeasure(const CMeasure &measure, double value, size_t graphicsId, const std::string& label)
{
    m_measure = measure;
    m_values.push_back(value);
    m_graphicsId = graphicsId;
    m_label = label;
}

CObjectMeasure::CObjectMeasure(const CMeasure &measure, const std::vector<double> &values, size_t graphicsId, const std::string &label)
{
    m_measure = measure;
    m_values = values;
    m_graphicsId = graphicsId;
    m_label = label;
}

CObjectMeasure::CObjectMeasure(const CMeasure &measure, std::initializer_list<double> values, size_t graphicsId, const std::string &label)
{
    m_measure = measure;
    m_values.insert(m_values.end(), values);
    m_graphicsId = graphicsId;
    m_label = label;
}

CMeasure CObjectMeasure::getMeasureInfo() const
{
    return m_measure;
}

std::vector<double> CObjectMeasure::getValues() const
{
    return m_values;
}

void CObjectMeasure::setValues(const std::vector<double> &values)
{
    m_values = values;
}

QJsonObject CObjectMeasure::toJson() const
{
    QJsonObject measure;
    measure["id"] = m_measure.m_id;
    measure["name"] = QString::fromStdString(m_measure.m_name);
    measure["formula"] = QString::fromStdString(m_measure.m_formula);

    QJsonArray values;
    for (size_t i=0; i<m_values.size(); ++i)
        values.append(m_values[i]);

    QJsonObject root;
    root["measure"] = measure;
    root["values"] = values;
    root["graphicsId"] = static_cast<qint64>(m_graphicsId);
    root["label"] = QString::fromStdString(m_label);
    return root;
}

void CObjectMeasure::fromJson(const QJsonObject &obj)
{
    QJsonObject measure = obj["measure"].toObject();
    m_measure.m_id = measure["id"].toInt();
    m_measure.m_name = measure["name"].toString().toStdString();
    m_measure.m_formula = measure["formula"].toString().toStdString();

    m_values.clear();
    QJsonArray values = obj["values"].toArray();

    for (int i=0; i<values.size(); ++i)
        m_values.push_back(values[i].toDouble());

    m_graphicsId = static_cast<size_t>(obj["graphicsId"].toDouble());
    m_label = obj["label"].toString().toStdString();
}

//--------------------------------//
//----- class CBlobMeasureIO -----//
//--------------------------------//
CBlobMeasureIO::CBlobMeasureIO() : CWorkflowTaskIO(IODataType::BLOB_VALUES, "BlobMeasureIO")
{
    m_description = QObject::tr("Predefined measures computed from connected components (Surface, perimeter...).").toStdString();
    m_saveFormat = DataFileFormat::CSV;
}

CBlobMeasureIO::CBlobMeasureIO(const std::string &name) : CWorkflowTaskIO(IODataType::BLOB_VALUES, name)
{
    m_description = QObject::tr("Predefined measures computed from connected components (Surface, perimeter...).").toStdString();
    m_saveFormat = DataFileFormat::CSV;
}

CBlobMeasureIO::CBlobMeasureIO(const CBlobMeasureIO& io) : CWorkflowTaskIO(io)
{
    m_measures = io.m_measures;
}

CBlobMeasureIO::CBlobMeasureIO(const CBlobMeasureIO&& io) : CWorkflowTaskIO(io)
{
    m_measures = std::move(io.m_measures);
}

CBlobMeasureIO::~CBlobMeasureIO()
{
}

CBlobMeasureIO &CBlobMeasureIO::operator=(const CBlobMeasureIO& io)
{
    CWorkflowTaskIO::operator=(io);
    m_measures = io.m_measures;
    return *this;
}

std::string CBlobMeasureIO::repr() const
{
    std::stringstream s;
    s << "CBlobMeasureIO(" << m_name << ")";
    return s.str();
}

CBlobMeasureIO &CBlobMeasureIO::operator=(const CBlobMeasureIO&& io)
{
    CWorkflowTaskIO::operator=(io);
    m_measures = std::move(io.m_measures);
    return *this;
}

void CBlobMeasureIO::setObjectMeasure(size_t index, const CObjectMeasure &measure)
{
    if(index >= m_measures.size())
        addObjectMeasure(measure);
    else
    {
        ObjectMeasures measures;
        measures.push_back(measure);
        m_measures[index] = measures;
    }
}

ObjectsMeasures CBlobMeasureIO::getMeasures() const
{
    return m_measures;
}

ObjectMeasures CBlobMeasureIO::getBlobMeasures(size_t index) const
{
    if(index >= m_measures.size())
        throw CException(CoreExCode::INDEX_OVERFLOW, "Blob index out of range");

    return m_measures[index];
}

int CBlobMeasureIO::getBlobMeasureIndex(size_t index, std::string name)
{
    ObjectMeasures measures = getBlobMeasures(index);
    for(size_t i=0; i<measures.size(); ++i)
    {
        if(measures[i].m_measure.m_name == name)
            return i;
    }
    return -1;
}

int CBlobMeasureIO::getBlobMeasureIndex(size_t index, int id)
{
    return getBlobMeasureIndex(index, CMeasure::getName(id));
}

std::set<std::string> CBlobMeasureIO::getMeasuresNames() const
{
    std::set<std::string> names;
    //Iterate throw objects (ie lines)
    for(size_t i=0; i<m_measures.size(); ++i)
    {
        //Iterate throw object measures (ie columns)
        for(size_t j=0; j<m_measures[i].size(); ++j)
            names.insert(m_measures[i][j].m_measure.m_name);
    }
    return names;
}

bool CBlobMeasureIO::isDataAvailable() const
{
    return m_measures.size() > 0;
}

void CBlobMeasureIO::addObjectMeasure(const CObjectMeasure &measure)
{
    ObjectMeasures measures;
    measures.push_back(measure);
    m_measures.push_back(measures);
}

void CBlobMeasureIO::addObjectMeasures(const std::vector<CObjectMeasure> &measures)
{
    m_measures.push_back(measures);
}

void CBlobMeasureIO::clearData()
{
    m_measures.clear();
}

void CBlobMeasureIO::copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    auto type = ioPtr->getDataType();
    if (type == IODataType::OBJECT_DETECTION)
    {
        auto pObjectDetectionIO = std::dynamic_pointer_cast<CObjectDetectionIO>(ioPtr);
        if (pObjectDetectionIO)
        {
            auto blobMeasureOutPtr = pObjectDetectionIO->getBlobMeasureIO();
            if (blobMeasureOutPtr)
            {
                auto pBlobOut = dynamic_cast<const CBlobMeasureIO*>(blobMeasureOutPtr.get());
                if (pBlobOut)
                    *this = *pBlobOut;
            }
        }
    }
    else if (type == IODataType::INSTANCE_SEGMENTATION)
    {
        auto pInstanceSegIO = std::dynamic_pointer_cast<CInstanceSegIO>(ioPtr);
        if (pInstanceSegIO)
        {
            auto blobMeasureOutPtr = pInstanceSegIO->getBlobMeasureIO();
            if (blobMeasureOutPtr)
            {
                auto pBlobOut = dynamic_cast<const CBlobMeasureIO*>(blobMeasureOutPtr.get());
                if (pBlobOut)
                    *this = *pBlobOut;
            }
        }
    }
    else if (type == IODataType::KEYPOINTS)
    {
        auto keyptsIOPtr = std::dynamic_pointer_cast<CKeypointsIO>(ioPtr);
        if (keyptsIOPtr)
        {
            auto blobMeasureOutPtr = keyptsIOPtr->getBlobMeasureIO();
            if (blobMeasureOutPtr)
            {
                auto pBlobOut = dynamic_cast<const CBlobMeasureIO*>(blobMeasureOutPtr.get());
                if (pBlobOut)
                    *this = *pBlobOut;
            }
        }
    }
}

void CBlobMeasureIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension == ".csv")
        return loadCSV(path);
    else
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .csv files.", __func__, __FILE__, __LINE__);
}

void CBlobMeasureIO::loadCSV(const std::string &path)
{
    std::string line;
    std::ifstream file(path);

    // Header labels: object index, id, category and list of measure
    std::getline(file, line);
    std::vector<std::string> measureNames;
    Utils::String::tokenize(line, measureNames, ";");

    // Get objects measure
    while (std::getline(file, line))
    {
        ObjectMeasures measures;
        std::vector<std::string> strData;
        Utils::String::tokenize(line, strData, ";");

        for (size_t i=3; i<strData.size(); ++i)
        {
            auto measureId = CMeasure::getIdFromName(measureNames[i]);
            CMeasure measure(measureId, measureNames[i]);

            std::vector<std::string> strValues;
            Utils::String::tokenize(strData[i], strValues, "-");

            std::vector<double> values;
            for (size_t j=0; j<strValues.size(); ++j)
                values.push_back(std::stod(strValues[j]));

            measures.push_back(CObjectMeasure(measure, values, std::stoi(strData[1]), strData[2]));
        }
        m_measures.push_back(measures);
    }
}

void CBlobMeasureIO::save(const std::string &path)
{
    CWorkflowTaskIO::save(path);
    auto extension = Utils::File::extension(path);

    if (extension == ".csv")
        saveCSV(path);
    else if (extension == ".json")
        saveJSON(path);
    else
        throw CException(CoreExCode::NOT_IMPLEMENTED, "Export format not available yet", __func__, __FILE__, __LINE__);
}

QJsonObject CBlobMeasureIO::toJsonInternal() const
{
    QJsonArray objects;
    for (size_t i=0; i<m_measures.size(); ++i)
    {
        QJsonArray measures;
        for (size_t j=0; j<m_measures[i].size(); ++j)
        {
            QJsonObject measure = m_measures[i][j].toJson();
            measures.append(measure);
        }
        objects.append(measures);
    }

    QJsonObject root;
    root["objects"] = objects;
    return root;
}

std::string CBlobMeasureIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CBlobMeasureIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

void CBlobMeasureIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading blob measures: invalid JSON structure", __func__, __FILE__, __LINE__);

    QJsonObject root = jsonDoc.object();
    if (root.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading blob measures: empty JSON structure", __func__, __FILE__, __LINE__);

    m_measures.clear();
    QJsonArray objectArray = root["objects"].toArray();

    for (int i=0; i<objectArray.size(); ++i)
    {
        ObjectMeasures measures;
        QJsonArray measuresArray = objectArray[i].toArray();

        for (int j=0; j<measuresArray.size(); ++j)
        {
            CObjectMeasure objMeasure;
            objMeasure.fromJson(measuresArray[j].toObject());
            measures.push_back(objMeasure);
        }
        m_measures.push_back(measures);
    }
}

void CBlobMeasureIO::saveCSV(const std::string &path) const
{
    std::ofstream file;
    file.open(path, std::ios::out | std::ios::trunc);

    auto names = getMeasuresNames();
    file << "Object;Id;Category;";

    for(auto it=names.begin(); it!=names.end(); ++it)
        file << *it + ";";
    file << "\n";

    //Iterate throw objects (ie lines)
    for(size_t i=0; i<m_measures.size(); ++i)
    {
        std::map<std::string, std::string>  mapNameValues;
        file << std::to_string(i) + ";";
        file << std::to_string(m_measures[i][0].m_graphicsId) + ";";
        file << m_measures[i][0].m_label + ";";

        //Iterate throw object measures (ie columns)
        for(size_t j=0; j<m_measures[i].size(); ++j)
        {
            std::string strValues;
            auto measure = m_measures[i][j];

            for(size_t k=0; k<measure.m_values.size(); ++k)
            {
                strValues += std::to_string(measure.m_values[k]);
                if(k != measure.m_values.size() - 1)
                    strValues += "-";
            }
            mapNameValues.insert(std::make_pair(m_measures[i][j].m_measure.m_name, strValues));
        }

        for(auto it=names.begin(); it!=names.end(); ++it)
        {
            auto itMeasure = mapNameValues.find(*it);
            if(itMeasure == mapNameValues.end())
                file << ";";
            else
                file << itMeasure->second + ";";
        }
        file << "\n";
    }
    file.close();
}

void CBlobMeasureIO::saveJSON(const std::string &path) const
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson(QJsonDocument::Compact));
}

std::shared_ptr<CBlobMeasureIO> CBlobMeasureIO::clone() const
{
    return std::static_pointer_cast<CBlobMeasureIO>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CBlobMeasureIO::cloneImp() const
{
    return std::shared_ptr<CBlobMeasureIO>(new CBlobMeasureIO(*this));
}
