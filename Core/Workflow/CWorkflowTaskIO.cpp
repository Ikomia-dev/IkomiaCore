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

#include "CWorkflowTaskIO.h"
#include "Main/CoreTools.hpp"
#include <ostream>

CWorkflowTaskIO::CWorkflowTaskIO()
{
}

CWorkflowTaskIO::CWorkflowTaskIO(IODataType dataType)
{
    m_dataType = dataType;
}

CWorkflowTaskIO::CWorkflowTaskIO(IODataType dataType, const std::string &name)
{
    m_dataType = dataType;
    m_name = name;
}

CWorkflowTaskIO::CWorkflowTaskIO(const CWorkflowTaskIO& io)
{
    m_name = io.m_name;
    m_description = io.m_description;
    m_saveFolder = io.m_saveFolder;
    m_saveBaseName = io.m_saveBaseName;
    m_dataType = io.m_dataType;
    m_saveFormat = io.m_saveFormat;
    m_dimCount = io.m_dimCount;
    m_bAutoSave = io.m_bAutoSave;
    m_bDisplayable = io.m_bDisplayable;
    m_infoPtr = io.m_infoPtr;
}

CWorkflowTaskIO::CWorkflowTaskIO(const CWorkflowTaskIO&& io)
{
    m_name = std::move(io.m_name);
    m_description = std::move(io.m_description);
    m_saveFolder = std::move(io.m_saveFolder);
    m_saveBaseName = std::move(io.m_saveBaseName);
    m_dataType = std::move(io.m_dataType);
    m_saveFormat = std::move(io.m_saveFormat);
    m_dimCount = std::move(io.m_dimCount);
    m_bAutoSave = std::move(io.m_bAutoSave);
    m_bDisplayable = std::move(io.m_bDisplayable);
    m_infoPtr = std::move(io.m_infoPtr);
}

CWorkflowTaskIO &CWorkflowTaskIO::operator=(const CWorkflowTaskIO &io)
{
    m_name = io.m_name;
    m_description = io.m_description;
    m_saveFolder = io.m_saveFolder;
    m_saveBaseName = io.m_saveBaseName;
    m_dataType = io.m_dataType;
    m_saveFormat = io.m_saveFormat;
    m_dimCount = io.m_dimCount;
    m_bAutoSave = io.m_bAutoSave;
    m_bDisplayable = io.m_bDisplayable;
    m_infoPtr = io.m_infoPtr;
    return *this;
}

CWorkflowTaskIO &CWorkflowTaskIO::operator=(const CWorkflowTaskIO&& io)
{
    m_name = std::move(io.m_name);
    m_description = std::move(io.m_description);
    m_saveFolder = std::move(io.m_saveFolder);
    m_saveBaseName = std::move(io.m_saveBaseName);
    m_dataType = std::move(io.m_dataType);
    m_saveFormat = std::move(io.m_saveFormat);
    m_dimCount = std::move(io.m_dimCount);
    m_bAutoSave = std::move(io.m_bAutoSave);
    m_bDisplayable = std::move(io.m_bDisplayable);
    m_infoPtr = std::move(io.m_infoPtr);
    return *this;
}

std::ostream& operator<<(std::ostream& os, const CWorkflowTaskIO& io)
{
    io.to_ostream(os);
    return os;
}

CWorkflowTaskIO::~CWorkflowTaskIO()
{
    deleteTemporaryFiles();
}

std::string CWorkflowTaskIO::repr() const
{
    std::stringstream s;
    s << "CWorkflowTaskIO(" << Utils::Workflow::getIODataEnumName(m_dataType) << ", " << m_name << ")";
    return s.str();
}

std::string CWorkflowTaskIO::getName() const
{
    return m_name;
}

std::string CWorkflowTaskIO::getDescription() const
{
    return m_description;
}

IODataType CWorkflowTaskIO::getDataType() const
{
    return m_dataType;
}

DataFileFormat CWorkflowTaskIO::getSaveFormat() const
{
    return m_saveFormat;
}

std::vector<DataFileFormat> CWorkflowTaskIO::getPossibleSaveFormats() const
{
    if(m_saveFormat == DataFileFormat::NONE)
        return std::vector<DataFileFormat>();
    else
    {
        std::vector<DataFileFormat> formats = {m_saveFormat};
        return formats;
    }
}

std::string CWorkflowTaskIO::getSavePath() const
{
    return m_saveFolder + m_saveBaseName + Utils::Data::getFileFormatExtension(m_saveFormat);
}

size_t CWorkflowTaskIO::getDimensionCount() const
{
    return m_dimCount;
}

size_t CWorkflowTaskIO::getUnitElementCount() const
{
    return 1;
}

CDataInfoPtr CWorkflowTaskIO::getDataInfo()
{
    return m_infoPtr;
}

std::string CWorkflowTaskIO::getSourceFilePath()
{
    if(m_infoPtr)
        return m_infoPtr->getFileName();
    else
        return "";
}

bool CWorkflowTaskIO::isDataAvailable() const
{
    return false;
}

bool CWorkflowTaskIO::isAutoSave() const
{
    return m_bAutoSave;
}

bool CWorkflowTaskIO::isDisplayable() const
{
    return m_bDisplayable;
}

bool CWorkflowTaskIO::isComposite() const
{
    return false;
}

void CWorkflowTaskIO::setDataType(IODataType type)
{
    m_dataType = type;
}

void CWorkflowTaskIO::setName(const std::string &name)
{
    m_name = name;
}

void CWorkflowTaskIO::setDimensionCount(size_t nb)
{
    m_dimCount = nb;
}

void CWorkflowTaskIO::setSaveInfo(const std::string &folder, const std::string &baseName)
{
    m_saveFolder = folder;
    m_saveBaseName = baseName;
}

void CWorkflowTaskIO::setDescription(const std::string &description)
{
    m_description = description;
}

void CWorkflowTaskIO::setAutoSave(bool bEnable)
{
    m_bAutoSave = bEnable;
}

void CWorkflowTaskIO::setSaveFormat(DataFileFormat format)
{
    m_saveFormat = format;
}

void CWorkflowTaskIO::setDisplayable(bool bEnable)
{
    m_bDisplayable = bEnable;
}

void CWorkflowTaskIO::clearData()
{
}

std::shared_ptr<CWorkflowTaskIO> CWorkflowTaskIO::clone() const
{
    return cloneImp();
}

void CWorkflowTaskIO::copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    // Does nothing by default, should be overriden if needed
    Q_UNUSED(ioPtr);
}

void CWorkflowTaskIO::copyStaticData(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    if(ioPtr)
        m_dimCount = ioPtr->getDimensionCount();
}

void CWorkflowTaskIO::addTemporaryFile(const std::string &path)
{
    m_tempFiles.insert(path);
}

void CWorkflowTaskIO::load(const std::string &path)
{
    Q_UNUSED(path);
}

void CWorkflowTaskIO::save()
{
}

void CWorkflowTaskIO::save(const std::string &path)
{
    Q_UNUSED(path);
}

std::string CWorkflowTaskIO::getClassName(IODataType ioDataType)
{
    switch(ioDataType)
    {
        case IODataType::NONE:
            return "";

        case IODataType::IMAGE:
        case IODataType::IMAGE_BINARY:
        case IODataType::IMAGE_LABEL:
        case IODataType::VOLUME:
        case IODataType::VOLUME_BINARY:
        case IODataType::VOLUME_LABEL:
        case IODataType::POSITION:
        case IODataType::DESCRIPTORS:
            return "CImageIO";

        case IODataType::VIDEO:
        case IODataType::VIDEO_BINARY:
        case IODataType::VIDEO_LABEL:
        case IODataType::LIVE_STREAM:
        case IODataType::LIVE_STREAM_BINARY:
        case IODataType::LIVE_STREAM_LABEL:
            return "CVideoIO";

        case IODataType::INPUT_GRAPHICS:
            return "CGraphicsInput";

        case IODataType::OUTPUT_GRAPHICS:
            return "CGraphicsOutput";

        case IODataType::BLOB_VALUES:
            return "CBlobMeasureIO";

        case IODataType::NUMERIC_VALUES:
            return "CNumericIO";

        case IODataType::WIDGET:
            return "CWidgetOutput";

        case IODataType::PROJECT_FOLDER:
        case IODataType::FOLDER_PATH:
        case IODataType::FILE_PATH:
            return "CPathIO";

        case IODataType::ARRAY:
            return "CArrayIO";

        case IODataType::DATA_DICT:
            return "CWorkflowTaskIO";

        case IODataType::OBJECT_DETECTION:
            return "CObjectDetectionIO";

        case IODataType::INSTANCE_SEGMENTATION:
            return "CInstanceSegIO";

        case IODataType::SEMANTIC_SEGMENTATION:
            return "CSemanticSegIO";

        case IODataType::KEYPOINTS:
            return "CKeypointsIO";

        case IODataType::TEXT:
            return "CTextIO";

        case IODataType::JSON:
            return "CJsonIO";

        default: return "";
    }
}

std::string CWorkflowTaskIO::toJson() const
{
    return std::string();
}

std::string CWorkflowTaskIO::toJson(const std::vector<std::string>& options) const
{
    Q_UNUSED(options);
    return std::string();
}

void CWorkflowTaskIO::fromJson(const std::string &jsonStr)
{
    Q_UNUSED(jsonStr);
}

std::string CWorkflowTaskIO::toFormattedJson(const QJsonDocument &doc, const std::vector<std::string> &options) const
{
    std::string format = "compact";
    auto it = std::find(options.begin(), options.end(), "json_format");

    if (it != options.end())
    {
        size_t index = it - options.begin() + 1;
        format = options[index];
    }

    if (format == "compact")
        return doc.toJson(QJsonDocument::Compact).toStdString();
    else
        return doc.toJson(QJsonDocument::Indented).toStdString();
}

void CWorkflowTaskIO::to_ostream(std::ostream &os) const
{
    os << "-----------------------------------" << std::endl;
    os << "-\t" << "I/O: " << m_name << std::endl;
    os << "-----------------------------------" << std::endl;
    os << "Description: " << m_description << std::endl;
    os << "Save folder: " << m_saveFolder << std::endl;
    os << "Auto-save: " << m_bAutoSave << std::endl;
    os << "Data type: " << Utils::Workflow::getIODataName(m_dataType).toStdString() << std::endl;
    os << "Save format: " << Utils::Data::getFileFormatExtension(m_saveFormat) << std::endl;
    os << "Dimension count: " << m_dimCount << std::endl;
    os << "Displayable: " << m_bDisplayable << std::endl;
    os << "Composite: " << isComposite() << std::endl;
    os << "Data available: " << isDataAvailable() << std::endl;

    if(m_infoPtr)
        os << *(m_infoPtr);
}

std::shared_ptr<CWorkflowTaskIO> CWorkflowTaskIO::cloneImp() const
{
    return std::shared_ptr<CWorkflowTaskIO>(new CWorkflowTaskIO(*this));
}

void CWorkflowTaskIO::deleteTemporaryFiles()
{
    for(auto it=m_tempFiles.begin(); it!=m_tempFiles.end(); ++it)
    {
        try
        {
            boost::filesystem::path path(*it);
            boost::filesystem::remove(path);
        }
        catch (const boost::filesystem::filesystem_error& e)
        {
            Utils::print(e.code().message(), QtWarningMsg);
        }
    }
}
