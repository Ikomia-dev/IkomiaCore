// Copyright (C) 2023 Ikomia SAS
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

#include <QObject>
#include <QJsonParseError>

#include "CJsonIO.h"
#include "Main/CoreTools.hpp"
#include "CException.h"



CJsonIO::CJsonIO() :
    CJsonIO(QJsonDocument(), "JsonIO")
{ }

CJsonIO::CJsonIO(const std::string &name) :
    CJsonIO(QJsonDocument(), name)
{ }

CJsonIO::CJsonIO(const QJsonDocument& rootJSON, const std::string& name) :
    CWorkflowTaskIO(IODataType::JSON, name),
    m_rootJSON(rootJSON)
{
    m_description = QObject::tr("Data stored in JSON format.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
}

CJsonIO::CJsonIO(const CJsonIO &io) :
    CWorkflowTaskIO(io),
    m_rootJSON(io.m_rootJSON)
{ }

CJsonIO::CJsonIO(const CJsonIO &&io) :
    CWorkflowTaskIO(io),
    m_rootJSON(std::move(io.m_rootJSON))
{ }

CJsonIO::~CJsonIO()
{ }

CJsonIO& CJsonIO::operator = (const CJsonIO &io)
{
    CWorkflowTaskIO::operator = (io);
    m_rootJSON = io.m_rootJSON;

    return *this;
}

CJsonIO& CJsonIO::operator = (const CJsonIO &&io)
{
    CWorkflowTaskIO::operator = (io);
    m_rootJSON = std::move(io.m_rootJSON);

    return *this;
}

std::string CJsonIO::repr() const
{
    std::stringstream s;
    s << "CJsonIO(" << m_name << ")";
    return s.str();
}

bool CJsonIO::isDataAvailable() const
{
    return !(m_rootJSON.isNull());
}

void CJsonIO::clearData()
{
    m_rootJSON = QJsonDocument();
}

QJsonDocument CJsonIO::getData() const
{
    return m_rootJSON;
}

void CJsonIO::setData(const QJsonDocument &doc)
{
    m_rootJSON = doc;
}

void CJsonIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "Invalid file format, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    m_rootJSON = QJsonDocument::fromJson(jsonFile.readAll());
    if(m_rootJSON.isNull() || m_rootJSON.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading JSON I/O: invalid JSON structure", __func__, __FILE__, __LINE__);
}

void CJsonIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    jsonFile.write(m_rootJSON.toJson(QJsonDocument::Compact));
}

// FIXME : this method is set for compatibility with the 'doDisplayText' signal
// It will be able to removed if a specific doDisplayJson is created
std::string CJsonIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "indented"};
    return toJson(options);
}

// FIXME : this method is set for compatibility with the 'doDisplayText' signal
// It will be able to removed if a specific doDisplayJson is created
std::string CJsonIO::toJson(const std::vector<std::string>& options) const
{
    // By default, the JSON output is 'compact'
    QJsonDocument::JsonFormat format = QJsonDocument::Compact;

    // We check if the option 'json_format' is defined into the option list
    if(std::find(options.begin(), options.end(), "json_format") == options.end())
    {
        // This method intends to arrange data in a JSON form but the option
        // 'json_format' was not found inside 'options': an exception is thrown
        throw CException(
            CoreExCode::INVALID_PARAMETER,
            "The option 'json_format' was not set inside the option list",
            __func__, __FILE__, __LINE__
        );
    }

    // If 'indented' is defined inside 'options' then we modify the output's style
    if(std::find(options.begin(), options.end(), "indented") != options.end())
    {
        format = QJsonDocument::Indented;
    }

    return toString(format);
}

std::string CJsonIO::toString(QJsonDocument::JsonFormat format) const
{
    return m_rootJSON.toJson(format).toStdString();
}

void CJsonIO::fromString(const std::string &str)
{
    // Variable used to store errors
    QJsonParseError error;

    // The 'str' variable is parsed using the Qt's library
    QJsonDocument result = QJsonDocument::fromJson(QByteArray::fromStdString(str), &error);
    if(result.isNull())
    {
        throw CException(
            CoreExCode::INVALID_JSON_FORMAT,
            QString("Invalid JSON document : %1").arg(error.errorString()).toStdString(),
            __func__, __FILE__, __LINE__
        );
    }

    // Everything is fine: result is stored into the data member
    m_rootJSON = result;
}

std::shared_ptr<CJsonIO> CJsonIO::clone() const
{
    return std::static_pointer_cast<CJsonIO>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CJsonIO::cloneImp() const
{
    return std::shared_ptr<CJsonIO>(new CJsonIO(*this));
}
