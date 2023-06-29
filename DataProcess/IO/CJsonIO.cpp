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
    CJsonIO(QJsonDocument(), "CJsonIO")
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
    // Self-assignment check...
    if(this != &io)
    {
        // Not a self-assignment : data are copied
        CWorkflowTaskIO::operator = (io);
        m_rootJSON = io.m_rootJSON;
    }
    return *this;
}

CJsonIO& CJsonIO::operator = (const CJsonIO &&io)
{
    // Self-assignment check...
    if(this != &io)
    {
        // Not a self-assignment : data are copied
        CWorkflowTaskIO::operator = (io);
        m_rootJSON = std::move(io.m_rootJSON);
    }
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
    throw CException(
        CoreExCode::NOT_IMPLEMENTED,
        "Not implemented yet...",
        __func__, __FILE__, __LINE__
    );
}

void CJsonIO::save(const std::string &path)
{
    throw CException(
        CoreExCode::NOT_IMPLEMENTED,
        "Not implemented yet...",
        __func__, __FILE__, __LINE__
    );
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
