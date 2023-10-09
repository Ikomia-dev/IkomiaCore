/*
 * Copyright (C) 2023 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the Ikomia API libraries.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include "CScene3dIO.h"
#include "Main/CoreDefine.hpp"
#include "Main/CoreTools.hpp"
#include "CException.h"
#include "ExceptionCode.hpp"

#include <iostream>
#include <memory>
#include <QFile>
#include <QString>
#include <QJsonDocument>


CScene3dIO::CScene3dIO() :
    CWorkflowTaskIO(IODataType::SCENE_3D, "CScene3dIO")
{
    // Data will be saved in JSON format
    setSaveFormat(DataFileFormat::JSON);
}

CScene3dIO::CScene3dIO(const std::string &name) :
    CWorkflowTaskIO(IODataType::SCENE_3D, name)
{
    // Data will be saved in JSON format
    setSaveFormat(DataFileFormat::JSON);
}

CScene3dIO::CScene3dIO(const CScene3dIO &io) :
    CWorkflowTaskIO(io),
    m_scene3d(io.getScene3d())
{ }

CScene3dIO::CScene3dIO(const CScene3dIO &&io) :
    CWorkflowTaskIO(io),
    m_scene3d(io.getScene3d())
{ }

CScene3dIO& CScene3dIO::operator = (const CScene3dIO &io)
{
    CWorkflowTaskIO::operator = (io);
    m_scene3d = io.getScene3d();

    return *this;
}

CScene3dIO& CScene3dIO::operator = (const CScene3dIO &&io)
{
    CWorkflowTaskIO::operator = (io);
    m_scene3d = io.getScene3d();

    return *this;
}

std::string CScene3dIO::repr() const
{
    std::stringstream s;
    s << "CScene3dIO(" << m_name << ")";
    return s.str();
}

bool CScene3dIO::isDataAvailable() const
{
    // Data are available if there is one or more layers in the 3D scene
    // (event if these layers are empty)
    return (m_scene3d.getLstLayers().size() > 0);
}

const CScene3d& CScene3dIO::getScene3d() const
{
    return m_scene3d;
}

void CScene3dIO::setScene3d(const CScene3d& scene3d)
{
    m_scene3d = scene3d;
}

std::shared_ptr<CScene3dIO> CScene3dIO::clone() const
{
    return std::static_pointer_cast<CScene3dIO>(cloneImp());
}

void CScene3dIO::clearData()
{
    m_scene3d.clear();
    m_infoPtr = nullptr;
}

void CScene3dIO::save()
{
    // The output directory and name are automatically computed
    save(getSavePath());
}

void CScene3dIO::save(const std::string &path)
{
    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson());
}

void CScene3dIO::load(const std::string &path)
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

std::string CScene3dIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CScene3dIO::toJson(const std::vector<std::string>& options) const
{
    return toFormattedJson(toJsonInternal(), options);
}

void CScene3dIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading text detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

std::shared_ptr<CWorkflowTaskIO> CScene3dIO::cloneImp() const
{
    return std::shared_ptr<CScene3dIO>(new CScene3dIO(*this));
}

QJsonDocument CScene3dIO::toJsonInternal() const
{
    QJsonObject root;
    root["scene3d"] = m_scene3d.toJson();

    return QJsonDocument(root);
}

void CScene3dIO::fromJsonInternal(const QJsonDocument& doc)
{
    clearData();

    QJsonObject root = doc.object();
    m_scene3d = CScene3d::fromJson(root["scene3d"].toObject());
}
