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

#include "CScene3dImage2d.h"
#include "DataProcessTools.hpp"

#include <QJsonObject>


CScene3dImage2d::CScene3dImage2d() :
    CScene3dObject(),
    m_data(CMat())
{ }

CScene3dImage2d::CScene3dImage2d(const CMat &data, bool isVisible) :
    CScene3dObject(isVisible),
    m_data(data)
{ }

CScene3dImage2d::CScene3dImage2d(const CScene3dImage2d &img) :
    CScene3dObject(img),
    m_data(img.getData())
{ }

CScene3dImage2d& CScene3dImage2d::operator = (const CScene3dImage2d& img)
{
    // To avoid invalid self-assignment
    if(this != &img)
    {
        CScene3dObject::operator = (img);

        m_data = img.getData();
    }

    return *this;
}

CMat CScene3dImage2d::getData() const
{
    return m_data;
}

void CScene3dImage2d::setData(const CMat &data)
{
    m_data = data;
}

std::size_t CScene3dImage2d::getWidth() const
{
    return m_data.getNbCols();
}

std::size_t CScene3dImage2d::getHeight() const
{
    return m_data.getNbRows();
}

QJsonObject CScene3dImage2d::toJson() const
{
    QJsonObject obj = CScene3dObject::toJson();

    std::vector<std::string> options = {"json_format", "compact", "image_format", "jpg"};
    obj["image"] = QString::fromStdString(Utils::Image::toJson(m_data, options));
    obj["kind"] = "IMAGE2D";

    return obj;
}

CScene3dImage2dPtr CScene3dImage2d::fromJson(const QJsonObject& obj)
{
    if(obj["kind"] != "IMAGE2D")
    {
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid object type: 'IMAGE2D' expected", __func__, __FILE__, __LINE__);
    }

    CMat data = Utils::Image::fromJson(obj["image"].toString().toStdString());

    return CScene3dImage2d::create(
        data,
        obj["isVisible"].toBool()
    );
}

CScene3dImage2dPtr CScene3dImage2d::create(const CMat &data, bool isVisible)
{
    return std::make_shared<CScene3dImage2d>(data, isVisible);
}
