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

#include "CScene3dObject.h"

#include <QJsonObject>


CScene3dObject::CScene3dObject() :
    m_isVisible(true)
{ }

CScene3dObject::CScene3dObject(bool isVisible) :
    m_isVisible(isVisible)
{ }

CScene3dObject::CScene3dObject(const CScene3dObject& obj) :
    m_isVisible(obj.isVisible())
{ }

CScene3dObject::~CScene3dObject()
{ }

CScene3dObject& CScene3dObject::operator = (const CScene3dObject &obj)
{
    // To avoid invalid self-assignment
    if(this != &obj)
    {
        m_isVisible = obj.isVisible();
    }

    return *this;
}

bool CScene3dObject::isVisible() const
{
    return m_isVisible;
}

void CScene3dObject::setVisibility(bool isVisible)
{
    m_isVisible = isVisible;
}

QJsonObject CScene3dObject::toJson() const
{
    QJsonObject obj;
    obj["isVisible"] = m_isVisible;

    return obj;
}
