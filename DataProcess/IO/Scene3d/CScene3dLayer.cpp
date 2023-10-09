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

#include "CScene3d.h"
#include "CScene3dImage2d.h"
#include "CScene3dShapePoint.h"
#include "CScene3dShapeCircle.h"
#include "CScene3dShapePoly.h"

#include <QJsonArray>
#include <QJsonObject>


CScene3dLayer::CScene3dLayer() :
    m_isVisible(true),
    m_lstObjects(std::vector<CScene3dObjectPtr>())
{ }

CScene3dLayer::CScene3dLayer(const CScene3dLayer& layer) :
    m_isVisible(layer.isVisible()),
    m_lstObjects(layer.getLstObjects())
{ }

CScene3dLayer& CScene3dLayer::operator = (const CScene3dLayer& layer)
{
    m_isVisible = layer.isVisible();
    m_lstObjects = layer.getLstObjects();

    return *this;
}

bool CScene3dLayer::isVisible() const
{
    return m_isVisible;
}

void CScene3dLayer::setVisibility(bool isVisible)
{
    m_isVisible = isVisible;
}

void CScene3dLayer::addObject(CScene3dObjectPtr obj)
{
    // The new object is put at the end of the list
    m_lstObjects.push_back(obj);
}

const std::vector<CScene3dObjectPtr>& CScene3dLayer::getLstObjects() const
{
    return m_lstObjects;
}

QJsonObject CScene3dLayer::toJson() const
{
    QJsonObject obj;

    QJsonArray lst_objects;
    for(auto object : m_lstObjects)
    {
        lst_objects.push_back(object->toJson());
    }
    obj["isVisible"] = m_isVisible;
    obj["objects"] = lst_objects;

    return obj;
}

CScene3dLayer CScene3dLayer::fromJson(const QJsonObject& obj)
{
    CScene3dLayer layer;

    layer.setVisibility(obj["isVisible"].toBool());

    QJsonArray lstObjectsArray = obj["objects"].toArray();
    for(auto x: lstObjectsArray)
    {
        QJsonObject objectObject = x.toObject();
        std::string objectObjectKind = objectObject["kind"].toString().toStdString();

        if(objectObjectKind == "IMAGE2D")
        {
            layer.addObject(
                CScene3dImage2d::fromJson(objectObject)
            );
        }
        else if(objectObjectKind == "SHAPE_POINT")
        {
            layer.addObject(
                CScene3dShapePoint::fromJson(objectObject)
            );
        }
        else if(objectObjectKind == "SHAPE_CIRCLE")
        {
            layer.addObject(
                CScene3dShapeCircle::fromJson(objectObject)
            );
        }
        else if(objectObjectKind == "SHAPE_POLY")
        {
            layer.addObject(
                CScene3dShapePoly::fromJson(objectObject)
            );
        }
        else
        {
            throw CException(CoreExCode::INVALID_JSON_FORMAT, "Unknown object type", __func__, __FILE__, __LINE__);
        }
    }

    return layer;
}
