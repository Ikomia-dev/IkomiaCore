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

#include "CScene3dShapeCircle.h"
#include "CScene3dObject.h"
#include "CException.h"
#include "ExceptionCode.hpp"

#include <memory>
#include <QJsonObject>


CScene3dShapeCircle::CScene3dShapeCircle() :
    CScene3dObject(),
    m_center(CScene3dCoord(0.0, 0.0, 0.0)),
    m_color(CScene3dColor(1.0, 1.0, 1.0)),
    m_radius(0.0)
{ }

CScene3dShapeCircle::CScene3dShapeCircle(
    double centerX, double centerY, double centerZ, CScene3dCoordSystem cs,
    double colorR, double colorG, double colorB,
    double radius,
    bool isVisible
) :
    CScene3dObject(isVisible),
    m_center(CScene3dCoord(centerX, centerY, centerZ, cs)),
    m_color(CScene3dColor(colorR, colorG, colorB)),
    m_radius(radius)
{ }

CScene3dShapeCircle::CScene3dShapeCircle(const CScene3dShapeCircle &shape) :
    CScene3dObject(shape),
    m_center(shape.getCenter()),
    m_color(shape.getColor()),
    m_radius(shape.getRadius())
{ }

CScene3dShapeCircle& CScene3dShapeCircle::operator = (const CScene3dShapeCircle& shape)
{
    CScene3dObject::operator = (shape);
    m_center = shape.getCenter();
    m_color = shape.getColor();
    m_radius = shape.getRadius();

    return *this;
}

const CScene3dCoord& CScene3dShapeCircle::getCenter() const
{
    return m_center;
}

void CScene3dShapeCircle::setCenter(const CScene3dCoord& center)
{
    m_center = center;
}

void CScene3dShapeCircle::setCenter(double centerX, double centerY, double centerZ, CScene3dCoordSystem cs)
{
    m_center = CScene3dCoord(centerX, centerY, centerZ, cs);
}

const CScene3dColor& CScene3dShapeCircle::getColor() const
{
    return m_color;
}

void CScene3dShapeCircle::setColor(const CScene3dColor& color)
{
    m_color = color;
}

void CScene3dShapeCircle::setColor(double colorR, double colorG, double colorB)
{
    m_color = CScene3dColor(colorR, colorG, colorB);
}

double CScene3dShapeCircle::getRadius() const
{
    return m_radius;
}

void CScene3dShapeCircle::setRadius(double radius)
{
    m_radius = radius;
}

QJsonObject CScene3dShapeCircle::toJson() const
{
    QJsonObject obj = CScene3dObject::toJson();

    obj["kind"] = "SHAPE_CIRCLE";
    obj["center"] = m_center.toJson();
    obj["color"] = m_color.toJson();
    obj["radius"] = m_radius;

    return obj;
}

CScene3dShapeCirclePtr CScene3dShapeCircle::fromJson(const QJsonObject& obj)
{
    if(obj["kind"] != "SHAPE_CIRCLE")
    {
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid object type: 'SHAPE_CIRCLE' expected", __func__, __FILE__, __LINE__);
    }

    CScene3dCoord center = CScene3dCoord::fromJson(obj["center"].toObject());
    CScene3dColor color = CScene3dColor::fromJson(obj["color"].toObject());

    return CScene3dShapeCircle::create(
        center.getCoordX1(), center.getCoordX2(), center.getCoordX3(), center.getCoordSystem(),
        color.getColorR(), color.getColorG(), color.getColorB(),
        obj["radius"].toDouble(),
        obj["isVisible"].toBool()
    );
}

CScene3dShapeCirclePtr CScene3dShapeCircle::create(
    double centerX, double centerY, double centerZ, CScene3dCoordSystem cs,
    double colorR, double colorG, double colorB,
    double radius,
    bool isVisible
) {
    return std::make_shared<CScene3dShapeCircle>(
        centerX, centerY, centerZ, cs,
        colorR, colorG, colorB,
        radius,
        isVisible
    );
}
