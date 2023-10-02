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

#include "CScene3dShapePoint.h"

#include <memory>

#include "CScene3dObject.h"


CScene3dShapePoint::CScene3dShapePoint() :
    CScene3dObject(),
    m_position(CScene3dCoord(0.0, 0.0, 0.0)),
    m_color(CScene3dColor(1.0, 1.0, 1.0)),
    m_size(1.0)
{ }

CScene3dShapePoint::CScene3dShapePoint(
    double positionX, double positionY, double positionZ, CScene3dCoordSystem cs,
    double colorR, double colorG, double colorB,
    double size,
    bool isVisible
) :
    CScene3dObject(isVisible),
    m_position(CScene3dCoord(positionX, positionY, positionZ, cs)),
    m_color(CScene3dColor(colorR, colorG, colorB)),
    m_size(size)
{ }

CScene3dShapePoint::CScene3dShapePoint(const CScene3dShapePoint &shape) :
    CScene3dObject(shape),
    m_position(shape.getPosition()),
    m_color(shape.getColor()),
    m_size(shape.getSize())
{ }

CScene3dShapePoint& CScene3dShapePoint::operator = (const CScene3dShapePoint& shape)
{
    if(this != &shape)
    {
        CScene3dObject::operator = (shape);
        m_position = shape.getPosition();
        m_color = shape.getColor();
        m_size = shape.getSize();
    }

    return *this;
}

const CScene3dCoord& CScene3dShapePoint::getPosition() const
{
    return m_position;
}

void CScene3dShapePoint::setPosition(const CScene3dCoord& position)
{
    m_position = position;
}

void CScene3dShapePoint::setPosition(double positionX, double positionY, double positionZ, CScene3dCoordSystem cs)
{
    m_position = CScene3dCoord(positionX, positionY, positionZ, cs);
}

const CScene3dColor& CScene3dShapePoint::getColor() const
{
    return m_color;
}

void CScene3dShapePoint::setColor(const CScene3dColor& color)
{
    m_color = color;
}

void CScene3dShapePoint::setColor(double colorR, double colorG, double colorB)
{
    m_color = CScene3dColor(colorR, colorG, colorB);
}

double CScene3dShapePoint::getSize() const
{
    return m_size;
}

void CScene3dShapePoint::setSize(double size)
{
    m_size = size;
}

CScene3dShapePointPtr CScene3dShapePoint::create(
    double positionX, double positionY, double positionZ, CScene3dCoordSystem cs,
    double colorR, double colorG, double colorB,
    double size,
    bool isVisible
) {
    return std::make_shared<CScene3dShapePoint>(
        positionX, positionY, positionZ, cs,
        colorR, colorG, colorB,
        size,
        isVisible
    );
}
