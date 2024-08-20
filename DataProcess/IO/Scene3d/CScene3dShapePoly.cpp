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

#include "CScene3dShapePoly.h"
#include "CScene3dObject.h"
#include "CException.h"
#include "ExceptionCode.hpp"

#include <memory>
#include <QJsonArray>
#include <QJsonObject>


CScene3dShapePoly::CScene3dShapePoly() :
    CScene3dObject(),
    m_lstPts(std::vector<CScene3dPt>()),
    m_lineWidth(1.0)
{ }

CScene3dShapePoly::CScene3dShapePoly(
    const std::vector<CScene3dPt>& lstPts,
    double lineWidth,
    bool isVisible
) :
    CScene3dObject(isVisible),
    m_lstPts(lstPts),
    m_lineWidth(lineWidth)
{ }

CScene3dShapePoly::CScene3dShapePoly(const CScene3dShapePoly &shape) :
    CScene3dObject(shape),
    m_lstPts(shape.getLstPts()),
    m_lineWidth(shape.getLineWidth())
{ }

CScene3dShapePoly& CScene3dShapePoly::operator = (const CScene3dShapePoly& shape)
{
    CScene3dObject::operator = (shape);
    m_lstPts = shape.getLstPts();
    m_lineWidth = shape.getLineWidth();

    return *this;
}

const std::vector<CScene3dPt>& CScene3dShapePoly::getLstPts() const
{
    return m_lstPts;
}

void CScene3dShapePoly::addPoint(const CScene3dPt &pt)
{
    m_lstPts.push_back(pt);
}

void CScene3dShapePoly::addPoint(const CScene3dCoord &coord, const CScene3dColor &color)
{
    m_lstPts.push_back(
        CScene3dPt(coord, color)
    );
}

void CScene3dShapePoly::addPoint(
    double coordX1, double coordX2, double coordX3, CScene3dCoordSystem cs,
    double colorR, double colorG, double colorB
) {
    m_lstPts.push_back(
        CScene3dPt(
            CScene3dCoord(coordX1, coordX2, coordX3, cs),
            CScene3dColor(colorR, colorG, colorB)
        )
    );
}

double CScene3dShapePoly::getLineWidth() const
{
    return m_lineWidth;
}

void CScene3dShapePoly::setLineWidth(double lineWidth)
{
    m_lineWidth = lineWidth;
}

QJsonObject CScene3dShapePoly::toJson() const
{
    QJsonObject obj = CScene3dObject::toJson();

    QJsonArray lstPts;
    for(auto pt : m_lstPts)
    {
        QJsonObject point;
        point["position"] = std::get<0>(pt).toJson();
        point["color"] = std::get<1>(pt).toJson();

        lstPts.push_back(point);
    }

    obj["kind"] = "SHAPE_POLY";
    obj["lstPts"] = lstPts;
    obj["lineWidth"] = m_lineWidth;

    return obj;
}

CScene3dShapePolyPtr CScene3dShapePoly::fromJson(const QJsonObject& obj)
{
    if(obj["kind"] != "SHAPE_POLY")
    {
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid object type: 'SHAPE_POINT' expected", __func__, __FILE__, __LINE__);
    }

    std::vector<CScene3dPt> lstPts;

    QJsonArray lstPtsArray = obj["lstPts"].toArray();
    for(auto point: lstPtsArray)
    {
        QJsonObject pointObject = point.toObject();

        lstPts.push_back(
            CScene3dPt(
                CScene3dCoord::fromJson(pointObject["position"].toObject()),
                CScene3dColor::fromJson(pointObject["color"].toObject())
            )
        );
    }

    return CScene3dShapePoly::create(
        lstPts,
        obj["lineWidth"].toDouble(),
        obj["isVisible"].toBool()
    );
}

CScene3dShapePolyPtr CScene3dShapePoly::create(
    double lineWidth,
    bool isVisible
) {
    return std::make_shared<CScene3dShapePoly>(
        std::vector<CScene3dPt>(),
        lineWidth,
        isVisible
    );
}

CScene3dShapePolyPtr CScene3dShapePoly::create(
    const std::vector<CScene3dPt>& lstPts,
    double lineWidth,
    bool isVisible
) {
    return std::make_shared<CScene3dShapePoly>(
        lstPts,
        lineWidth,
        isVisible
    );
}

