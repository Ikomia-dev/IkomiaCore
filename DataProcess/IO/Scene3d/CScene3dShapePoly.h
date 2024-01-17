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

#ifndef CSCENE3DSHAPEPOLY_H
#define CSCENE3DSHAPEPOLY_H

#include <memory>
#include <tuple>
#include <vector>
#include <QJsonObject>
#include "DataProcessGlobal.hpp"
#include "CScene3dColor.h"
#include "CScene3dCoord.h"
#include "CScene3dObject.h"


/**
 * Alias onto a tuple made of a coordinate and a color
 */
using CScene3dPt = std::tuple<CScene3dCoord, CScene3dColor>;

class CScene3dShapePoly;

/**
 * Alias onto the CScene3dShapePoly's shared pointer type.
 */
using CScene3dShapePolyPtr = std::shared_ptr<CScene3dShapePoly>;


/**
 * @brief The CScene3dShapePoly class represents a polygon into the 3D scene.
 * This polygon is defined by a list of points (position + color) and the
 * size of the line connecting these points.
 */
class DATAPROCESSSHARED_EXPORT CScene3dShapePoly : public CScene3dObject
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dShapePoly();

    /**
     * @brief Custom constructor.
     * @param lstPts: points of the polygon.
     * @param lineWidth: width of the line connecting the polygon's points.
     * @param isVisible: true if the circle should be displayed, false otherwise.
     */
    CScene3dShapePoly(
        const std::vector<CScene3dPt>& lstPts,
        double lineWidth,
        bool isVisible
    );

    /**
     * @brief Copy constructor.
     */
    CScene3dShapePoly(const CScene3dShapePoly &shape);

    /**
     * @brief Assignment operator.
     */
    CScene3dShapePoly& operator = (const CScene3dShapePoly& shape);

    /**
     * @brief Return the polygon's points.
     */
    const std::vector<CScene3dPt>& getLstPts() const;

    /**
     * @brief Add a point to the polygon.
     * @param pt: point (coordinates + color) to add to the polygon.
     */
    void addPoint(const CScene3dPt &pt);

    /**
     * @brief Add a point to the polygon.
     * @param coord: coordinates of the point to add to the polygon.
     * @param color: color of the point to add to the polygon.
     */
    void addPoint(const CScene3dCoord &coord, const CScene3dColor &color);

    /**
     * Add a point to the polygon.
     * @param coordX: the X component of the point to add to the polygon.
     * @param coordY: the Y component of the point to add to the polygon.
     * @param coordZ: the Z component of the point to add to the polygon.
     * @param cs: the coordinate system associated to the point to add to the polygon.
     * @param colorR: the red component of the color of the point to add to the polygon.
     * @param colorG: the green component of the color of the point to add to the polygon.
     * @param colorB: the blue component of the color of the point to add to the polygon.
     */
    void addPoint(double coordX, double coordY, double coordZ, CScene3dCoordSystem cs, double colorR, double colorG, double colorB);

    /**
     * @brief Return the width of the line connecting the polygon's points.
     */
    double getLineWidth() const;

    /**
     * @brief Set the width of the line connecting the polygon's points.
     * @param lineWidth: width of the line connecting the polygon's points.
     */
    void setLineWidth(double lineWidth);

    /**
     * @brief Static method used to create an new 'CScene3dShapePoly' instance.
     * This polygon has no point defined. Points should be added by one of the
     * 'addPoint()' methods.
     * @param lineWidth: width of the line connecting the polygon's points.
     * @param isVisible: true if the circle should be displayed, false otherwise.
     * @return Return a shared_ptr of the created instance.
     */
    static CScene3dShapePolyPtr create(
        double lineWidth,
        bool isVisible
    );

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const override;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3dShapePolyPtr fromJson(const QJsonObject& obj);

    /**
     * @brief Static method used to create an new 'CScene3dShapePoly' instance.
     * @param lstPts: points of the polygon.
     * @param lineWidth: width of the line connecting the polygon's points.
     * @param isVisible: true if the circle should be displayed, false otherwise.
     * @return Return a shared_ptr of the created instance.
     */
    static CScene3dShapePolyPtr create(
        const std::vector<CScene3dPt>& lstPts,
        double lineWidth,
        bool isVisible
    );

protected:
    /**
     * List of points (coordinates + color)
     */
    std::vector<CScene3dPt> m_lstPts;

    /**
     * Width of the line connecting the polygon's points.
     */
    double m_lineWidth;
};

#endif // CSCENE3DSHAPEPOLY_H
