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

#ifndef CSCENE3DCOORD_H
#define CSCENE3DCOORD_H

#include <QJsonObject>


/**
 * @brief This enumeration is used to defined the coordinate system
 * associated to a point. Conversion of the point's coordinates can be made
 * during the rendering process if its coordinate system is not equal to
 * the 3d scene's coordinate system. The accepted values are:
 * - UNDEFINED: coordinate system not defined: no conversion will be made
 * - IMAGE: 2D image coordinate system (u, v)
 * - REAL_RIGHT_HANDED: 3D right-handed coordinate system (x, y, z)
 */
enum class CScene3dCoordSystem : unsigned char {
    UNDEFINED,
    IMAGE,
    REAL_RIGHT_HANDED,
};


/**
 * @brief The CScene3dCoord class represents a 3D coordinate (x1, x2, x3).
 * A coordinate system is associated to this coordinate.
 */
class CScene3dCoord
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dCoord();

    /**
     * @brief Custom constructor.
     * @param coordX1: the X1 component.
     * @param coordX2: the X2 component.
     * @param coordX3: the X3 component.
     * @param cs: the coordinate system associated to these coordinates.
     */
    CScene3dCoord(double coordX1, double coordX2, double coordX3, CScene3dCoordSystem cs=CScene3dCoordSystem::UNDEFINED);

    /**
     * @brief Copy constructor.
     */
    CScene3dCoord(const CScene3dCoord &coord);

    /**
     * @brief Assignment operator.
     */
    CScene3dCoord& operator = (const CScene3dCoord &coord);

    /**
     * @brief Return the X1 component.
     */
    double getCoordX1() const;

    /**
     * @brief Return the X2 component.
     */
    double getCoordX2() const;

    /**
     * @brief Return the X3 component.
     */
    double getCoordX3() const;

    /**
     * @brief Change the current coordinate.
     * @param coordX1: the new X1 component.
     * @param coordX2: the new X2 component.
     * @param coordX3: the new X3 component.
     * @param cs: the new coordinate system associated to this coordinate.
     */
    void setCoord(double coordX1, double coordX2, double coordX3, CScene3dCoordSystem cs=CScene3dCoordSystem::UNDEFINED);

    /**
     * @brief Return the coordinate system associated to this coordinate.
     */
    CScene3dCoordSystem getCoordSystem() const;

    /**
     * @brief Change the coordinate system associated to this coordinate.
     * @param cs: the new coordinate system.
     */
    void setCoordSystem(CScene3dCoordSystem cs);

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3dCoord fromJson(const QJsonObject& obj);

protected:
    /**
     * The three components of this coordinate.
     */
    double m_coord[3];

    /**
     * The coordinate system associated to this coordinate.
     */
    CScene3dCoordSystem m_coordSystem;
};


#endif // CSCENE3DCOORD_H
