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

#ifndef CSCENE3DSHAPECIRCLE_H
#define CSCENE3DSHAPECIRCLE_H

#include <memory>
#include <QJsonObject>

#include "CScene3dColor.h"
#include "CScene3dCoord.h"
#include "CScene3dObject.h"


class CScene3dShapeCircle;

/**
 * Alias onto the CScene3dShapeCircle's shared pointer type.
 */
using CScene3dShapeCirclePtr = std::shared_ptr<CScene3dShapeCircle>;


/**
 * @brief The CScene3dShapeCircle class represents a circle into the 3D scene.
 * This circle is defined by the position of its center, a RGB color and a radius.
 */
class CScene3dShapeCircle : public CScene3dObject
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dShapeCircle();

    /**
     * @brief Custom constructor.
     * @param centerX: the X component of the circle's center.
     * @param centerY: the Y component of the circle's center.
     * @param centerZ: the Z component of the circle's center.
     * @param cs: the coordinate system associated to the circle's center.
     * @param colorR: the red component of the circle.
     * @param colorG: the green component of the circle.
     * @param colorB: the blue component of the circle.
     * @param radius: the radius of the circle.
     * @param isVisible: true if the circle should be displayed, false otherwise.
     */
    CScene3dShapeCircle(
        double centerX, double centerY, double centerZ, CScene3dCoordSystem cs,
        double colorR, double colorG, double colorB,
        double radius,
        bool isVisible
    );

    /**
     * @brief Copy constructor.
     */
    CScene3dShapeCircle(const CScene3dShapeCircle &shape);

    /**
     * @brief Assignment operator.
     */
    CScene3dShapeCircle& operator = (const CScene3dShapeCircle& shape);

    /**
     * @brief Return the coordinates of the circle's center.
     */
    const CScene3dCoord& getCenter() const;

    /**
     * @brief Set the coordinates of the circle's center.
     * @param center: the new coordinates of the center.
     */
    void setCenter(const CScene3dCoord& center);

    /**
     * @brief Set the coordinates of the circle's center.
     * @param centerX: the X component of the center.
     * @param centerY: the Y component of the center.
     * @param centerZ: the Z component of the center.
     * @param cs: the coordinate system associated to the center.
     */
    void setCenter(double centerX, double centerY, double centerZ, CScene3dCoordSystem cs);

    /**
     * @brief Return the color of the circle.
     */
    const CScene3dColor& getColor() const;

    /**
     * @brief Set the color of the circle.
     * @param color: the new color.
     */
    void setColor(const CScene3dColor& color);

    /**
     * @brief Set the color of the circle.
     * @param colorR: the red component of the new color.
     * @param colorG: the green component of the new color.
     * @param colorB: the blue component of the new color.
     */
    void setColor(double colorR, double colorG, double colorB);

    /**
     * @brief Return the radius of the circle.
     */
    double getRadius() const;

    /**
     * @brief Change the radius of the circle.
     * @param radius: the new radius.
     */
    void setRadius(double radius);

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const override;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3dShapeCirclePtr fromJson(const QJsonObject& obj);

    /**
     * @brief Static method used to create an new 'CScene3dShapeCircle' instance.
     * @param centerX: the X component of the circle's center.
     * @param centerY: the Y component of the circle's center.
     * @param centerZ: the Z component of the circle's center.
     * @param cs: the coordinate system associated to the circle's center.
     * @param colorR: the red component of the circle.
     * @param colorG: the green component of the circle.
     * @param colorB: the blue component of the circle.
     * @param radius: the radius of the circle.
     * @param isVisible: true if the circle should be displayed, false otherwise.
     * @return Return a shared_ptr of the created instance.
     */
    static CScene3dShapeCirclePtr create(
        double centerX, double centerY, double centerZ, CScene3dCoordSystem cs,
        double colorR, double colorG, double colorB,
        double radius,
        bool isVisible
    );

protected:
    /**
     * Coordinates (x, y,z) of the circle's center.
     */
    CScene3dCoord m_center;

    /**
     * Color (r, g, b) of the circle.
     */
    CScene3dColor m_color;

    /**
     * Radius of the circle.
     */
    double m_radius;
};


#endif // CSCENE3DIO_H
