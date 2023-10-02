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

#ifndef CSCENE3DSHAPEPOINT_H
#define CSCENE3DSHAPEPOINT_H

#include <memory>

#include "CScene3dColor.h"
#include "CScene3dCoord.h"
#include "CScene3dObject.h"


class CScene3dShapePoint;

/**
 * Alias onto the CScene3dShapePoint's shared pointer type.
 */
using CScene3dShapePointPtr = std::shared_ptr<CScene3dShapePoint>;


/**
 * @brief The CScene3dShapePoint class represents a point into the 3D scene.
 * This point is defined by its position, a RGB color and a size. Currently,
 * a point is represented by a full circle.
 */
class CScene3dShapePoint : public CScene3dObject
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dShapePoint();

    /**
     * @brief Custom constructor.
     * @param positionX: the X component of the point's position.
     * @param positionY: the Y component of the point's position.
     * @param positionZ: the Z component of the point's position.
     * @param cs: the coordinate system associated to the point's position.
     * @param colorR: the red component of the point.
     * @param colorG: the green component of the point.
     * @param colorB: the blue component of the point.
     * @param size: the size of the point (for its visual representation).
     * @param isVisible: true if the point should be displayed, false otherwise.
     */
    CScene3dShapePoint(
        double positionX, double positionY, double positionZ, CScene3dCoordSystem cs,
        double colorR, double colorG, double colorB,
        double size,
        bool isVisible
    );

    /**
     * @brief Copy constructor.
     */
    CScene3dShapePoint(const CScene3dShapePoint &shape);

    /**
     * @brief Assignment operator.
     */
    CScene3dShapePoint& operator = (const CScene3dShapePoint& shape);

    /**
     * @brief Return the coordinates of the point's position.
     */
    const CScene3dCoord& getPosition() const;

    /**
     * @brief Set the coordinates of the point.
     * @param position: the new coordinates of the point.
     */
    void setPosition(const CScene3dCoord& position);

    /**
     * @brief Set the coordinates of the point.
     * @param positionX: the X component of the point's position.
     * @param positionY: the Y component of the point's position.
     * @param positionZ: the Z component of the point's position.
     * @param cs: the coordinate system associated to the point's position.
     */
    void setPosition(double positionX, double positionY, double positionZ, CScene3dCoordSystem cs);

    /**
     * @brief Return the color of the point.
     */
    const CScene3dColor& getColor() const;

    /**
     * @brief Set the color of the point.
     * @param color: the new color.
     */
    void setColor(const CScene3dColor& color);

    /**
     * @brief Set the color of the point.
     * @param colorR: the red component of the new color.
     * @param colorG: the green component of the new color.
     * @param colorB: the blue component of the new color.
     */
    void setColor(double colorR, double colorG, double colorB);

    /**
     * @brief Return the size of the point.
     */
    double getSize() const;

    /**
     * @brief Change the size of the point.
     * @param size: the new size.
     */
    void setSize(double size);

    /**
     * @brief Static method used to create an new 'CScene3dShapePoint' instance.
     * @param positionX: the X component of the point's position.
     * @param positionY: the Y component of the point's position.
     * @param positionZ: the Z component of the point's position.
     * @param cs: the coordinate system associated to the point's position.
     * @param colorR: the red component of the point.
     * @param colorG: the green component of the point.
     * @param colorB: the blue component of the point.
     * @param size: the size of the point (for its visual representation).
     * @param isVisible: true if the point should be displayed, false otherwise.
     * @return Return a shared_ptr of the created instance.
     */
    static CScene3dShapePointPtr create(
        double positionX, double positionY, double positionZ, CScene3dCoordSystem cs,
        double colorR, double colorG, double colorB,
        double size,
        bool isVisible
    );

protected:
    /**
     * Position (x, y,z) of the point.
     */
    CScene3dCoord m_position;

    /**
     * Color (r, g, b) of the point.
     */
    CScene3dColor m_color;

    /**
     * Size of the point (for its visual representation).
     */
    double m_size;
};


#endif // CSCENE3DSHAPEPOINT_H
