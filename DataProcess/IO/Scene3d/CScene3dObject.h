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

#ifndef CSCENE3DOBJECT_H
#define CSCENE3DOBJECT_H

#include <memory>
#include <QJsonObject>


/**
 * @brief This class represents a generic object inside a 3d scene.
 * This class is a base class for concrete objects like images, points, circles,
 * or polygons. It cannot be instantiated: derived class must be used instead.
 * Example of derived class: CScene3dImage2d, CScene3dShapePoint, CScene3dShapeCircle, CScene3dShapePoly...
 */
class CScene3dObject
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dObject();

    /**
     * @brief Custom constructor
     * @param isVisible: visibility of the current object (true = visible / false = hidden)
     */
    CScene3dObject(bool isVisible);

    /**
     * @brief Copy constructor.
     */
    CScene3dObject(const CScene3dObject& obj);

    /**
     * @brief Destructor. This destructor is abstract to forbid instanciation
     * of this class: derived classes must be used instead.
     */
    virtual ~CScene3dObject() = 0;

    /**
     * @brief Assignment operator.
     */
    CScene3dObject& operator = (const CScene3dObject &obj);

    /**
     * @brief Return the current visibility of the object.
     */
    bool isVisible() const;

    /**
     * @brief Set the current visibility of the object.
     * @param isVisible: visibility of the current object (true = visible / false = hidden)
     */
    void setVisibility(bool isVisible);

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    virtual QJsonObject toJson() const;

protected:
    /**
     * Current visibility of the object (true = visible / false = hidden).
     */
    bool m_isVisible;
};

/**
 * Alias onto the CScene3dObject's shared pointer type.
 */
using CScene3dObjectPtr = std::shared_ptr<CScene3dObject>;

#endif // CSCENE3DOBJECT_H
