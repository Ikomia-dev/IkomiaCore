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

#ifndef CSCENE3DVECTORFIELD_H
#define CSCENE3DVECTORFIELD_H

#include <memory>
#include <QJsonObject>

#include "Data/CMat.hpp"
#include "CScene3dObject.h"


class CScene3dVectorField;

/**
 * Alias onto the CScene3dVectorField's shared pointer type.
 */
using CScene3dVectorFieldPtr = std::shared_ptr<CScene3dVectorField>;


/**
 * @brief The CScene3dVectorField class represents a
 */
class DATAPROCESSSHARED_EXPORT CScene3dVectorField : public CScene3dObject
{

public:
    /**
     * @brief Default constructor.
     */
    CScene3dVectorField();

    /**
     * @brief Custom constructor.
     * @param data: a CMat class containing the value of each pixel.
     * @param scaleFactor: value used to scale the whole vector field.
     * @param isVisible: true if the vector field should be displayed, false otherwise.
     */
    CScene3dVectorField(const CMat &data, double scaleFactor, bool isVisible);

    /**
     * @brief Copy constructor.
     */
    CScene3dVectorField(const CScene3dVectorField &vf);

    /**
     * @brief Assignment operator.
     */
    CScene3dVectorField& operator = (const CScene3dVectorField &vf);

    /**
     * @brief Return the vector field's data, a CMat class containing the value of each vector.
     */
    CMat getData() const;

    /**
     * @brief Set the vector field's data.
     * @param data: a CMat class containing the value of each vector.
     */
    void setData(const CMat &data);

    /**
     * @brief Return the vector field's scale factor.
     */
    double getScaleFactor() const;

    /**
     * @brief Set the vector field's scale factor.
     * @param scaleFactor: a double value used to scale the whole vector field.
     */
    void setScaleFactor(double scaleFactor);

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const override;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3dVectorFieldPtr fromJson(const QJsonObject& obj);

    /**
     * @brief Static method used to create an new 'CScene3dVectorField' instance.
     * @param data: a CMat class containing the value of each vector.
     * @param isVisible: true if the vector field should be displayed, false otherwise.
     * @return Return a shared_ptr of the created instance.
     */
    static CScene3dVectorFieldPtr create(
        const CMat &data,
        double scaleFactor,
        bool isVisible
    );


protected:
    /**
     * A CMat class containing the value of each vector.
     */
    CMat m_data;

    /**
     * Global value used to scale the whole vector field.
     */
    double m_scaleFactor;
};

#endif // CSCENE3DVECTORFIELD_H
