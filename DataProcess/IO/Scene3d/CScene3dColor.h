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

#ifndef CSCENE3DCOLOR_H
#define CSCENE3DCOLOR_H

#include <QJsonObject>
#include "DataProcessGlobal.hpp"


/**
 * @brief The CScene3dColor class represents a RGB color.
 * @details This class is used to associate a color for each point of
 * the objects. The three components Red, Green and Blue (RGB) are store
 * with real values.These values must be defined inside the interval [0.0, 1.0].
 */
class DATAPROCESSSHARED_EXPORT CScene3dColor
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dColor();

    /**
     * @brief Custom constructor.
     * @param colorR: the red component, a real value defined in [0.0, 1.0].
     * @param colorG: the green component, a real value defined in [0.0, 1.0].
     * @param colorB: the blue component, a real value defined in [0.0, 1.0].
     */
    CScene3dColor(double colorR, double colorG, double colorB);

    /**
     * @brief Copy constructor.
     */
    CScene3dColor(const CScene3dColor &color);

    /**
     * @brief Assignment operator.
     */
    CScene3dColor& operator = (const CScene3dColor &color);

    /**
     * @brief Return the red component.
     */
    double getColorR() const;

    /**
     * @brief Return the green component.
     */
    double getColorG() const;

    /**
     * @brief Return the blue component.
     */
    double getColorB() const;

    /**
     * @brief Return the red component, scaled inside the interval [minValue, maxValue].
     */
    double getScaledColorR(double minValue, double maxValue) const;

    /**
     * @brief Return the green component, scaled inside the interval [minValue, maxValue].
     */
    double getScaledColorG(double minValue, double maxValue) const;

    /**
     * @brief Return the blue component, scaled inside the interval [minValue, maxValue].
     */
    double getScaledColorB(double minValue, double maxValue) const;

    /**
     * @brief Change the current color.
     * @param colorR: the new red component, a real value defined in [0.0, 1.0].
     * @param colorG: the new green component, a real value defined in [0.0, 1.0].
     * @param colorB: the new blue component, a real value defined in [0.0, 1.0].
     */
    void setColor(double colorR, double colorG, double colorB);

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3dColor fromJson(const QJsonObject& obj);

protected:
    /**
     * The three components red, green and blue.
     */
    double m_color[3];
};

#endif // CSCENE3DCOLOR_H
