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

#ifndef CSCENE3DLAYER_H
#define CSCENE3DLAYER_H

#include <vector>
#include <QJsonObject>
#include "DataProcessGlobal.hpp"
#include "CScene3dObject.h"


/**
 * @brief The CScene3dLayer class represents a layer inside a 3D scene. Multiple layers
 * can be put inside this scene. Each layer can contain multiple objects (images, points, circles, polygons...).
 */
class DATAPROCESSSHARED_EXPORT CScene3dLayer
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3dLayer();

    /**
     * @brief Copy constructor.
     */
    CScene3dLayer(const CScene3dLayer& layer);

    /**
     * @brief Assignment operator.
     */
    CScene3dLayer& operator = (const CScene3dLayer& layer);

    /**
     * @brief Return the current visibility of the layer.
     */
    bool isVisible() const;

    /**
     * @brief Change the current visibility of the layer.
     * @param isVisible: visibility of the current layer (true = visible / false = hidden)
     */
    void setVisibility(bool isVisible);

    /**
     * @brief Add an object inside the layer. This object is put at the end of the list.
     * @param obj: object to add to the layer.
     */
    void addObject(CScene3dObjectPtr obj);

    /**
     * @brief Return the list of the objects contained inside the layer.
     */
    const std::vector<CScene3dObjectPtr>& getLstObjects() const;

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3dLayer fromJson(const QJsonObject& obj);

protected:
    /**
     * Visibility of the layer.
     */
    bool m_isVisible;

    /**
     * List of the objects contained inside the layer.
     */
    std::vector<CScene3dObjectPtr> m_lstObjects;
};

#endif // CSCENE3DLAYER_H
