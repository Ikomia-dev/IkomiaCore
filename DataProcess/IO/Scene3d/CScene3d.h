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

#ifndef CSCENE3D_H
#define CSCENE3D_H

#include <string>
#include <vector>
#include <QJsonObject>

#include "CScene3dCoord.h"
#include "CScene3dLayer.h"


/**
 * @brief The CScene3d class represents a three-dimensional scene, that is to
 * say an object used to manage 2D objects like images or 3D objects like points,
 * circles or polygons.
 * @details This class handles a list of layers. Each layer contains 2D or 3D
 * objects. An 'Image 2D coordinate system origin' is also defined. This is used
 * to make conversion between points expressed in the real 3D coordinates and
 * points expressed in the 2D image coordinates. No visual representation is
 * made by this class.
 */
class CScene3d
{
public:
    /**
     * @brief Default constructor.
     */
    CScene3d();

    /**
     * @brief Copy constructor.
     */
    CScene3d(const CScene3d &scene);

    /**
     * @brief Assignment operator.
     */
    CScene3d& operator = (const CScene3d &scene);

    /**
     * @brief Clear the whole scene.
     */
    void clear();

    /**
     * @brief Change the current visible layer. Other layers are hidden.
     * @param index: index of the new visible layer.
     */
    void setCurrentVisibleLayer(int index);

    /**
     * @brief Change the visibility of a layer. Visibility of the
     * other layers is not changed.
     * @param index: index of the affected layer.
     * @param visibility: new visibility - true: visible / false: hidden.
     */
    void setLayerVisibility(int index, bool visibility);

    /**
     * @brief Add a new layer in the scene. The new layer is put
     * at the end of the list.
     * @param layer: the new layer to add.
     */
    void addLayer(const CScene3dLayer &layer);

    /**
     * @brief Return the list of the scene's layers.
     */
    const std::vector<CScene3dLayer>& getLstLayers() const;

    /**
     * @brief Return the layer associated to the given index.
     * @param index: index of the layer.
     */
    const CScene3dLayer& getLayer(int index) const;

    /**
     * @brief Return the layer associated to the given index.
     * @param index: index of the layer.
     */
    CScene3dLayer& getLayer(int index);

    /**
     * @brief Return the current '2D image coordinate system origin',
     * that is to say the position of the 2D image origin (0, 0)
     * according to the 3D real origin. This relation is used to
     * transform coordinates expressed in the 2D image system into
     * the 3D real system.
     */
    const CScene3dCoord& getImage2dCoordSystemOrigin() const;

    /**
     * @brief Set the current '2D image coordinate system origin',
     * that is to say the position of the 2D image origin (0, 0)
     * according to the 3D real origin. This relation is used to
     * transform coordinates expressed in the 2D image system into
     * the 3D real system.
     * @param origin: the new '2D image coordinate system origin'.
     */
    void setImage2dCoordSystemOrigin(const CScene3dCoord &origin);

    /**
     * @brief Serialize data into a Qt's JSON object.
     */
    QJsonObject toJson() const;

    /**
     * @brief Deserialize data from a Qt's JSON object.
     * @param obj: data to deserialize, must be in a JSON format.
     */
    static CScene3d fromJson(const QJsonObject& obj);

protected:
    /**
     * List of the layers.
     */
    std::vector<CScene3dLayer> m_lstLayers;

    /**
     * 2D image coordinate system origin.
     */
    CScene3dCoord m_image2dCoordSystemOrigin;
};

#endif // CSCENE3D_H
