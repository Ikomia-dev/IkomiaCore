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

#include "CScene3d.h"

#include <QJsonArray>
#include <QJsonObject>


CScene3d::CScene3d() :
    m_lstLayers(std::vector<CScene3dLayer>()),
    m_image2dCoordSystemOrigin{0.0, 0.0, 0.0}
{ }

CScene3d::CScene3d(const CScene3d &scene) :
    m_lstLayers(scene.getLstLayers()),
    m_image2dCoordSystemOrigin(scene.getImage2dCoordSystemOrigin())
{ }

CScene3d& CScene3d::operator = (const CScene3d &scene)
{
    m_lstLayers = scene.getLstLayers();
    m_image2dCoordSystemOrigin = scene.getImage2dCoordSystemOrigin();

    return *this;
}

void CScene3d::clear()
{
    // The list of layers is clear
    m_lstLayers.clear();
}

void CScene3d::setCurrentVisibleLayer(int index)
{
    for(std::size_t i=0; i < m_lstLayers.size(); ++i)
    {
        // Only the layer associated to 'index' is visible -> setVisibility(true)
        // Other layers are hidden -> setVisibility(false)
        m_lstLayers[i].setVisibility(i == index);
    }
}

void CScene3d::setLayerVisibility(int index, bool visibility)
{
    // The visibility of the layer associated to 'index' is changed
    // Other layers are not changed
    m_lstLayers[index].setVisibility(visibility);
}

void CScene3d::addLayer(const CScene3dLayer &layer)
{
    // The new layer is put at the end of the list
    m_lstLayers.push_back(layer);
}

const std::vector<CScene3dLayer>& CScene3d::getLstLayers() const
{
    return m_lstLayers;
}

const CScene3dLayer& CScene3d::getLayer(int index) const
{
    return m_lstLayers[index];
}

CScene3dLayer& CScene3d::getLayer(int index)
{
    return m_lstLayers[index];
}

const CScene3dCoord& CScene3d::getImage2dCoordSystemOrigin() const
{
    return m_image2dCoordSystemOrigin;
}

void CScene3d::setImage2dCoordSystemOrigin(const CScene3dCoord &origin)
{
    m_image2dCoordSystemOrigin = origin;
}

QJsonObject CScene3d::toJson() const
{
    QJsonObject obj;

    QJsonArray lstLayersArray;
    for(auto layer : m_lstLayers)
    {
        lstLayersArray.push_back(layer.toJson());
    }
    obj["layers"] = lstLayersArray;

    obj["image2dCoordSystemOrigin"] = m_image2dCoordSystemOrigin.toJson();

    return obj;
}

CScene3d CScene3d::fromJson(const QJsonObject& obj)
{
    CScene3d scene;

    QJsonArray lstLayersArray = obj["layers"].toArray();
    for(auto layer : lstLayersArray)
    {
        scene.addLayer(
            CScene3dLayer::fromJson(layer.toObject())
        );
    }

    scene.setImage2dCoordSystemOrigin(
        CScene3dCoord::fromJson(obj["image2dCoordSystemOrigin"].toObject())
    );

    return scene;
}
