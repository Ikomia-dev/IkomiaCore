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

#include "CScene3dIO.h"

#include <iostream>
#include <memory>


CScene3dIO::CScene3dIO() :
    CWorkflowTaskIO(IODataType::SCENE_3D, "CScene3dIO")
{ }

CScene3dIO::CScene3dIO(const std::string &name) :
    CWorkflowTaskIO(IODataType::SCENE_3D, name)
{ }

CScene3dIO::CScene3dIO(const CScene3dIO &io) :
    CWorkflowTaskIO(io),
    m_scene3d(io.getScene3d())
{ }

CScene3dIO::CScene3dIO(const CScene3dIO &&io) :
    CWorkflowTaskIO(io),
    m_scene3d(io.getScene3d())
{ }

CScene3dIO& CScene3dIO::operator = (const CScene3dIO &io)
{
    // To avoid invalid self-assignment
    if(this != &io)
    {
        CWorkflowTaskIO::operator = (io);

        m_scene3d = io.getScene3d();
    }

    return *this;
}

CScene3dIO& CScene3dIO::operator = (const CScene3dIO &&io)
{
    // To avoid invalid self-assignment
    if(this != &io)
    {
        CWorkflowTaskIO::operator = (io);

        m_scene3d = io.getScene3d();
    }

    return *this;
}

std::string CScene3dIO::repr() const
{
    std::stringstream s;
    s << "CScene3dIO(" << m_name << ")";
    return s.str();
}

bool CScene3dIO::isDataAvailable() const
{
    // Data are available if there is one or more layers in the 3D scene
    // (event if these layers are empty)
    return (m_scene3d.getLstLayers().size() > 0);
}

const CScene3d& CScene3dIO::getScene3d() const
{
    return m_scene3d;
}

void CScene3dIO::setScene3d(const CScene3d& scene3d)
{
    m_scene3d = scene3d;
}

std::shared_ptr<CScene3dIO> CScene3dIO::clone() const
{
    return std::static_pointer_cast<CScene3dIO>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CScene3dIO::cloneImp() const
{
    return std::shared_ptr<CScene3dIO>(new CScene3dIO(*this));
}
