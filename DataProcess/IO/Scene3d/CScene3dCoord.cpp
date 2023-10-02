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

#include "CScene3dCoord.h"


CScene3dCoord::CScene3dCoord() :
    m_coord{0.0, 0.0, 0.0},
    m_coordSystem(CScene3dCoordSystem::UNDEFINED)
{ }

CScene3dCoord::CScene3dCoord(double coordX1, double coordX2, double coordX3, CScene3dCoordSystem cs) :
    m_coord{coordX1, coordX2, coordX3},
    m_coordSystem(cs)
{ }

CScene3dCoord::CScene3dCoord(const CScene3dCoord &coord) :
    m_coord{coord.getCoordX1(), coord.getCoordX2(), coord.getCoordX3()},
    m_coordSystem(coord.getCoordSystem())
{ }

CScene3dCoord& CScene3dCoord::operator = (const CScene3dCoord &coord)
{
    // To avoid invalid self-assignment
    if(this != &coord)
    {
        m_coord[0] = coord.getCoordX1();
        m_coord[1] = coord.getCoordX2();
        m_coord[2] = coord.getCoordX3();
        m_coordSystem = coord.getCoordSystem();
    }

    return *this;
}

double CScene3dCoord::getCoordX1() const
{
    return m_coord[0];
}

double CScene3dCoord::getCoordX2() const
{
    return m_coord[1];
}

double CScene3dCoord::getCoordX3() const
{
    return m_coord[2];
}

void CScene3dCoord::setCoord(double coordX1, double coordX2, double coordX3, CScene3dCoordSystem cs)
{
    m_coord[0] = coordX1;
    m_coord[1] = coordX2;
    m_coord[2] = coordX3;
    m_coordSystem = cs;
}

CScene3dCoordSystem CScene3dCoord::getCoordSystem() const
{
    return m_coordSystem;
}

void CScene3dCoord::setCoordSystem(CScene3dCoordSystem cs)
{
    m_coordSystem = cs;
}
