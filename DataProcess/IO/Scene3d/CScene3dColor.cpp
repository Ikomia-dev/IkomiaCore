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

#include "CScene3dColor.h"


CScene3dColor::CScene3dColor() :
    m_color{0.0, 0.0, 0.0}
{ }

CScene3dColor::CScene3dColor(double colorR, double colorG, double colorB) :
    m_color{colorR, colorG, colorB}
{ }

CScene3dColor::CScene3dColor(const CScene3dColor &color) :
    m_color{color.getColorR(), color.getColorG(), color.getColorB()}
{ }

CScene3dColor& CScene3dColor::operator = (const CScene3dColor &color)
{
    // To avoid invalid self-assignment
    if(this != &color)
    {
        m_color[0] = color.getColorR();
        m_color[1] = color.getColorG();
        m_color[2] = color.getColorB();
    }

    return *this;
}

double CScene3dColor::getColorR() const
{
    return m_color[0];
}

double CScene3dColor::getColorG() const
{
    return m_color[1];
}

double CScene3dColor::getColorB() const
{
    return m_color[2];
}

double CScene3dColor::getScaledColorR(double minValue, double maxValue) const
{
    return minValue + m_color[0]*(maxValue - minValue);
}

double CScene3dColor::getScaledColorG(double minValue, double maxValue) const
{
    return minValue + m_color[1]*(maxValue - minValue);
}

double CScene3dColor::getScaledColorB(double minValue, double maxValue) const
{
    return minValue + m_color[2]*(maxValue - minValue);
}

void CScene3dColor::setColor(double colorR, double colorG, double colorB)
{
    m_color[0] = colorR;
    m_color[1] = colorG;
    m_color[2] = colorB;
}
